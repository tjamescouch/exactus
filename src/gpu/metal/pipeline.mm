#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "pipeline.h"
#include <stdexcept>
#include <cstring>
#include <random>

static void ensure(bool ok, const char* msg) {
  if (!ok) throw std::runtime_error(msg);
}

MetalPipeline::~MetalPipeline() {
  // ARC handles Objective-C objects when compiled as OBJCXX with ARC; nothing to do.
}

void MetalPipeline::init(uint32_t w, uint32_t h, const SimParams& p) {
  width_ = w; height_ = h; params_ = p;

  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  ensure(dev != nil, "No Metal device available");
  device_ = dev;

  queue_ = [dev newCommandQueue];
  ensure(queue_ != nil, "Failed to create command queue");

  NSError* err = nil;
  NSURL* url = [NSURL fileURLWithPath:@METALLIB_PATH];
  library_ = [dev newLibraryWithURL:url error:&err];
  ensure(library_ != nil && err == nil, "Failed to load metallib (check METALLIB_PATH)");

  id<MTLFunction> fn = [library_ newFunctionWithName:@"wavelet_step"];
  ensure(fn != nil, "Kernel function 'wavelet_step' not found");

  pso_ = [dev newComputePipelineStateWithFunction:fn error:&err];
  ensure(pso_ != nil && err == nil, "Failed to create compute pipeline state");

  params_buf_ = [dev newBufferWithLength:sizeof(SimParams) options:MTLResourceStorageModeShared];
  ensure(params_buf_ != nil, "Failed to create params buffer");
  update_params_buffer();

  create_textures();
}

void MetalPipeline::create_textures() {
  MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                                                               width:width_
                                                                              height:height_
                                                                           mipmapped:NO];
  td.textureType = MTLTextureType2DArray;
  td.arrayLength = 2; // channels
  td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;

  tex_U_      = [device_ newTextureWithDescriptor:td];
  tex_U_next_ = [device_ newTextureWithDescriptor:td];
  ensure(tex_U_ && tex_U_next_, "Failed to create textures");

  // Zero initialize both
  std::vector<float> zero(width_ * height_, 0.0f);
  for (int slice = 0; slice < 2; ++slice) {
    MTLRegion r = MTLRegionMake2D(0,0,width_,height_);
    [tex_U_ replaceRegion:r mipmapLevel:0 slice:slice withBytes:zero.data() bytesPerRow:width_*sizeof(float) bytesPerImage:zero.size()*sizeof(float)];
    [tex_U_next_ replaceRegion:r mipmapLevel:0 slice:slice withBytes:zero.data() bytesPerRow:width_*sizeof(float) bytesPerImage:zero.size()*sizeof(float)];
  }
}

void MetalPipeline::update_params_buffer() {
  std::memcpy([params_buf_ contents], &params_, sizeof(SimParams));
}

void MetalPipeline::seed_initial_field(uint64_t seed, float amp, int count) {
  // Deterministic sparse impulses on both channels
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<uint32_t> X(0, width_-1), Y(0, height_-1);
  std::uniform_real_distribution<float> A(-amp, amp);

  // Read/modify/write pattern: prepare host planes, then upload.
  std::vector<float> ch0(width_ * height_, 0.0f);
  std::vector<float> ch1(width_ * height_, 0.0f);

  for (int i=0;i<count;i++) {
    uint32_t x = X(rng), y = Y(rng);
    float a0 = A(rng), a1 = A(rng);
    ch0[y*width_ + x] += a0;
    ch1[y*width_ + x] += a1;
  }

  MTLRegion r = MTLRegionMake2D(0,0,width_,height_);
  [tex_U_ replaceRegion:r mipmapLevel:0 slice:0 withBytes:ch0.data() bytesPerRow:width_*sizeof(float) bytesPerImage:ch0.size()*sizeof(float)];
  [tex_U_ replaceRegion:r mipmapLevel:0 slice:1 withBytes:ch1.data() bytesPerRow:width_*sizeof(float) bytesPerImage:ch1.size()*sizeof(float)];
}

void MetalPipeline::swap_surfaces() {
  id<MTLTexture> tmp = tex_U_;
  tex_U_ = tex_U_next_;
  tex_U_next_ = tmp;
}

void MetalPipeline::step() {
  update_params_buffer();

  id<MTLCommandBuffer> cb = [queue_ commandBuffer];
  ensure(cb != nil, "Failed to create command buffer");

  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  ensure(enc != nil, "Failed to create compute encoder");

  [enc setComputePipelineState:pso_];
  [enc setTexture:tex_U_ atIndex:0];
  [enc setTexture:tex_U_next_ atIndex:1];
  [enc setBuffer:params_buf_ offset:0 atIndex:0];

  MTLSize tg = MTLSizeMake(16, 16, 1);
  MTLSize grid = MTLSizeMake((width_  + tg.width -1)/tg.width * tg.width,
                             (height_ + tg.height-1)/tg.height* tg.height,
                             1);
  [enc dispatchThreads:grid threadsPerThreadgroup:tg];
  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];

  swap_surfaces();
}

void MetalPipeline::readback(std::vector<float>& ch0, std::vector<float>& ch1) const {
  ch0.resize(width_*height_);
  ch1.resize(width_*height_);
  MTLRegion r = MTLRegionMake2D(0,0,width_,height_);
  [tex_U_ getBytes:ch0.data() bytesPerRow:width_*sizeof(float) bytesPerImage:ch0.size()*sizeof(float) fromRegion:r mipmapLevel:0 slice:0];
  [tex_U_ getBytes:ch1.data() bytesPerRow:width_*sizeof(float) bytesPerImage:ch1.size()*sizeof(float) fromRegion:r mipmapLevel:0 slice:1];
}
