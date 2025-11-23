#include <pybind11/numpy.h>
#include "metalsp/network_types.hpp"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <vector>
#include <cstring>
#include <random>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <atomic>

namespace py = pybind11;
namespace metalsp {

// === Global training knobs (atomic for safety) ===
static std::atomic<float> G_LAMBDA{1e-3f};   // ridge/weight decay
static std::atomic<float> G_GMAX{5.0f};      // gradient clamp in apply pass

void set_train_hyperparams(float lambda, float gmax) {
  if (lambda < 0.f) lambda = 0.f;
  if (gmax <= 0.f)  gmax = 1.f;
  G_LAMBDA.store(lambda, std::memory_order_relaxed);
  G_GMAX.store(gmax,   std::memory_order_relaxed);
}

std::vector<float> debug_output_host;

static id<MTLDevice>                 g_device           = nil;
static id<MTLCommandQueue>           g_queue            = nil;
static id<MTLComputePipelineState>   g_pipeline_sgd1    = nil;
static id<MTLComputePipelineState>   g_pipeline_grad    = nil;
static id<MTLComputePipelineState>   g_pipeline_apply   = nil;
static id<MTLComputePipelineState>   g_predict_pipeline = nil;
static id<MTLBuffer>                 g_coefficientsBuf  = nil;

static const char* METAL_SOURCE = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct Meta {
  uint  D, M, m;
  float learning_rate;
  uint  use_lut, n_degree;
  uint  pascal_cols, pascal_rows;
  uint  batch_size;
  float lambda_;    // ridge
  float gmax_;      // gradient clamp
};

inline uint nCr(uint n, uint k, constant uint* P, uint rows, uint cols){
  if(n>=rows||k>=cols) return 0u; return P[n*cols+k];
}

inline void get_monomial_indices_math(uint idx,uint deg,uint D,
  constant uint* P,uint rows,uint cols, thread uint* out){
  uint rem=idx, max_n=min(rows-1, D+deg);
  for(int r=int(deg); r>0; --r){
    uint k=r,low=k,high=max_n,chosen=k;
    while(low<=high){
      uint mid=low+((high-low)>>1);
      uint val=nCr(mid,k,P,rows,cols);
      if(val<=rem){ chosen=mid; low=mid+1; } else { if(mid==0)break; high=mid-1; }
    }
    out[r-1]=chosen-(k-1);
    rem-=nCr(chosen,k,P,rows,cols);
    max_n=(chosen>0?chosen-1:0);
  }
}
inline void get_monomial_indices_lut(uint idx,uint deg, constant uint* lut, thread uint* out){
  uint base=idx*deg; for(uint i=0;i<deg;++i) out[i]=lut[base+i];
}

inline float compute_feature(uint idx,uint deg,uint D,
  constant uint* P,uint rows,uint cols, device const float* x,
  constant uint* lut,uint use_lut){
  uint ind[8]; // raise if deg>8
  if(use_lut==1) get_monomial_indices_lut(idx,deg,lut,ind);
  else get_monomial_indices_math(idx,deg,D,P,rows,cols,ind);
  float v=1.0f; for(uint i=0;i<deg;++i) v*=x[ind[i]]; return v;
}

// ---- Single-sample SGD (bs=1) ----
kernel void sgd1_kernel(
  device float       *coefficients     [[buffer(0)]],
  const device float *all_input_x      [[buffer(1)]],
  const device float *all_target_y     [[buffer(2)]],
  const device uint  *update_indices   [[buffer(3)]],
  constant Meta      *meta             [[buffer(4)]],
  device   float     *debug_output     [[buffer(5)]],
  constant uint      *pascal_table     [[buffer(6)]],
  constant uint      *indices_lut      [[buffer(7)]],
  uint tid_tg [[thread_position_in_threadgroup]],
  uint tgid   [[threadgroup_position_in_grid]],
  uint TPG    [[threads_per_threadgroup]]
){
  const uint N=meta->n_degree;
  threadgroup float tg_sum[64];
  threadgroup float tg_grad;

  const uint sample=tgid;
  device const float* x = all_input_x + sample*meta->D;
  const float y = all_target_y[sample];

  float local=0.0f;
  for(uint i=tid_tg;i<meta->m;i+=TPG){
    uint k=update_indices[i];
    float phi=compute_feature(k,N,meta->D,pascal_table,meta->pascal_rows,meta->pascal_cols,x,indices_lut,meta->use_lut);
    local += coefficients[k]*phi;
  }
  if(tid_tg<64) tg_sum[tid_tg]=local;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if(tid_tg==0){
    float s=0.0f; uint active=(TPG<64?TPG:64);
    for(uint t=0;t<active;++t) s+=tg_sum[t];
    float scale = (meta->m>0)? (float(meta->M)/float(meta->m)) : 0.0f;
    float y_hat = s*scale;
    float err   = y_hat - y;
    tg_grad = 2.0f*err;
    debug_output[0]=y_hat; debug_output[1]=tg_grad;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float g = tg_grad;
  for(uint i=tid_tg;i<meta->m;i+=TPG){
    uint k=update_indices[i];
    float phi=compute_feature(k,N,meta->D,pascal_table,meta->pascal_rows,meta->pascal_cols,x,indices_lut,meta->use_lut);
    float gk = clamp(g*phi, -meta->gmax_, meta->gmax_);
    coefficients[k] -= meta->learning_rate * (gk + meta->lambda_*coefficients[k]);
  }
}

// ---- Two-pass minibatch ----
kernel void grad_kernel(
  device float       *coefficients     [[buffer(0)]],
  const device float *all_input_x      [[buffer(1)]],
  const device float *all_target_y     [[buffer(2)]],
  const device uint  *update_indices   [[buffer(3)]],
  constant Meta      *meta             [[buffer(4)]],
  device   float     *debug_output     [[buffer(5)]],
  constant uint      *pascal_table     [[buffer(6)]],
  constant uint      *indices_lut      [[buffer(7)]],
  device   float     *grad_out         [[buffer(8)]],
  uint tid_tg [[thread_position_in_threadgroup]],
  uint tgid   [[threadgroup_position_in_grid]],
  uint TPG    [[threads_per_threadgroup]]
){
  const uint N=meta->n_degree;
  const uint B=meta->batch_size;
  threadgroup float tg_sum[64];
  threadgroup float tg_grad;

  const uint sample=tgid;
  device const float* x = all_input_x + sample*meta->D;
  const float y = all_target_y[sample];

  float local=0.0f;
  for(uint i=tid_tg;i<meta->m;i+=TPG){
    uint k=update_indices[i];
    float phi=compute_feature(k,N,meta->D,pascal_table,meta->pascal_rows,meta->pascal_cols,x,indices_lut,meta->use_lut);
    local += coefficients[k]*phi;
  }
  if(tid_tg<64) tg_sum[tid_tg]=local;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if(tid_tg==0){
    float s=0.0f; uint active=(TPG<64?TPG:64);
    for(uint t=0;t<active;++t) s+=tg_sum[t];
    float scale = (meta->m>0)? (float(meta->M)/float(meta->m)) : 0.0f;
    float y_hat = s*scale;
    float err   = y_hat - y;
    tg_grad = 2.0f*err;
    if(sample==0){ debug_output[0]=y_hat; debug_output[1]=tg_grad; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint base = sample * meta->m;
  for(uint i=tid_tg;i<meta->m;i+=TPG){
    uint k=update_indices[i];
    float phi=compute_feature(k,N,meta->D,pascal_table,meta->pascal_rows,meta->pascal_cols,x,indices_lut,meta->use_lut);
    grad_out[base + i] = tg_grad * phi; // unclipped here, clip at apply
  }
}

kernel void apply_kernel(
  device float       *coefficients     [[buffer(0)]],
  const device uint  *update_indices   [[buffer(3)]],
  constant Meta      *meta             [[buffer(4)]],
  device   float     *grad_out         [[buffer(8)]],
  uint gid [[thread_position_in_grid]]
){
  const uint i = gid;
  if(i >= meta->m) return;
  const uint B = meta->batch_size;

  float gsum = 0.0f;
  for(uint s=0; s<B; ++s) gsum += grad_out[s*meta->m + i];
  float gmean = (B>0) ? (gsum / float(B)) : 0.0f;

  // clip and apply ridge
  gmean = clamp(gmean, -meta->gmax_, meta->gmax_);

  const uint k = update_indices[i];
  float step = meta->learning_rate * (gmean + meta->lambda_*coefficients[k]);
  coefficients[k] -= step;
}

// ---- Prediction ----
kernel void predict_kernel(
  device float       *coefficients     [[buffer(0)]],
  const device float *all_input_x      [[buffer(1)]],
  device float       *predictions      [[buffer(2)]],
  const device uint  *update_indices   [[buffer(3)]],
  constant Meta      *meta             [[buffer(4)]],
  constant uint      *pascal_table     [[buffer(6)]],
  constant uint      *indices_lut      [[buffer(7)]],
  uint tid_tg [[thread_position_in_threadgroup]],
  uint tgid   [[threadgroup_position_in_grid]],
  uint TPG    [[threads_per_threadgroup]]
){
  const uint N=meta->n_degree;
  threadgroup float tg_sum[64];

  const uint sample=tgid;
  device const float* x = all_input_x + sample*meta->D;

  float local=0.0f;
  for(uint i=tid_tg;i<meta->m;i+=TPG){
    uint k=update_indices[i];
    float phi=compute_feature(k,N,meta->D,pascal_table,meta->pascal_rows,meta->pascal_cols,x,indices_lut,meta->use_lut);
    local += coefficients[k]*phi;
  }
  if(tid_tg<64) tg_sum[tid_tg]=local;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if(tid_tg==0){
    float s=0.0f; uint active=(TPG<64?TPG:64);
    for(uint t=0;t<active;++t) s+=tg_sum[t];
    float scale = (meta->m>0)? (float(meta->M)/float(meta->m)) : 0.0f;
    predictions[sample] = s*scale;
  }
}
)METAL";

struct MetaGPU {
  uint32_t D, M, m;
  float    learning_rate;
  uint32_t use_lut, n_degree;
  uint32_t pascal_cols, pascal_rows;
  uint32_t batch_size;
  float    lambda_;
  float    gmax_;
};

// helpers (nCr_cpu, generate_indices_lut, init_metal) — unchanged from your last version
static long long nCr_cpu(int n,int r){
  if(r<0||r>n) return 0;
  if(r==0||r==n) return 1;
  if(r>n/2) r=n-r;
  long long res=1; for(int i=1;i<=r;++i) res=(res*(n-i+1))/i; return res;
}
static std::vector<uint32_t> generate_indices_lut(uint32_t M,uint32_t N,uint32_t D){
  std::vector<uint32_t> lut; lut.reserve(size_t(M)*N);
  for(uint32_t k=0;k<M;++k){
    uint32_t rem=k, c=D+N;
    for(int r=int(N); r>0; --r){
      while(true){
        long long v=nCr_cpu(int(c), r);
        if(v<=rem){ lut.push_back(c-(r-1)); rem-=uint32_t(v); --c; break; }
        --c;
      }
    }
  }
  return lut;
}
static id<MTLLibrary> buildLib(NSString* src, id<MTLDevice> dev){
  NSError *err=nil; id<MTLLibrary> lib=[dev newLibraryWithSource:src options:nil error:&err];
  if(!lib) throw std::runtime_error("Metal compile failed"); return lib;
}
static void init_metal() {
  if (g_device) return;
  g_device = MTLCreateSystemDefaultDevice();
  g_queue  = [g_device newCommandQueue];
  NSString *src = [NSString stringWithUTF8String:METAL_SOURCE];
  id<MTLLibrary> lib = buildLib(src, g_device);
  NSError *err=nil;
  g_pipeline_sgd1    = [g_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"sgd1_kernel"]   error:&err];
  g_pipeline_grad    = [g_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"grad_kernel"]   error:&err];
  g_pipeline_apply   = [g_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"apply_kernel"]  error:&err];
  g_predict_pipeline = [g_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"predict_kernel"] error:&err];
  if (!g_pipeline_sgd1 || !g_pipeline_grad || !g_pipeline_apply || !g_predict_pipeline)
    throw std::runtime_error("Pipeline create failed");
}

std::vector<float> mc_network_fit(
  const std::vector<float>& X_flat,
  const std::vector<float>& y,
  const std::vector<float>& coefficients_in,
  float learning_rate,
  int   epochs,
  int   batch_size,
  int   degree,
  py::array_t<uint32_t>& pascal_np)
{
  init_metal();

  py::buffer_info bi=pascal_np.request();
  auto* Pptr = static_cast<uint32_t*>(bi.ptr);
  const uint32_t P_elems=(uint32_t)bi.size;
  const uint32_t pascal_cols=(uint32_t)(degree+1);
  const uint32_t pascal_rows=P_elems/pascal_cols;

  const uint32_t N_S = (uint32_t)y.size();
  const uint32_t D   = (uint32_t)(X_flat.size()/std::max<uint32_t>(1,N_S));
  const uint32_t M   = (uint32_t)coefficients_in.size();
  const uint32_t N   = (uint32_t)degree;

  if(!g_coefficientsBuf || [g_coefficientsBuf length] != M*sizeof(float))
    g_coefficientsBuf=[g_device newBufferWithBytes:coefficients_in.data() length:M*sizeof(float) options:MTLResourceStorageModeShared];
  else
    std::memcpy([g_coefficientsBuf contents], coefficients_in.data(), M*sizeof(float));

  id<MTLBuffer> Xbuf   =[g_device newBufferWithBytes:X_flat.data() length:X_flat.size()*sizeof(float) options:MTLResourceStorageModeShared];
  id<MTLBuffer> Ybuf   =[g_device newBufferWithBytes:y.data()      length:y.size()*sizeof(float)      options:MTLResourceStorageModeShared];
  id<MTLBuffer> Pbuf   =[g_device newBufferWithBytes:Pptr          length:P_elems*sizeof(uint32_t)    options:MTLResourceStorageModeShared];
  id<MTLBuffer> dbgBuf =[g_device newBufferWithLength:2*sizeof(float) options:MTLResourceStorageModeShared];

  const bool dense = (M < 1'000'000);
  id<MTLBuffer> idxBuf=nil, lutBuf=nil;
  uint32_t m_updates = dense ? M : std::min<uint32_t>(10000u, M);
  if(dense){
    std::vector<uint32_t> idx(M); std::iota(idx.begin(), idx.end(), 0u);
    idxBuf=[g_device newBufferWithBytes:idx.data() length:M*sizeof(uint32_t) options:MTLResourceStorageModeShared];
    auto lut = generate_indices_lut(M,N,D);
    lutBuf=[g_device newBufferWithBytes:lut.data() length:lut.size()*sizeof(uint32_t) options:MTLResourceStorageModeShared];
  } else {
    idxBuf=[g_device newBufferWithLength:m_updates*sizeof(uint32_t) options:MTLResourceStorageModeShared];
    uint32_t dummy[4]={0,0,0,0};
    lutBuf=[g_device newBufferWithBytes:dummy length:sizeof(dummy) options:MTLResourceStorageModeShared];
  }

  MetaGPU meta{ D, M, m_updates, learning_rate,
                (uint32_t)(dense?1:0), (uint32_t)N,
                pascal_cols, pascal_rows,
                1u,                      // batch_size (patched below)
                G_LAMBDA.load(std::memory_order_relaxed),
                G_GMAX.load(std::memory_order_relaxed) };

  std::mt19937 rng(12345);
  std::uniform_int_distribution<uint32_t> U(0, M-1);
  const uint32_t TPG=64;

  for(int e=0;e<epochs;++e){
    @autoreleasepool {
      for(uint32_t base=0; base<N_S; base += (uint32_t)batch_size){
        uint32_t bs = std::min<uint32_t>((uint32_t)batch_size, N_S - base);
        meta.batch_size = bs;

        if(!dense){
          std::vector<uint32_t> tmp(m_updates);
          for(uint32_t i=0;i<m_updates;++i) tmp[i]=U(rng);
          std::memcpy([idxBuf contents], tmp.data(), m_updates*sizeof(uint32_t));
        }

        if(bs==1){
          id<MTLCommandBuffer> cmd=[g_queue commandBuffer];
          id<MTLComputeCommandEncoder> enc=[cmd computeCommandEncoder];
          [enc setComputePipelineState:g_pipeline_sgd1];
          [enc setBuffer:g_coefficientsBuf offset:0 atIndex:0];
          [enc setBuffer:Xbuf offset: base*D*sizeof(float) atIndex:1];
          [enc setBuffer:Ybuf offset: base*sizeof(float)   atIndex:2];
          [enc setBuffer:idxBuf           offset:0 atIndex:3];
          [enc setBytes:&meta length:sizeof(MetaGPU) atIndex:4];
          [enc setBuffer:dbgBuf offset:0 atIndex:5];
          [enc setBuffer:Pbuf   offset:0 atIndex:6];
          [enc setBuffer:lutBuf offset:0 atIndex:7];
          [enc dispatchThreads:MTLSizeMake(bs*TPG,1,1)
         threadsPerThreadgroup:MTLSizeMake(TPG,1,1)];
          [enc endEncoding];
          [cmd commit];
          [cmd waitUntilCompleted];

          debug_output_host.resize(2);
          std::memcpy(debug_output_host.data(), [dbgBuf contents], 2*sizeof(float));
        } else {
          id<MTLBuffer> gradBuf=[g_device newBufferWithLength:bs*m_updates*sizeof(float)
                                                      options:MTLResourceStorageModeShared];

          // Pass 1
          {
            id<MTLCommandBuffer> cmd=[g_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc=[cmd computeCommandEncoder];
            [enc setComputePipelineState:g_pipeline_grad];
            [enc setBuffer:g_coefficientsBuf offset:0 atIndex:0];
            [enc setBuffer:Xbuf offset: base*D*sizeof(float) atIndex:1];
            [enc setBuffer:Ybuf offset: base*sizeof(float)   atIndex:2];
            [enc setBuffer:idxBuf           offset:0 atIndex:3];
            [enc setBytes:&meta length:sizeof(MetaGPU) atIndex:4];
            [enc setBuffer:dbgBuf offset:0 atIndex:5];
            [enc setBuffer:Pbuf   offset:0 atIndex:6];
            [enc setBuffer:lutBuf offset:0 atIndex:7];
            [enc setBuffer:gradBuf offset:0 atIndex:8];
            [enc dispatchThreads:MTLSizeMake(bs*TPG,1,1)
           threadsPerThreadgroup:MTLSizeMake(TPG,1,1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];

            debug_output_host.resize(2);
            std::memcpy(debug_output_host.data(), [dbgBuf contents], 2*sizeof(float));
          }

          // Pass 2
          {
            id<MTLCommandBuffer> cmd=[g_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc=[cmd computeCommandEncoder];
            [enc setComputePipelineState:g_pipeline_apply];
            [enc setBuffer:g_coefficientsBuf offset:0 atIndex:0];
            [enc setBuffer:idxBuf           offset:0 atIndex:3];
            [enc setBytes:&meta length:sizeof(MetaGPU) atIndex:4];
            [enc setBuffer:gradBuf offset:0 atIndex:8];
            [enc dispatchThreads:MTLSizeMake(m_updates,1,1)
           threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
          }
        }
      }
    }
  }

  std::vector<float> out(M);
  std::memcpy(out.data(), [g_coefficientsBuf contents], M*sizeof(float));
  return out;
}

std::vector<float> mc_network_predict(
  const std::vector<float>& X_flat,
  const std::vector<float>& coefficients,
  int input_dim,
  int degree,
  py::array_t<uint32_t>& pascal_np)
{
  init_metal();

  py::buffer_info bi=pascal_np.request();
  auto* Pptr = static_cast<uint32_t*>(bi.ptr);
  const uint32_t P_elems=(uint32_t)bi.size;
  const uint32_t pascal_cols=(uint32_t)(degree+1);
  const uint32_t pascal_rows=P_elems/pascal_cols;

  const uint32_t D = (uint32_t)input_dim;
  const uint32_t M = (uint32_t)coefficients.size();
  const uint32_t N_S = (uint32_t)(X_flat.size()/D);
  const uint32_t N   = (uint32_t)degree;

  if(!g_coefficientsBuf || [g_coefficientsBuf length] != M*sizeof(float))
    g_coefficientsBuf=[g_device newBufferWithBytes:coefficients.data() length:M*sizeof(float) options:MTLResourceStorageModeShared];
  else
    std::memcpy([g_coefficientsBuf contents], coefficients.data(), M*sizeof(float));

  id<MTLBuffer> Xbuf =[g_device newBufferWithBytes:X_flat.data() length:X_flat.size()*sizeof(float) options:MTLResourceStorageModeShared];

  id<MTLBuffer> Pbuf =[g_device newBufferWithBytes:Pptr length:P_elems*sizeof(uint32_t) options:MTLResourceStorageModeShared];

  const bool use_lut=(M<1'000'000);
  id<MTLBuffer> idxBuf=nil, lutBuf=nil;

  std::vector<uint32_t> idx(M); std::iota(idx.begin(), idx.end(), 0u);
  idxBuf=[g_device newBufferWithBytes:idx.data() length:M*sizeof(uint32_t) options:MTLResourceStorageModeShared];

  if(use_lut){
    auto lut=generate_indices_lut(M,N,D);
    lutBuf=[g_device newBufferWithBytes:lut.data() length:lut.size()*sizeof(uint32_t) options:MTLResourceStorageModeShared];
  } else {
    uint32_t dummy[4]={0,0,0,0};
    lutBuf=[g_device newBufferWithBytes:dummy length:sizeof(dummy) options:MTLResourceStorageModeShared];
  }

  MetaGPU meta{ (uint32_t)D,(uint32_t)M,(uint32_t)M,0.0f,(uint32_t)use_lut,(uint32_t)N,
                pascal_cols,pascal_rows, 1u,
                G_LAMBDA.load(std::memory_order_relaxed),
                G_GMAX.load(std::memory_order_relaxed) };

  id<MTLBuffer> pred =[g_device newBufferWithLength:N_S*sizeof(float) options:MTLResourceStorageModeShared];

  @autoreleasepool {
    id<MTLCommandBuffer> cmd=[g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc=[cmd computeCommandEncoder];
    [enc setComputePipelineState:g_predict_pipeline];
    [enc setBuffer:g_coefficientsBuf offset:0 atIndex:0];
    [enc setBuffer:idxBuf           offset:0 atIndex:3];
    [enc setBytes:&meta length:sizeof(MetaGPU) atIndex:4];
    [enc setBuffer:Pbuf offset:0 atIndex:6];
    [enc setBuffer:lutBuf offset:0 atIndex:7];

    const uint32_t BATCH=1024, TPG=64;
    for(uint32_t base=0;base<N_S;base+=BATCH){
      uint32_t bs=std::min<uint32_t>(BATCH, N_S-base);
      [enc setBuffer:Xbuf  offset: base*D*sizeof(float) atIndex:1];
      [enc setBuffer:pred  offset: base*sizeof(float)   atIndex:2];
      [enc dispatchThreads:MTLSizeMake(bs*TPG,1,1)
     threadsPerThreadgroup:MTLSizeMake(TPG,1,1)];
    }
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
  }

  std::vector<float> out(N_S);
  std::memcpy(out.data(), [pred contents], N_S*sizeof(float));
  return out;
}

void free_gpu_memory(){ g_coefficientsBuf=nil; }

} // namespace metalsp
