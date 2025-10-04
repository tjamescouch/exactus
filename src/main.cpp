#include "gpu/metal/pipeline.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <filesystem>
#include <limits>
#include <cinttypes>

static void die(const char* msg) {
  std::fprintf(stderr, "error: %s\n", msg);
  std::exit(1);
}

static void write_ppm(const std::string& path, uint32_t W, uint32_t H,
                      const std::vector<unsigned char>& rgb)
{
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) die("failed to open ppm for write");
  std::fprintf(f, "P6\n%u %u\n255\n", W, H);
  std::fwrite(rgb.data(), 1, rgb.size(), f);
  std::fclose(f);
}

// Map (ch0, ch1) -> RGB by magnitude+phase (HSV-ish, cheap).
static std::vector<unsigned char> visualize_uv(uint32_t W, uint32_t H,
                                               const std::vector<float>& u,
                                               const std::vector<float>& v)
{
  std::vector<unsigned char> out(W*H*3);
  // Compute max mag for normalization
  float maxmag = 1e-6f;
  for (size_t i=0;i<u.size();++i) {
    float m = std::sqrt(u[i]*u[i] + v[i]*v[i]);
    if (m > maxmag) maxmag = m;
  }
  for (uint32_t y=0;y<H;++y){
    for (uint32_t x=0;x<W;++x){
      size_t i = y*W + x;
      float a = std::atan2(v[i], u[i]); // -pi..pi
      float h = (a + 3.1415926535f) / (2.0f*3.1415926535f); // 0..1
      float m = std::sqrt(u[i]*u[i] + v[i]*v[i]) / maxmag;  // 0..1
      float s = 0.9f;
      // HSV->RGB (approx)
      float c = s * m;
      float hp = h * 6.0f;
      float xcol = c * (1.0f - std::fabs(std::fmod(hp,2.0f) - 1.0f));
      float r=0,g=0,b=0;
      if      (0<=hp && hp<1){ r=c; g=xcol; b=0; }
      else if (1<=hp && hp<2){ r=xcol; g=c; b=0; }
      else if (2<=hp && hp<3){ r=0; g=c; b=xcol; }
      else if (3<=hp && hp<4){ r=0; g=xcol; b=c; }
      else if (4<=hp && hp<5){ r=xcol; g=0; b=c; }
      else                   { r=c; g=0; b=xcol; }
      float m0 = 0.1f; // value floor for visibility
      r += m0; g += m0; b += m0;
      r = std::fmin(std::fmax(r,0.0f),1.0f);
      g = std::fmin(std::fmax(g,0.0f),1.0f);
      b = std::fmin(std::fmax(b,0.0f),1.0f);
      out[3*i+0] = (unsigned char)std::lround(r*255.0f);
      out[3*i+1] = (unsigned char)std::lround(g*255.0f);
      out[3*i+2] = (unsigned char)std::lround(b*255.0f);
    }
  }
  return out;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <width> <height> <steps> [seed] [frame_every]\n", argv[0]);
    return 2;
  }
  const uint32_t W = (uint32_t)std::stoul(argv[1]);
  const uint32_t H = (uint32_t)std::stoul(argv[2]);
  const uint32_t STEPS = (uint32_t)std::stoul(argv[3]);
  uint64_t seed = (argc >=5) ? std::strtoull(argv[4], nullptr, 10) : 0xC0FFEEBADC0DEULL;
  uint32_t frame_every = (argc >=6) ? (uint32_t)std::stoul(argv[5]) : 50;

  SimParams P;
  P.width = W; P.height = H;
  P.dt = 0.1f; P.eta = 0.05f; P.m2 = 0.2f; P.lambda_ = 0.01f;
  P.alpha = {0.6f, 0.3f, 0.15f};
  P.W00 = 0.0f; P.W01 = 0.05f; P.W10 = -0.05f; P.W11 = 0.0f;

  MetalPipeline pipe;
  try {
    pipe.init(W, H, P);
  } catch (const std::exception& e) {
    die(e.what());
  }

  pipe.seed_initial_field(seed, /*amp=*/1.0f, /*count=*/(int)std::max<uint32_t>(64, (W*H)/4096));

  std::filesystem::create_directories("out");

  std::vector<float> ch0, ch1;

  for (uint32_t t=0; t<STEPS; ++t) {
    pipe.step();

    if (t % frame_every == 0 || t+1 == STEPS) {
      pipe.readback(ch0, ch1);
      auto rgb = visualize_uv(W, H, ch0, ch1);
      char name[256];
      std::snprintf(name, sizeof(name), "out/frame_%06u.ppm", t);
      write_ppm(name, W, H, rgb);
      std::fprintf(stdout, "wrote %s\n", name);
    }
  }

  return 0;
}
