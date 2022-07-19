// Copyright (c) 2017, The OctNet authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the <organization> nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL OCTNET AUTHORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "octnet/create/create.h"
#include "octnet/cpu/cpu.h"
#include "octnet/cpu/combine.h"

class OctreeCreateFromDenseCpu : public OctreeCreateCpu {
public:
  OctreeCreateFromDenseCpu(ot_size_t depth_, ot_size_t height_, ot_size_t width_, const ot_data_t* data_, int n_ranges_, const ot_data_t* ranges_) : 
      OctreeCreateCpu((depth_ + 7) / 8, (height_ + 7) / 8, (width_ + 7) / 8, 1), 
      depth(depth_), height(height_), width(width_), data(data_), n_ranges(n_ranges_), ranges(ranges_) {}

  virtual ~OctreeCreateFromDenseCpu() {}
  
  virtual bool is_occupied(float cx, float cy, float cz, float vd, float vh, float vw, int gd, int gh, int gw, OctreeCreateHelperCpu* helper) {
    // printf("is_occupied %f,%f,%f, %f,%f,%f\n", cx, cy, cz, vw, vh, vd);
    int d1 = cz - vd/2.f; int d2 = cz + vd/2.f;
    int h1 = cy - vh/2.f; int h2 = cy + vh/2.f;
    int w1 = cx - vw/2.f; int w2 = cx + vw/2.f;
    // printf("  %d,%d, %d,%d, %d,%d, (%d,%d,%d)\n", d1,d2, h1,h2, w1,w2, depth,height,width);

    for(int d = d1; d < d2; ++d) {
      for(int h = h1; h < h2; ++h) {
        for(int w = w1; w < w2; ++w) {
          // printf("    l %d,%d,%d\n", d,h,w);
          if(d >= 0 && h >= 0 && w >= 0 && d < depth && h < height && w < width) {
            float val = data[(d * height + h) * width + w];
            // printf("    lval %f\n", val);
            
            for(int ridx = 0; ridx < n_ranges; ++ridx) {
              if(val >= ranges[ridx*2+0] && val < ranges[ridx*2+1]) {
                return true;
              }
            }
          }
        }
      }
    }
    return false;
  }

  virtual void get_data(bool oc, float cx, float cy, float cz, float vd, float vh, float vw, int gd, int gh, int gw, OctreeCreateHelperCpu* helper, ot_data_t* dst) {
    // printf("get_data %d: %f,%f,%f, %f,%f,%f -> %p\n", oc, cx,cy,cz, vw,vh,vd, dst);
    if(oc) {
      dst[0] = 1;
    }
    else {
      dst[0] = 0;
    }
  }

private:
  const ot_size_t depth;
  const ot_size_t height;
  const ot_size_t width;
  const ot_data_t* data;

  int n_ranges;
  const ot_data_t* ranges;
};


extern "C"
octree* octree_create_from_dense_batch_cpu(const ot_data_t* data, int batch_size, int depth, int height, int width, int n_ranges, const ot_data_t* ranges, bool fit, int fit_multiply, bool pack, int n_threads) {
  // create individual octrees
  octree** octrees = new octree*[batch_size];
  for (int n = 0; n < batch_size; ++n) {
    octrees[n] = octree_create_from_dense_cpu(data + n*depth*height*width, depth, height, width, n_ranges, ranges, fit, fit_multiply, pack, n_threads);
  }
  
  // stack/combine octrees
  octree* ret = octree_new_cpu();
  octree_combine_n_cpu(octrees, batch_size, ret);
  
  // clean up
  for (int n = 0; n < batch_size; ++n) {
    octree_free_cpu(octrees[n]);
  }
  
  return ret;
}


extern "C"
octree* octree_create_from_dense_cpu(const ot_data_t* data, int depth, int height, int width, int n_ranges, const ot_data_t* ranges, bool fit, int fit_multiply, bool pack, int n_threads) {
  OctreeCreateFromDenseCpu create(depth, height, width, data, n_ranges, ranges);
  return create(fit, fit_multiply, pack, n_threads);
}





class OctreeCreateFromDense2Cpu : public OctreeCreateCpu {
public:
  OctreeCreateFromDense2Cpu(ot_size_t depth_, ot_size_t height_, ot_size_t width_, const ot_data_t* occupancy_, const ot_data_t* features_, ot_size_t feature_size_) : 
      OctreeCreateCpu((depth_ + 7) / 8, (height_ + 7) / 8, (width_ + 7) / 8, feature_size_), 
      depth(depth_), height(height_), width(width_), occupancy(occupancy_), features(features_) {}

  virtual ~OctreeCreateFromDense2Cpu() {}
  
  virtual bool is_occupied(float cx, float cy, float cz, float vd, float vh, float vw, int gd, int gh, int gw, OctreeCreateHelperCpu* helper) {
    int d1 = cz - vd/2.f; int d2 = cz + vd/2.f;
    int h1 = cy - vh/2.f; int h2 = cy + vh/2.f;
    int w1 = cx - vw/2.f; int w2 = cx + vw/2.f;

    for(int d = d1; d < d2; ++d) {
      for(int h = h1; h < h2; ++h) {
        for(int w = w1; w < w2; ++w) {
          if(d >= 0 && h >= 0 && w >= 0 && d < depth && h < height && w < width) {
            float val = occupancy[(d * height + h) * width + w];
            if(val != 0) {
              return true;
            }
          }
        }
      }
    }
    return false;
  }

  virtual void get_data(bool oc, float cx, float cy, float cz, float vd, float vh, float vw, int gd, int gh, int gw, OctreeCreateHelperCpu* helper, ot_data_t* dst) {
    int d1 = cz - vd/2.f; int d2 = cz + vd/2.f;
    int h1 = cy - vh/2.f; int h2 = cy + vh/2.f;
    int w1 = cx - vw/2.f; int w2 = cx + vw/2.f;

    for(int f = 0; f < feature_size; ++f) {
      dst[f] = 0;
      for(int d = d1; d < d2; ++d) {
        for(int h = h1; h < h2; ++h) {
          for(int w = w1; w < w2; ++w) {
            if(d >= 0 && h >= 0 && w >= 0 && d < depth && h < height && w < width) {
              dst[f] = features[((f * depth + d) * height + h) * width + w];
            }
          }
        }
      }
      dst[f] /= (d2 - d1) * (h2 - h1) * (w2 - w1);
    }
  }

private:
  const ot_size_t depth;
  const ot_size_t height;
  const ot_size_t width;

  const ot_data_t* occupancy;
  const ot_data_t* features;
};

octree* octree_create_from_dense2_cpu(const ot_data_t* occupancy, const ot_data_t* features, int feature_size, int depth, int height, int width, bool fit, int fit_multiply, bool pack, int n_threads) {
  OctreeCreateFromDense2Cpu create(depth, height, width, occupancy, features, feature_size);
  return create(fit, fit_multiply, pack, n_threads);
}
