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
#include "octnet/cpu/math.h"
#include <math.h>
#include <iostream>

#if defined(_OPENMP)
#include <omp.h>
#endif

/// Create octrees from dense feature arrays, i.e. occupation is determined
/// whether one of the features is different from zero and data is set to the features.
/// @author David Stutz
class OctreeCreateFromDenseFeaturesCpu : public OctreeCreateCpu {
public:
  /// Constructor.
  /// \param depth_ dense depth
  /// \param height_ dense height
  /// \param width_ dense width
  /// \param feature_size_ dense feature size (= octree feature size)
  /// \param data_ data array in format dhwc
  OctreeCreateFromDenseFeaturesCpu(ot_size_t depth_, ot_size_t height_, ot_size_t width_, ot_size_t feature_size_, const ot_data_t* data_, ot_data_t tr_dist) : 
      OctreeCreateCpu((depth_ + 7) / 8, (height_ + 7) / 8, (width_ + 7) / 8, feature_size_), 
      depth(depth_), height(height_), width(width_), data(data_), tr_dist(tr_dist) {}

  /// Destructor.
  virtual ~OctreeCreateFromDenseFeaturesCpu() {}
  
  /// Determine if the given position is occupied, i.e. one feature is different
  /// from zero.
  /// @param cx
  /// @param cy
  /// @param cz
  /// @param vd
  /// @param vh
  /// @param vw
  /// @param gd
  /// @param gh
  /// @param gw
  /// @param helper
  /// @return 
  virtual bool is_occupied(float cx, float cy, float cz, float vd, float vh, float vw, int gd, int gh, int gw, OctreeCreateHelperCpu* helper) {
    int d1 = cz - vd/2.f; int d2 = cz + vd/2.f;
    int h1 = cy - vh/2.f; int h2 = cy + vh/2.f;
    int w1 = cx - vw/2.f; int w2 = cx + vw/2.f;

    for(int d = d1; d < d2; ++d) {
      for(int h = h1; h < h2; ++h) {
        for(int w = w1; w < w2; ++w) {
          if(d >= 0 && h >= 0 && w >= 0 && d < depth && h < height && w < width) {
            // std::cout << "features: ";
            bool isnumeric = true;
            ot_data_t prod = 1.0;
            for (int f = 0; isnumeric && f < feature_size; ++f) {
              float val = data[((d*height + h)*width + w)*feature_size + f];
              
              // occupied if at least one of the features is significantly
              // different from zero.
              prod *= val;
              // if(isinf(val)){
              //   isnumeric = false;
              // } 
              // std::cout << val << std::endl;

              // if(val > epsilon || val < -epsilon) {
              //   return true;
              // }
            }
            // if(isnumeric) return true;
            if(prod < tr_dist && prod > -tr_dist) return true;
          }
        }
      }
    }
    
    return false;
  }
  
  /// Get the data for the location, i.e. the features.
  /// @param oc
  /// @param cx
  /// @param cy
  /// @param cz
  /// @param vd
  /// @param vh
  /// @param vw
  /// @param gd
  /// @param gh
  /// @param gw
  /// @param helper
  /// @param dst
  virtual void get_data(bool oc, float cx, float cy, float cz, float vd, float vh, float vw, int gd, int gh, int gw, OctreeCreateHelperCpu* helper, ot_data_t* dst) {
    int d = cz - vd/2.f;
    int h = cy - vh/2.f;
    int w = cx - vw/2.f;
    if(oc){
      for (int f = 0; f < feature_size; ++f) {
        dst[f] = data[((d*height + h)*width + w)*feature_size + f];
      }
    } else {
      for (int f = 0; f < feature_size; ++f) {
        dst[f] = 0;
      }
    }
  }

private:
  const ot_size_t depth;
  const ot_size_t height;
  const ot_size_t width;
  const ot_data_t* data;
  const ot_data_t tr_dist;
};


extern "C"
octree* octree_create_from_dense_features_batch_cpu(const ot_data_t* data, int batch_size, int depth, int height, int width, int feature_size, ot_data_t tr_dist, bool fit, int fit_multiply, bool pack, int n_threads) {
  // create individual octrees
  octree** octrees = new octree*[batch_size];
  // #pragma omp parallel for
  for (int n = 0; n < batch_size; ++n) {
    int offset = depth * height * width * feature_size * n;
    octrees[n] = octree_create_from_dense_features_cpu(data + offset, depth, height, width, feature_size, tr_dist, fit, fit_multiply, pack, n_threads);
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
octree* octree_create_from_dense_features_cpu(const ot_data_t* data, int depth, int height, int width, int feature_size, ot_data_t tr_dist, bool fit, int fit_multiply, bool pack, int n_threads) {
  OctreeCreateFromDenseFeaturesCpu create(depth, height, width, feature_size, data, tr_dist);
  return create(fit, fit_multiply, pack, n_threads);
}


/// Create octrees from dense feature arrays, i.e. occupation is determined
/// whether one of the features is different from zero and data is set to the features.
/// @author David Stutz
class OctreeCreateFromDenseFeaturesCpuInverted : public OctreeCreateCpu {
public:
  /// Constructor.
  /// \param depth_ dense depth
  /// \param height_ dense height
  /// \param width_ dense width
  /// \param feature_size_ dense feature size (= octree feature size)
  /// \param data_ data array in format dhwc
  OctreeCreateFromDenseFeaturesCpuInverted(ot_size_t depth_, ot_size_t height_, ot_size_t width_, ot_size_t feature_size_, const ot_data_t* data_, ot_data_t tr_dist) : 
      OctreeCreateCpu((depth_ + 7) / 8, (height_ + 7) / 8, (width_ + 7) / 8, feature_size_), 
      depth(depth_), height(height_), width(width_), data(data_), tr_dist(tr_dist) {}

  /// Destructor.
  virtual ~OctreeCreateFromDenseFeaturesCpuInverted() {}
  
  /// Determine if the given position is occupied, i.e. one feature is different
  /// from zero.
  /// @param cx
  /// @param cy
  /// @param cz
  /// @param vd
  /// @param vh
  /// @param vw
  /// @param gd
  /// @param gh
  /// @param gw
  /// @param helper
  /// @return 
  virtual bool is_occupied(float cx, float cy, float cz, float vd, float vh, float vw, int gd, int gh, int gw, OctreeCreateHelperCpu* helper) {
    int d1 = cz - vd/2.f; int d2 = cz + vd/2.f;
    int h1 = cy - vh/2.f; int h2 = cy + vh/2.f;
    int w1 = cx - vw/2.f; int w2 = cx + vw/2.f;

    for(int d = d1; d < d2; ++d) {
      for(int h = h1; h < h2; ++h) {
        for(int w = w1; w < w2; ++w) {
          if(d >= 0 && h >= 0 && w >= 0 && d < depth && h < height && w < width) {
            // std::cout << "features: ";
            bool isnumeric = true;
            ot_data_t prod = 1.0;
            for (int f = 0; isnumeric && f < feature_size; ++f) {
              float val = data[((f*depth + d)*height + h)*width + w];
              
              // occupied if at least one of the features is significantly
              // different from zero.
              prod *= val;
              // if(isinf(val)){
              //   isnumeric = false;
              // } 
              // std::cout << val << std::endl;

              // if(val > epsilon || val < -epsilon) {
              //   return true;
              // }
            }
            // if(isnumeric) return true;
            if(prod <= tr_dist && prod >= -tr_dist) return true;
          }
        }
      }
    }
    
    return false;
  }
  
  /// Get the data for the location, i.e. the features.
  /// @param oc
  /// @param cx
  /// @param cy
  /// @param cz
  /// @param vd
  /// @param vh
  /// @param vw
  /// @param gd
  /// @param gh
  /// @param gw
  /// @param helper
  /// @param dst
  virtual void get_data(bool oc, float cx, float cy, float cz, float vd, float vh, float vw, int gd, int gh, int gw, OctreeCreateHelperCpu* helper, ot_data_t* dst) {
    int d = cz - vd/2.f;
    int h = cy - vh/2.f;
    int w = cx - vw/2.f;
    if(oc){
      for (int f = 0; f < feature_size; ++f) {
        dst[f] = data[((f*depth + d)*height + h)*width + w];
      }
    } else {
      for (int f = 0; f < feature_size; ++f) {
        dst[f] = 0;
      }
    }
  }

private:
  const ot_size_t depth;
  const ot_size_t height;
  const ot_size_t width;
  const ot_data_t* data;
  const ot_data_t tr_dist;
};


extern "C"
octree* octree_create_from_dense_features_batch_inverted_cpu(const ot_data_t* data, int batch_size, int depth, int height, int width, int feature_size, ot_data_t tr_dist, bool fit, int fit_multiply, bool pack, int n_threads) {
  // create individual octrees
  octree** octrees = new octree*[batch_size];
 
  #pragma omp parallel for
  for (int n = 0; n < batch_size; ++n) {
    int offset = depth * height * width * feature_size * n;
    octrees[n] = octree_create_from_dense_features_inverted_cpu(data + offset, depth, height, width, feature_size, tr_dist, fit, fit_multiply, pack, n_threads);
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
octree* octree_create_from_dense_features_inverted_cpu(const ot_data_t* data, int depth, int height, int width, int feature_size, ot_data_t tr_dist, bool fit, int fit_multiply, bool pack, int n_threads) {
  OctreeCreateFromDenseFeaturesCpuInverted create(depth, height, width, feature_size, data, tr_dist);
  return create(fit, fit_multiply, pack, n_threads);
}