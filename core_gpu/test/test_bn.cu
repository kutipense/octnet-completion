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

#include <cstdlib>
#include <cstdio>
#include <cmath>

#include "octnet/core/core.h"
#include "octnet/gpu/gpu.h"
#include "octnet/test/objects.h"
#include "octnet/gpu/bn.h"

#define EPS 1e-4

inline void expect(bool should_be_true, const char* message) {
  if (!should_be_true) {
    printf("%s\n", message);
    exit(-1);
  }
}

void test_norm() {
  octree* grid_h = create_test_bn_octree_2x8x8x8x2_fixed();
  octree* grid_d = octree_new_gpu();
  octree_to_gpu(grid_h, grid_d);
  
  ot_data_t avg1 = 0;
  ot_data_t avg2 = 0;
  ot_data_t var1 = 0;
  ot_data_t var2 = 0;
  
  for (int i = 0; i < 60; i += 2) {
    ot_data_t factor = 1;
    if (i < 30) {
      factor = 4*4*4;
    }
    if (i >= 30 && i < 44) {
      factor = 2*2*2;
    }
    
    avg1 += factor*i;
    avg2 += factor*(i + 1);
    var1 += factor*i*i;
    var2 += factor*(i + 1)*(i + 1);
  }
  
  avg1 /= 2*8*8*8;
  avg2 /= 2*8*8*8;
  var1 = var1/(2*8*8*8) - avg1*avg1;
  var2 = var2/(2*8*8*8) - avg2*avg2;
  
  ot_data_t* avgs_h = new ot_data_t[2];
  ot_data_t* vars_h = new ot_data_t[2];
  avgs_h[0] = 0;
  avgs_h[1] = 0;
  vars_h[0] = 0;
  vars_h[1] = 0;
  
  ot_data_t* avgs_d; cudaMalloc((ot_data_t**)&avgs_d, 2*sizeof(ot_data_t));
  ot_data_t* vars_d; cudaMalloc((ot_data_t**)&vars_d, 2*sizeof(ot_data_t));
  cudaMemcpy(avgs_d, avgs_h, 2*sizeof(ot_data_t), cudaMemcpyHostToDevice);
  cudaMemcpy(vars_d, vars_h, 2*sizeof(ot_data_t), cudaMemcpyHostToDevice);
  
  octree* grid_norm_d = octree_new_gpu();
  octree_bn_norm_gpu(grid_d, avgs_d, vars_d, grid_norm_d);
  octree* grid_norm_h = octree_new_cpu();
  octree_to_cpu(grid_norm_d, grid_norm_h);
  
  cudaMemcpy(avgs_h, avgs_d, 2*sizeof(ot_data_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(vars_h, vars_d, 2*sizeof(ot_data_t), cudaMemcpyDeviceToHost);
  
  char* buffer = new char[100];
  sprintf(buffer, "average 1 not right: %f != %f", avg1, avgs_h[0]);
  expect(fabs(avg1 - avgs_h[0]) < EPS, buffer);
  
  sprintf(buffer, "average 2 not right: %f != %f", avg2, avgs_h[1]);
  expect(fabs(avg2 - avgs_h[1]) < EPS, "average 2 not right");
  
  sprintf(buffer, "variance 1 not right: %f != %f", var1, vars_h[0]);
  expect(fabs(var1 - vars_h[0]) < EPS, "variance 1 not right");
  
  sprintf(buffer, "variance 2 not right: %f != %f", var2, vars_h[1]);
  expect(fabs(var2 - vars_h[1]) < EPS, "variance 2 not right");
//  printf("%f %f %f %f %f %f %f %f", avg1, avgs[0], avg2, avgs[1], var1, vars[0], var2, vars[1]);
  
  for(int idx = 0; idx < grid_h->n_leafs; ++idx) {
    for (int c = 0; c < 2; ++c) {
      ot_data_t val = (grid_h->data[idx*2 + c] - avgs_h[c])/sqrt(vars_h[c] + 1e-12);
      sprintf(buffer, "normalization not right (%d, %d): %f != %f", idx, c, grid_norm_h->data[idx*2 + c], val);
      expect(fabs(grid_norm_h->data[idx*2 + c] - val) < EPS, buffer);
    }
  }
  
  octree_free_cpu(grid_h);
  octree_free_cpu(grid_norm_h);
  octree_free_gpu(grid_d);
  octree_free_gpu(grid_norm_d);
  cudaFree(avgs_d);
  cudaFree(vars_d);
  delete[] avgs_h;
  delete[] vars_h;
  delete[] buffer;
}

// Not really good test as all the gradients should be zero.
void test_norm_bwd() {
  const ot_data_t one = 1.f;
  const ot_data_t two = 2.f;
  octree* grid_in_h = create_test_bn_octree_2x8x8x8x2_value(one);
  octree* grad_out_h = create_test_bn_octree_2x8x8x8x2_value(two);
  
  ot_data_t* avgs_h = new ot_data_t[2];
  ot_data_t* vars_h = new ot_data_t[2];
  avgs_h[0] = 5;
  avgs_h[1] = 0.5;
  vars_h[0] = 2.5;
  vars_h[1] = 0.5;
  
  ot_data_t* avgs_d; cudaMalloc((ot_data_t**)&avgs_d, 2*sizeof(ot_data_t));
  ot_data_t* vars_d; cudaMalloc((ot_data_t**)&vars_d, 2*sizeof(ot_data_t));
  cudaMemcpy(avgs_d, avgs_h, 2*sizeof(ot_data_t), cudaMemcpyHostToDevice);
  cudaMemcpy(vars_d, vars_h, 2*sizeof(ot_data_t), cudaMemcpyHostToDevice);
  
  octree* grid_in_d = octree_new_gpu();
  octree* grad_out_d = octree_new_gpu();
  octree* grad_in_d = octree_new_gpu();
  
  octree_to_gpu(grid_in_h, grid_in_d);
  octree_to_gpu(grad_out_h, grad_out_d);
  
  octree_bn_norm_bwd_gpu(grid_in_d, grad_out_d, avgs_d, vars_d, grad_in_d);
  octree* grad_in_h = octree_new_cpu();
  octree_to_cpu(grad_in_d, grad_in_h);

  cudaMemcpy(avgs_h, avgs_d, 2*sizeof(ot_data_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(vars_h, vars_d, 2*sizeof(ot_data_t), cudaMemcpyDeviceToHost);
  
  ot_data_t* dl_davgs = new ot_data_t[2];
  ot_data_t* dl_dvars = new ot_data_t[2];
  ot_data_t* dl_dx = new ot_data_t[2];
  
  for (int c = 0; c < 2; ++c) {
    dl_dvars[c] = 2*8*8*8*two*(one - avgs_h[c])*-.5*pow(vars_h[c] + 1e-12, -1.5f);
    dl_davgs[c] = 2*8*8*8*two*(-1)/sqrt(vars_h[c] + 1e-12) + dl_dvars[c]*(-2)*(one - avgs_h[c]);
    dl_dx[c] = two/sqrt(vars_h[c] + 1e-12) + dl_dvars[c]*2*(one - avgs_h[c])/(2*8*8*8) + dl_davgs[c]/(2*8*8*8);
  }
  
  char* buffer = new char[100];
  for(int idx = 0; idx < grad_in_h->n_leafs; ++idx) {
    for (int c = 0; c < 2; ++c) {
      sprintf(buffer, "gradients not right (%d, %d): %f != %f", idx, c, grad_in_h->data[idx*2 + c], dl_dx[c]);
      expect(fabs(grad_in_h->data[idx*2 + c] - dl_dx[c]) < EPS, buffer);
    }
  }
  
  octree_free_cpu(grid_in_h);
  octree_free_cpu(grad_out_h);
  octree_free_cpu(grad_in_h);
  octree_free_gpu(grad_in_d);
  octree_free_gpu(grad_out_d);
  octree_free_gpu(grid_in_d);
  delete[] avgs_h;
  delete[] vars_h;
  delete[] dl_davgs;
  delete[] dl_dvars;
  delete[] dl_dx;
  cudaFree(avgs_d);
  cudaFree(vars_d);
}

int main(int argc, char** argv) {
  test_norm();
  test_norm_bwd();
  return 0;
}