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

#include "octnet/gpu/combine.h"
#include "octnet/gpu/gpu.h"

#include <cstdio>
#include <cstdlib>



__global__ void kernel_concat(ot_data_t* out, int n_leafs, const ot_data_t* in1, const ot_data_t* in2, const ot_size_t feature_size_in1, const ot_size_t feature_size_in2, const ot_size_t feature_size_out) {
  CUDA_KERNEL_LOOP(vx_idx, n_leafs) {
    octree_cpy_leaf(in1 + vx_idx * feature_size_in1, feature_size_in1, out + vx_idx * feature_size_out);
    octree_cpy_leaf(in2 + vx_idx * feature_size_in2, feature_size_in2, out + vx_idx * feature_size_out + feature_size_in1);
  }
}

void octree_concat_gpu(const octree* in1, const octree* in2, bool check, octree* out) {
  if(DEBUG) { printf("[DEBUG] octree_concat_gpu\n"); }

  if(check && (!octree_equal_trees_gpu(in1, in2))) {
    printf("ERROR: tree structure of inputs do not match\n");
    exit(-1);
  }

  ot_size_t feature_size_in1 = in1->feature_size;
  ot_size_t feature_size_in2 = in2->feature_size;
  ot_size_t feature_size_out = feature_size_in1 + feature_size_in2;

  octree_resize_gpu(in1->n, in1->grid_depth, in1->grid_height, in1->grid_width, feature_size_out, in1->n_leafs, out);
  octree_cpy_trees_gpu_gpu(in1, out);
  octree_cpy_prefix_leafs_gpu_gpu(in1, out);

  kernel_concat<<<GET_BLOCKS(in1->n_leafs), CUDA_NUM_THREADS>>>(
      out->data, in1->n_leafs, in1->data, in2->data, feature_size_in1, feature_size_in2, feature_size_out
  );
  CUDA_POST_KERNEL_CHECK;
}


template <bool do_grad_in2>
__global__ void kernel_concat_bwd(ot_data_t* grad_in1, ot_data_t* grad_in2, int n_leafs, const ot_data_t* grad_out, const ot_size_t feature_size_in1, const ot_size_t feature_size_in2, const ot_size_t feature_size_out) {
  CUDA_KERNEL_LOOP(vx_idx, n_leafs) {
    octree_cpy_leaf(grad_out + vx_idx * feature_size_out, feature_size_in1, grad_in1 + vx_idx * feature_size_in1);
    if(do_grad_in2) {
      octree_cpy_leaf(grad_out + vx_idx * feature_size_out + feature_size_in1, feature_size_in2, grad_in2 + vx_idx * feature_size_in2);
    }
  }
}

void octree_concat_bwd_gpu(const octree* in1, const octree* in2, const octree* grad_out, bool do_grad_in2, octree* grad_in1, octree* grad_in2) {
  if(DEBUG) { printf("[DEBUG] octree_concat_bwd_gpu\n"); }

  octree_resize_as_gpu(in1, grad_in1);
  octree_cpy_trees_gpu_gpu(in1, grad_in1);
  octree_cpy_prefix_leafs_gpu_gpu(in1, grad_in1);
  
  octree_resize_as_gpu(in2, grad_in2);
  octree_cpy_trees_gpu_gpu(in2, grad_in2);
  octree_cpy_prefix_leafs_gpu_gpu(in2, grad_in2);

  ot_size_t feature_size_in1 = in1->feature_size;
  ot_size_t feature_size_in2 = in2->feature_size;
  ot_size_t feature_size_out = feature_size_in1 + feature_size_in2;

  if(do_grad_in2) {
    kernel_concat_bwd<true><<<GET_BLOCKS(in1->n_leafs), CUDA_NUM_THREADS>>>(
       grad_in1->data, grad_in2->data, in1->n_leafs, grad_out->data, feature_size_in1, feature_size_in2, feature_size_out
    );
  }
  else {
    kernel_concat_bwd<false><<<GET_BLOCKS(in1->n_leafs), CUDA_NUM_THREADS>>>(
       grad_in1->data, grad_in2->data, in1->n_leafs, grad_out->data, feature_size_in1, feature_size_in2, feature_size_out
    );

  }
  CUDA_POST_KERNEL_CHECK;
}







__global__ void kernel_concat_dense(ot_data_t* out, int n_leafs, const octree in1, const ot_data_t* in2, const ot_size_t feature_size1, const ot_size_t feature_size2, const ot_size_t feature_size_out) {
  const int dense_depth = 8 * in1.grid_depth;
  const int dense_height = 8 * in1.grid_height;
  const int dense_width = 8 * in1.grid_width;
  
  CUDA_KERNEL_LOOP(leaf_idx, n_leafs) {
    octree_cpy_leaf(in1.data + leaf_idx * feature_size1, feature_size1, out + leaf_idx * feature_size_out);

    int grid_idx = leaf_idx_to_grid_idx(&in1, leaf_idx);
    const ot_tree_t* tree = octree_get_tree(&in1, grid_idx);

    int cum_n_leafs = in1.prefix_leafs[grid_idx];
    int data_idx = leaf_idx - cum_n_leafs;
    int bit_idx = data_idx_to_bit_idx(tree, data_idx);

    int n,ds,hs,ws;
    int depth = octree_ind_to_dense_ind(&in1, grid_idx, bit_idx, &n, &ds,&hs,&ws);
    int width = width_from_depth(depth);

    for(int f = 0; f < feature_size2; ++f) {
      ot_data_t val = 0;
      for(int d = ds; d < ds+width; ++d) {
      for(int h = hs; h < hs+width; ++h) {
      for(int w = ws; w < ws+width; ++w) {
        val += in2[(((n * feature_size2 + f) * dense_depth + d) * dense_height + h) * dense_width + w];
      }
      }
      }

      out[leaf_idx * feature_size_out + feature_size1 + f] = val / (width*width*width);
    }
  }
}

void octree_concat_dense_gpu(const octree* in1, const ot_data_t* in2, ot_size_t feature_size2, octree* out) {
  if(DEBUG) { printf("[DEBUG] octree_concat_dense_gpu\n"); }

  ot_size_t feature_size1 = in1->feature_size;
  ot_size_t feature_size_out = feature_size1 + feature_size2;

  octree_resize_gpu(in1->n, in1->grid_depth, in1->grid_height, in1->grid_width, feature_size_out, in1->n_leafs, out);
  octree_cpy_trees_gpu_gpu(in1, out);
  octree_cpy_prefix_leafs_gpu_gpu(in1, out);

  kernel_concat_dense<<<GET_BLOCKS(in1->n_leafs), CUDA_NUM_THREADS>>>(
      out->data, in1->n_leafs, *in1, in2, feature_size1, feature_size2, feature_size_out
  );
  CUDA_POST_KERNEL_CHECK;
}


template <bool do_grad_in2>
__global__ void kernel_concat_dense_bwd(ot_data_t* grad_in1, ot_data_t* grad_in2, int n_leafs, const octree grad_out, const ot_size_t feature_size1, const ot_size_t feature_size2, const ot_size_t feature_size_out) {
  const int dense_depth = 8 * grad_out.grid_depth;
  const int dense_height = 8 * grad_out.grid_height;
  const int dense_width = 8 * grad_out.grid_width;

  CUDA_KERNEL_LOOP(leaf_idx, n_leafs) {
    octree_cpy_leaf(grad_out.data + leaf_idx * feature_size_out, feature_size1, grad_in1 + leaf_idx * feature_size1);

    if(do_grad_in2) {
      int grid_idx = leaf_idx_to_grid_idx(&grad_out, leaf_idx);
      const ot_tree_t* tree = octree_get_tree(&grad_out, grid_idx);

      int cum_n_leafs = grad_out.prefix_leafs[grid_idx];
      int data_idx = leaf_idx - cum_n_leafs;
      int bit_idx = data_idx_to_bit_idx(tree, data_idx);

      int n,ds,hs,ws;
      int depth = octree_ind_to_dense_ind(&grad_out, grid_idx, bit_idx, &n, &ds,&hs,&ws);
      int width = width_from_depth(depth);

      for(int f = 0; f < feature_size2; ++f) {
        ot_data_t val = grad_out.data[leaf_idx * grad_out.feature_size + feature_size1 + f];
        for(int d = ds; d < ds+width; ++d) {
        for(int h = hs; h < hs+width; ++h) {
        for(int w = ws; w < ws+width; ++w) {
          grad_in2[(((n * feature_size2 + f) * dense_depth + d) * dense_height + h) * dense_width + w] = val;
        }
        }
        }
      }
    }
  }
}

void octree_concat_dense_bwd_gpu(const octree* in1, const ot_data_t* in2, ot_size_t feature_size2, const octree* grad_out, bool do_grad_in2, octree* grad_in1, ot_data_t* grad_in2) {
  if(DEBUG) { printf("[DEBUG] octree_concat_dense_bwd_gpu\n"); }

  octree_resize_as_gpu(in1, grad_in1);
  octree_cpy_trees_gpu_gpu(in1, grad_in1);
  octree_cpy_prefix_leafs_gpu_gpu(in1, grad_in1);
  
  ot_size_t feature_size1 = in1->feature_size;
  ot_size_t feature_size_out = feature_size1 + feature_size2;

  if(do_grad_in2) {
    kernel_concat_dense_bwd<true><<<GET_BLOCKS(in1->n_leafs), CUDA_NUM_THREADS>>>(
       grad_in1->data, grad_in2, in1->n_leafs, *grad_out, feature_size1, feature_size2, feature_size_out
    );
  }
  else {
    kernel_concat_dense_bwd<false><<<GET_BLOCKS(in1->n_leafs), CUDA_NUM_THREADS>>>(
       grad_in1->data, grad_in2, in1->n_leafs, *grad_out, feature_size1, feature_size2, feature_size_out
    );

  }
  CUDA_POST_KERNEL_CHECK;
}
