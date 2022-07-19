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

#include "octnet/gpu/bn.h"
#include "octnet/gpu/gpu.h"

#define FAST_POW(x, y) pow(x, y)
#define FAST_SQRT(x) sqrt(x)
#define EPS 1e-12

// TODO in principle parallelization could also include the channel!
__global__ void kernel_bn_stat(const octree grid, const ot_size_t n_blocks, ot_data_t* avgs, ot_data_t* vars) {
  const ot_size_t channels = grid.feature_size;
  
  CUDA_KERNEL_LOOP(grid_idx, n_blocks) {
    ot_tree_t* tree = octree_get_tree(&grid, grid_idx);
    ot_data_t* in_data = octree_get_data(&grid, grid_idx);
    
    // check L0 split:
    if(!tree_isset_bit(tree, 0)) {
      ot_data_t factor = 8*8*8;
      for (int c = 0; c < channels; ++c) {
        ot_data_t val = in_data[c];
        ot_data_t fval = factor*val;
        atomicAdd(&avgs[c], fval);
        atomicAdd(&vars[c], fval*val);
      }
    }
    else {

      int bit_idx_l1 = 1;
      for(int bdl1 = 0; bdl1 < 2; ++bdl1) {
        for(int bhl1 = 0; bhl1 < 2; ++bhl1) {
          for(int bwl1 = 0; bwl1 < 2; ++bwl1) {
            
            // check L1 split:
            if(!tree_isset_bit(tree, bit_idx_l1)) {
              int data_idx = tree_data_idx(tree, bit_idx_l1, channels);
              ot_data_t factor = 4*4*4;
              for (int c = 0; c < channels; ++c) {
                ot_data_t val = (in_data + data_idx)[c];
                ot_data_t fval = factor*val;
                atomicAdd(&avgs[c], fval);
                atomicAdd(&vars[c], fval*val);
              }
            }
            else {

              int bit_idx_l2 = tree_child_bit_idx(bit_idx_l1);
              for(int bdl2 = 0; bdl2 < 2; ++bdl2) {
                for(int bhl2 = 0; bhl2 < 2; ++bhl2) {
                  for(int bwl2 = 0; bwl2 < 2; ++bwl2) {
                    
                    // check L2 split:
                    if(!tree_isset_bit(tree, bit_idx_l2)) {
                      int data_idx = tree_data_idx(tree, bit_idx_l2, channels);
                      ot_data_t factor = 2*2*2;
                      for (int c = 0; c < channels; ++c) {
                        ot_data_t val = (in_data + data_idx)[c];
                        ot_data_t fval = factor*val;
                        atomicAdd(&avgs[c], fval);
                        atomicAdd(&vars[c], fval*val);
                      }
                    }
                    else {

                      int bit_idx_l3 = tree_child_bit_idx(bit_idx_l2);
                      for(int bdl3 = 0; bdl3 < 2; ++bdl3) {
                        for(int bhl3 = 0; bhl3 < 2; ++bhl3) {
                          for(int bwl3 = 0; bwl3 < 2; ++bwl3) {
                            int data_idx = tree_data_idx(tree, bit_idx_l3, channels);
                            for (int c = 0; c < channels; ++c) {
                              ot_data_t val = (in_data + data_idx)[c];
                              atomicAdd(&avgs[c], val);
                              atomicAdd(&vars[c], val*val);
                            }
                            
                            bit_idx_l3++;
                          }
                        }
                      }

                    }
                    
                    bit_idx_l2++;
                  }
                }
              } 

            } // else L1
            
            bit_idx_l1++;
          } // for bwl1
        } // for bhl1
      } // for bdl1
    } // else L0
  }
}

__global__ void kernel_bn_stat_norm(const ot_size_t M, ot_data_t* avgs, ot_data_t* vars) {
  const int c = threadIdx.x;
  avgs[c] /= M;
  vars[c] /= M;
  vars[c] -= avgs[c]*avgs[c];
}

__global__ void kernel_bn_norm(const octree grid_in, const ot_size_t n_blocks, ot_data_t* avgs, ot_data_t* vars, octree grid) {
  const ot_size_t channels = grid.feature_size;
  
  CUDA_KERNEL_LOOP(grid_idx, n_blocks) {
    ot_tree_t* tree = octree_get_tree(&grid_in, grid_idx);
    ot_data_t* in_data = octree_get_data(&grid_in, grid_idx);
    ot_data_t* out_data = octree_get_data(&grid, grid_idx);
    
    // check L0 split:
    if(!tree_isset_bit(tree, 0)) {
      for (int c = 0; c < channels; ++c) {
        out_data[c] = (in_data[c] - avgs[c])/FAST_SQRT(vars[c] + EPS);
      }
    }
    else {

      int bit_idx_l1 = 1;
      for(int bdl1 = 0; bdl1 < 2; ++bdl1) {
        for(int bhl1 = 0; bhl1 < 2; ++bhl1) {
          for(int bwl1 = 0; bwl1 < 2; ++bwl1) {
            
            // check L1 split:
            if(!tree_isset_bit(tree, bit_idx_l1)) {
              int data_idx = tree_data_idx(tree, bit_idx_l1, channels);
              for (int c = 0; c < channels; ++c) {
                out_data[data_idx + c] = (in_data[data_idx + c] - avgs[c])/FAST_SQRT(vars[c] + EPS);
              }
            }
            else {

              int bit_idx_l2 = tree_child_bit_idx(bit_idx_l1);
              for(int bdl2 = 0; bdl2 < 2; ++bdl2) {
                for(int bhl2 = 0; bhl2 < 2; ++bhl2) {
                  for(int bwl2 = 0; bwl2 < 2; ++bwl2) {
                    
                    // check L2 split:
                    if(!tree_isset_bit(tree, bit_idx_l2)) {
                      int data_idx = tree_data_idx(tree, bit_idx_l2, channels);
                      for (int c = 0; c < channels; ++c) {
                        out_data[data_idx + c] = (in_data[data_idx + c] - avgs[c])/FAST_SQRT(vars[c] + EPS);
                      }
                    }
                    else {

                      int bit_idx_l3 = tree_child_bit_idx(bit_idx_l2);
                      for(int bdl3 = 0; bdl3 < 2; ++bdl3) {
                        for(int bhl3 = 0; bhl3 < 2; ++bhl3) {
                          for(int bwl3 = 0; bwl3 < 2; ++bwl3) {
                            int data_idx = tree_data_idx(tree, bit_idx_l3, channels);
                            for (int c = 0; c < channels; ++c) {
                              out_data[data_idx + c] = (in_data[data_idx + c] - avgs[c])/FAST_SQRT(vars[c] + EPS);
                            }
                            
                            bit_idx_l3++;
                          }
                        }
                      }

                    }
                    
                    bit_idx_l2++;
                  }
                }
              } 

            } // else L1
            
            bit_idx_l1++;
          } // for bwl1
        } // for bhl1
      } // for bdl1
    } // else L0
  }
}

void octree_bn_norm_gpu(const octree* grid_in, ot_data_t* avgs, ot_data_t* vars, octree* grid) {
  octree_resize_gpu(grid_in->n, grid_in->grid_depth, grid_in->grid_height, grid_in->grid_width, grid_in->feature_size, grid_in->n_leafs, grid);
  octree_cpy_scalars(grid_in, grid);
  octree_cpy_trees_gpu_gpu(grid_in, grid);
  octree_cpy_prefix_leafs_gpu_gpu(grid_in, grid);
  octree_fill_data_gpu(grid, 0);
  
  const ot_size_t n_blocks = octree_num_blocks(grid_in);
  const ot_size_t channels = grid_in->feature_size;
  const ot_size_t M = 8*grid->grid_depth*8*grid->grid_height*8*grid->grid_width*grid->n;

  kernel_bn_stat<<<GET_BLOCKS_T(n_blocks, 512), 512>>>(*grid_in, n_blocks, avgs, vars);
  CUDA_POST_KERNEL_CHECK;
  
  kernel_bn_stat_norm<<<1, channels>>>(M, avgs, vars);
  CUDA_POST_KERNEL_CHECK;
  
  kernel_bn_norm<<<GET_BLOCKS_T(n_blocks, 512), 512>>>(*grid_in, n_blocks, avgs, vars, *grid);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void kernel_bn_ss(const octree grid_in, ot_size_t n_data, ot_data_t* gamma, ot_data_t *beta, octree grid_out) {
  CUDA_KERNEL_LOOP(data_idx, n_data) {
    ot_data_t val = grid_in.data[data_idx];
    ot_size_t c = data_idx%grid_in.feature_size;
    grid_out.data[data_idx] = gamma[c]*val + beta[c];
  }
}

void octree_bn_ss_gpu(const octree* grid_in, ot_data_t *gamma, ot_data_t *beta, bool inplace, octree* grid_out) {
  if (!inplace) {
    octree_resize_as_gpu(grid_in, grid_out);
    octree_cpy_scalars(grid_in, grid_out);
    octree_cpy_trees_gpu_gpu(grid_in, grid_out);
    octree_cpy_prefix_leafs_gpu_gpu(grid_in, grid_out);
  }
  
  const ot_size_t n_data = grid_in->n_leafs*grid_in->feature_size;
  kernel_bn_ss<<<GET_BLOCKS(n_data), CUDA_NUM_THREADS>>>(*grid_in, n_data, gamma, beta, *grid_out);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void kernel_bn_stat_bwd(const octree grid_in, const octree grad_out, const ot_size_t n_blocks, ot_data_t* avgs, ot_data_t* vars, ot_data_t* grad_avgs, ot_data_t* grad_avgs_part, ot_data_t* grad_vars) {
  const ot_size_t channels = grid_in.feature_size;
  
  CUDA_KERNEL_LOOP(grid_idx, n_blocks) {
    ot_tree_t* tree = octree_get_tree(&grid_in, grid_idx);
    ot_data_t* in_data = octree_get_data(&grid_in, grid_idx);
    ot_data_t* grad_out_data = octree_get_data(&grad_out, grid_idx);
    
    // check L0 split:
    if(!tree_isset_bit(tree, 0)) {
      ot_data_t factor = 8*8*8;
      for (int c = 0; c < channels; ++c) {
        ot_data_t grad = grad_out_data[c];
        ot_data_t val = in_data[c];
        ot_data_t centered = factor*(val - avgs[c]);
        atomicAdd(&grad_avgs[c], factor*grad);
        atomicAdd(&grad_avgs_part[c], centered);
        atomicAdd(&grad_vars[c], grad*centered);
      }
    }
    else {

      int bit_idx_l1 = 1;
      for(int bdl1 = 0; bdl1 < 2; ++bdl1) {
        for(int bhl1 = 0; bhl1 < 2; ++bhl1) {
          for(int bwl1 = 0; bwl1 < 2; ++bwl1) {
            
            // check L1 split:
            if(!tree_isset_bit(tree, bit_idx_l1)) {
              int data_idx = tree_data_idx(tree, bit_idx_l1, channels);
              ot_data_t factor = 4*4*4;
              for (int c = 0; c < channels; ++c) {
                ot_data_t grad = grad_out_data[data_idx + c];
                ot_data_t val = in_data[data_idx + c];
                ot_data_t centered = factor*(val - avgs[c]);
                atomicAdd(&grad_avgs[c], factor*grad);
                atomicAdd(&grad_avgs_part[c], centered);
                atomicAdd(&grad_vars[c], grad*centered);
              }
            }
            else {

              int bit_idx_l2 = tree_child_bit_idx(bit_idx_l1);
              for(int bdl2 = 0; bdl2 < 2; ++bdl2) {
                for(int bhl2 = 0; bhl2 < 2; ++bhl2) {
                  for(int bwl2 = 0; bwl2 < 2; ++bwl2) {
                    
                    // check L2 split:
                    if(!tree_isset_bit(tree, bit_idx_l2)) {
                      int data_idx = tree_data_idx(tree, bit_idx_l2, channels);
                      ot_data_t factor = 2*2*2;
                      for (int c = 0; c < channels; ++c) {
                        ot_data_t grad = grad_out_data[data_idx + c];
                        ot_data_t val = in_data[data_idx + c];
                        ot_data_t centered = factor*(val - avgs[c]);
                        atomicAdd(&grad_avgs[c], factor*grad);
                        atomicAdd(&grad_avgs_part[c], centered);
                        atomicAdd(&grad_vars[c], grad*centered);
                      }
                    }
                    else {

                      int bit_idx_l3 = tree_child_bit_idx(bit_idx_l2);
                      for(int bdl3 = 0; bdl3 < 2; ++bdl3) {
                        for(int bhl3 = 0; bhl3 < 2; ++bhl3) {
                          for(int bwl3 = 0; bwl3 < 2; ++bwl3) {
                            int data_idx = tree_data_idx(tree, bit_idx_l3, channels);
                            for (int c = 0; c < channels; ++c) {
                              ot_data_t grad = grad_out_data[data_idx + c];
                              ot_data_t val = in_data[data_idx + c];
                              ot_data_t centered = (val - avgs[c]);
                              atomicAdd(&grad_avgs[c], grad);
                              atomicAdd(&grad_avgs_part[c], centered);
                              atomicAdd(&grad_vars[c], grad*centered);
                            }
                            
                            bit_idx_l3++;
                          }
                        }
                      }

                    }
                    
                    bit_idx_l2++;
                  }
                }
              } 

            } // else L1
            
            bit_idx_l1++;
          } // for bwl1
        } // for bhl1
      } // for bdl1
    } // else L0
  } // for grid_idx
}

__global__ void kernel_bn_stat_norm_bwd(const ot_size_t M, ot_data_t* vars, ot_data_t* grad_avgs, ot_data_t* grad_avgs_part, ot_data_t* grad_vars) {
  const ot_size_t c = threadIdx.x;
  grad_vars[c] *= -0.5f*FAST_POW(vars[c] + EPS, -1.5f);
  grad_avgs[c] *= -1.f/FAST_SQRT(vars[c] + EPS);
  grad_avgs[c] += grad_vars[c]/M*(-2.f)*grad_avgs_part[c];
}

__global__ void kernel_bn_norm_init_bwd(const ot_size_t M, ot_data_t* vars, ot_data_t* grad_avgs, ot_data_t* grad_vars, ot_data_t* over_vars_eps, ot_data_t* grad_vars_over_M, ot_data_t* grad_avgs_over_M) {
  const ot_size_t c = threadIdx.x;
  over_vars_eps[c] = 1.f/FAST_SQRT(vars[c] + EPS);
  grad_vars_over_M[c] = grad_vars[c]*2.f/M;
  grad_avgs_over_M[c] = grad_avgs[c]/M;
}

__global__ void kernel_bn_norm_bwd(const octree grid_in, const octree grad_out, const ot_size_t n_blocks, ot_data_t* avgs, ot_data_t* grad_avgs_over_M, ot_data_t* grad_vars_over_M, ot_data_t* over_vars_eps, octree grad_in) {
  const ot_size_t channels = grid_in.feature_size;
  
  CUDA_KERNEL_LOOP(grid_idx, n_blocks) {
    ot_tree_t* tree = octree_get_tree(&grid_in, grid_idx);
    ot_data_t* in_data = octree_get_data(&grid_in, grid_idx);
    ot_data_t* grad_out_data = octree_get_data(&grad_out, grid_idx);
    ot_data_t* grad_in_data = octree_get_data(&grad_in, grid_idx);
    
    // check L0 split:
    if(!tree_isset_bit(tree, 0)) {
      for (int c = 0; c < channels; ++c) {
        grad_in_data[c] = grad_out_data[c]*over_vars_eps[c] 
            + grad_vars_over_M[c]*(in_data[c] - avgs[c]) + grad_avgs_over_M[c];
      }
    }
    else {

      int bit_idx_l1 = 1;
      for(int bdl1 = 0; bdl1 < 2; ++bdl1) {
        for(int bhl1 = 0; bhl1 < 2; ++bhl1) {
          for(int bwl1 = 0; bwl1 < 2; ++bwl1) {
            
            // check L1 split:
            if(!tree_isset_bit(tree, bit_idx_l1)) {
              int data_idx = tree_data_idx(tree, bit_idx_l1, channels);
              for (int c = 0; c < channels; ++c) {
                grad_in_data[data_idx + c] = grad_out_data[data_idx + c]*over_vars_eps[c] 
                    + grad_vars_over_M[c]*(in_data[data_idx + c] - avgs[c]) + grad_avgs_over_M[c];
              }
            }
            else {

              int bit_idx_l2 = tree_child_bit_idx(bit_idx_l1);
              for(int bdl2 = 0; bdl2 < 2; ++bdl2) {
                for(int bhl2 = 0; bhl2 < 2; ++bhl2) {
                  for(int bwl2 = 0; bwl2 < 2; ++bwl2) {
                    
                    // check L2 split:
                    if(!tree_isset_bit(tree, bit_idx_l2)) {
                      int data_idx = tree_data_idx(tree, bit_idx_l2, channels);
                      for (int c = 0; c < channels; ++c) {
                        grad_in_data[data_idx + c] = grad_out_data[data_idx + c]*over_vars_eps[c] 
                            + grad_vars_over_M[c]*(in_data[data_idx + c] - avgs[c]) + grad_avgs_over_M[c];
                      }
                    }
                    else {

                      int bit_idx_l3 = tree_child_bit_idx(bit_idx_l2);
                      for(int bdl3 = 0; bdl3 < 2; ++bdl3) {
                        for(int bhl3 = 0; bhl3 < 2; ++bhl3) {
                          for(int bwl3 = 0; bwl3 < 2; ++bwl3) {
                            int data_idx = tree_data_idx(tree, bit_idx_l3, channels);
                            for (int c = 0; c < channels; ++c) {
                              grad_in_data[data_idx + c] = grad_out_data[data_idx + c]*over_vars_eps[c] 
                                  + grad_vars_over_M[c]*(in_data[data_idx + c] - avgs[c]) + grad_avgs_over_M[c];
                            }
                            
                            bit_idx_l3++;
                          }
                        }
                      }

                    }
                    
                    bit_idx_l2++;
                  }
                }
              } 

            } // else L1
            
            bit_idx_l1++;
          } // for bwl1
        } // for bhl1
      } // for bdl1
    } // else L0
  }
}

void octree_bn_norm_bwd_gpu(const octree* grid_in, const octree* grad_out, ot_data_t* avgs, ot_data_t* vars, octree* grad_in) {
  octree_resize_gpu(grad_out->n, grad_out->grid_depth, grad_out->grid_height, grad_out->grid_width, grad_out->feature_size, grad_out->n_leafs, grad_in);
  octree_cpy_scalars(grad_out, grad_in);
  octree_cpy_trees_gpu_gpu(grad_out, grad_in);
  octree_cpy_prefix_leafs_gpu_gpu(grad_out, grad_in);
  octree_fill_data_gpu(grad_in, 0);
  
  const ot_size_t n_blocks = octree_num_blocks(grid_in);
  const ot_size_t channels = grid_in->feature_size;
  
  // Alloc arrays for gradients wrt average/variance directly on GPU:
  ot_data_t* grad_avgs; DEVICE_MALLOC(grad_avgs, channels); 
  ot_data_t* grad_avgs_part; DEVICE_MALLOC(grad_avgs_part, channels); 
  ot_data_t* grad_vars; DEVICE_MALLOC(grad_vars, channels); 
  
  DEVICE_MEMSET(grad_avgs, 0, channels);
  DEVICE_MEMSET(grad_avgs_part, 0, channels);
  DEVICE_MEMSET(grad_vars, 0, channels);
  
  kernel_bn_stat_bwd<<<GET_BLOCKS_T(n_blocks, 512), 512>>>(*grid_in, *grad_out, n_blocks, avgs, vars, grad_avgs, grad_avgs_part, grad_vars);
  CUDA_POST_KERNEL_CHECK;
//  printf("octree_bn_norm_bwd_gpu kernel_bn_stat_bwd finished\n");
  
  const ot_size_t M = 8*grid_in->grid_depth*8*grid_in->grid_height*8*grid_in->grid_width*grid_in->n;
  kernel_bn_stat_norm_bwd<<<1, channels>>>(M, vars, grad_avgs, grad_avgs_part, grad_vars);
  CUDA_POST_KERNEL_CHECK;
//  printf("octree_bn_norm_bwd_gpu kernel_bn_stat_norm_bwd finished\n");
   
  // Alloc arrays for helper variables to reduce computation directly on GPU:
  ot_data_t* over_vars_eps; DEVICE_MALLOC(over_vars_eps, channels);
  ot_data_t* grad_avgs_over_M; DEVICE_MALLOC(grad_avgs_over_M, channels);
  ot_data_t* grad_vars_over_M; DEVICE_MALLOC(grad_vars_over_M, channels);
  
//  DEVICE_MEMSET(over_vars_eps, 0, channels);
//  DEVICE_MEMSET(grad_avgs_over_M, 0, channels);
//  DEVICE_MEMSET(grad_vars_over_M, 0, channels);
  
  kernel_bn_norm_init_bwd<<<1, channels>>>(M, vars, grad_avgs, grad_vars, over_vars_eps, grad_vars_over_M, grad_avgs_over_M);
  CUDA_POST_KERNEL_CHECK;
//  printf("octree_bn_norm_bwd_gpu kernel_bn_norm_init_bwd finished\n");
  
  kernel_bn_norm_bwd<<<GET_BLOCKS_T(n_blocks, 512), 512>>>(*grid_in, *grad_out, n_blocks, avgs, grad_avgs_over_M, grad_vars_over_M, over_vars_eps, *grad_in);
  CUDA_POST_KERNEL_CHECK;
//  printf("octree_bn_norm_bwd_gpu kernel_bn_norm_bwd finished\n");
  
  DEVICE_FREE(grad_avgs);
  DEVICE_FREE(grad_avgs_part);
  DEVICE_FREE(grad_vars);
  DEVICE_FREE(over_vars_eps);
  DEVICE_FREE(grad_avgs_over_M);
  DEVICE_FREE(grad_vars_over_M);
}
__global__ void kernel_bn_ss_bwd(const octree grad_out, ot_size_t n_data, ot_data_t* gamma, octree grad_in) {
  CUDA_KERNEL_LOOP(data_idx, n_data) {
    ot_data_t val = grad_out.data[data_idx];
    ot_size_t c = data_idx%grad_out.feature_size;
    grad_in.data[data_idx] = gamma[c]*val;
  }
}

void octree_bn_ss_bwd_gpu(const octree* grad_out, ot_data_t* gamma, bool inplace, octree* grad_in) {
  if (!inplace) {
    octree_resize_as_gpu(grad_out, grad_in);
    octree_cpy_scalars(grad_out, grad_in);
    octree_cpy_trees_gpu_gpu(grad_out, grad_in);
    octree_cpy_prefix_leafs_gpu_gpu(grad_out, grad_in);
  }

  const ot_size_t n_data = grad_out->n_leafs*grad_out->feature_size;
  kernel_bn_ss_bwd<<<GET_BLOCKS(n_data), CUDA_NUM_THREADS>>>(*grad_out, n_data, gamma, *grad_in);
}

__global__ void kernel_bn_ss_wbwd(const octree grid_in, const octree grad_out, ot_size_t n_data, ot_data_t* grad_gamma, ot_data_t* grad_beta) {
  CUDA_KERNEL_LOOP(data_idx, n_data) {
    ot_data_t grad = grad_out.data[data_idx];
    ot_data_t val = grid_in.data[data_idx];
    ot_size_t c = data_idx%grid_in.feature_size;
    atomicAdd(&grad_gamma[c], grad*val);
    atomicAdd(&grad_beta[c], grad);
  }
}

void octree_bn_ss_wbwd_gpu(const octree* grid_in, const octree* grad_out, ot_data_t* grad_gamma, ot_data_t* grad_beta) {
  const ot_size_t n_data = grid_in->n_leafs*grid_in->feature_size;  
  
  kernel_bn_ss_wbwd<<<GET_BLOCKS(n_data), CUDA_NUM_THREADS>>>(*grid_in, *grad_out, n_data, grad_gamma, grad_beta);
  CUDA_POST_KERNEL_CHECK;
}