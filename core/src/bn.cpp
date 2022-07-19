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

#include "octnet/core/core.h"
#include "octnet/cpu/cpu.h"
#include "octnet/cpu/bn.h"

#define FAST_POW(x, y) pow(x, y)
#define FAST_SQRT(x) sqrt(x)
#define EPS 1e-12

extern "C"
void octree_bn_stat_cpu(const octree* grid, ot_data_t* avgs, ot_data_t* vars) {
  const ot_size_t n_blocks = octree_num_blocks(grid);
  const ot_size_t channels = grid->feature_size;
  
  // first we compute running average and variance
  // need to remember using atomic to avoid race conditions!
  // not sure whether OpenMP makes much sense here ...
  #pragma omp parallel for
  for(int grid_idx = 0; grid_idx < n_blocks; ++grid_idx) {
    ot_tree_t* tree = octree_get_tree(grid, grid_idx);
    ot_data_t* in_data = octree_get_data(grid, grid_idx);
    
    // check L0 split:
    if(!tree_isset_bit(tree, 0)) {
      ot_data_t factor = 8*8*8;
      for (int c = 0; c < channels; ++c) {
        ot_data_t val = in_data[c];
        ot_data_t fval = factor*val; // reduce number of multiplications
        #pragma omp atomic
        avgs[c] += fval;
        #pragma omp atomic
        vars[c] += fval*val;
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
                ot_data_t fval = factor*val; // reduce number of multiplications
                #pragma omp atomic
                avgs[c] += fval;
                #pragma omp atomic
                vars[c] += fval*val;
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
                        ot_data_t fval = factor*val; // reduce number of multiplications
                        #pragma omp atomic
                        avgs[c] += fval;
                        #pragma omp atomic
                        vars[c] += fval*val;
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
                              #pragma omp atomic
                              avgs[c] += val;
                              #pragma omp atomic
                              vars[c] += val*val;
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
  
  const ot_size_t M = 8*grid->grid_depth*8*grid->grid_height*8*grid->grid_width*grid->n;
  for (int c = 0; c < channels; ++c) {
    avgs[c] /= M;
    vars[c] /= M;
    vars[c] -= avgs[c]*avgs[c];
  }
}

extern "C"
void octree_bn_norm_cpu(const octree* grid_in, ot_data_t* avgs, ot_data_t* vars, octree* grid) {
  octree_resize_cpu(grid_in->n, grid_in->grid_depth, grid_in->grid_height, grid_in->grid_width, grid_in->feature_size, grid_in->n_leafs, grid);
  octree_cpy_scalars(grid_in, grid);
  octree_cpy_trees_cpu_cpu(grid_in, grid);
  octree_cpy_prefix_leafs_cpu_cpu(grid_in, grid);
  octree_fill_data_cpu(grid, 0);
  
  const ot_size_t n_blocks = octree_num_blocks(grid_in);
  const ot_size_t channels = grid_in->feature_size;
  
  // first pass for computing statistics
  octree_bn_stat_cpu(grid_in, avgs, vars);
  
  // second pass to compute output
  #pragma omp parallel for
  for(int grid_idx = 0; grid_idx < n_blocks; ++grid_idx) {
    ot_tree_t* tree = octree_get_tree(grid_in, grid_idx);
    ot_data_t* in_data = octree_get_data(grid_in, grid_idx);
    ot_data_t* out_data = octree_get_data(grid, grid_idx);
    
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
  } // for grid_idx
}

extern "C"
void octree_bn_ss_cpu(const octree* grid_in, ot_data_t *gamma, ot_data_t *beta, bool inplace, octree* grid_out) {
  if (!inplace) {
    octree_resize_as_cpu(grid_in, grid_out);
    octree_cpy_scalars(grid_in, grid_out);
    octree_cpy_trees_cpu_cpu(grid_in, grid_out);
    octree_cpy_prefix_leafs_cpu_cpu(grid_in, grid_out);
  }

  const ot_size_t channels = grid_in->feature_size;
  
  #pragma omp parallel for
  for(int leaf_idx = 0; leaf_idx < grid_in->n_leafs; ++leaf_idx) {
    //printf("%d\n", leaf_idx);
    for(int c = 0; c < channels; ++c) {
      ot_data_t val = grid_in->data[leaf_idx * channels + c];
      //printf("%d\n", c);
      grid_out->data[leaf_idx * channels + c] = gamma[c]*val + beta[c];
    }
  }
}

extern "C"
void octree_bn_stat_bwd_cpu(const octree* grid_in, const octree* grad_out, ot_data_t* avgs, ot_data_t* vars, ot_data_t* grad_avgs, ot_data_t* grad_vars) {
  const ot_size_t n_blocks = octree_num_blocks(grid_in);
  const ot_size_t channels = grid_in->feature_size;
  
  ot_data_t* grad_avgs_part = new ot_data_t[channels];
  for (int c = 0; c < channels; ++c) {
    grad_avgs_part[c] = 0;
  }
  
  // remember race conditions
  #pragma omp parallel for
  for(int grid_idx = 0; grid_idx < n_blocks; ++grid_idx) {
    ot_tree_t* tree = octree_get_tree(grid_in, grid_idx);
    ot_data_t* in_data = octree_get_data(grid_in, grid_idx);
    ot_data_t* grad_out_data = octree_get_data(grad_out, grid_idx);
    
    // check L0 split:
    if(!tree_isset_bit(tree, 0)) {
      ot_data_t factor = 8*8*8;
      for (int c = 0; c < channels; ++c) {
        ot_data_t grad = grad_out_data[c];
        ot_data_t val = in_data[c];
        ot_data_t centered = factor*(val - avgs[c]);
        #pragma omp atomic
        grad_avgs[c] += factor*grad;
        #pragma omp atomic
        grad_avgs_part[c] += centered;
        #pragma omp atomic
        grad_vars[c] += grad*centered;
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
                #pragma omp atomic
                grad_avgs[c] += factor*grad;
                #pragma omp atomic
                grad_avgs_part[c] += centered;
                #pragma omp atomic
                grad_vars[c] += grad*centered;
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
                        #pragma omp atomic
                        grad_avgs[c] += factor*grad;
                        #pragma omp atomic
                        grad_avgs_part[c] += centered;
                        #pragma omp atomic
                        grad_vars[c] += grad*centered;
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
                              #pragma omp atomic
                              grad_avgs[c] += grad;
                              #pragma omp atomic
                              grad_avgs_part[c] += centered;
                              #pragma omp atomic
                              grad_vars[c] += grad*centered;
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
  
  const ot_size_t M = 8*grid_in->grid_depth*8*grid_in->grid_height*8*grid_in->grid_width*grid_in->n;
  for (int c = 0; c < channels; ++c) {
    // TODO avoid multiple additions for EPS
    grad_vars[c] *= -0.5f*FAST_POW(vars[c] + EPS, -1.5f);
    grad_avgs[c] *= -1.f/FAST_SQRT(vars[c] + EPS);
    grad_avgs[c] += grad_vars[c]/M*(-2.f)*grad_avgs_part[c];
  }
  
  delete[] grad_avgs_part;
}

extern "C"
void octree_bn_norm_bwd_cpu(const octree* grid_in, const octree* grad_out, ot_data_t* avgs, ot_data_t* vars, octree* grad_in) {
  octree_resize_cpu(grad_out->n, grad_out->grid_depth, grad_out->grid_height, grad_out->grid_width, grad_out->feature_size, grad_out->n_leafs, grad_in);
  octree_cpy_scalars(grad_out, grad_in);
  octree_cpy_trees_cpu_cpu(grad_out, grad_in);
  octree_cpy_prefix_leafs_cpu_cpu(grad_out, grad_in);
  octree_fill_data_cpu(grad_in, 0);
  
  const ot_size_t n_blocks = octree_num_blocks(grid_in);
  const ot_size_t channels = grad_out->feature_size;
  
  // first pass for computing the statistic gradients
  ot_data_t* grad_avgs = new ot_data_t[channels];
  ot_data_t* grad_vars = new ot_data_t[channels];
  
  for (int c = 0; c < channels; ++c) {
    grad_avgs[c] = 0;
    grad_vars[c] = 0;
  }
  
  octree_bn_stat_bwd_cpu(grid_in, grad_out, avgs, vars, grad_avgs, grad_vars);
  
  // pre-compute some of the terms
  ot_data_t* over_vars_eps = new ot_data_t[channels];
  ot_data_t* grad_vars_over_M = new ot_data_t[channels];
  ot_data_t* grad_avgs_over_M = new ot_data_t[channels];
  
  const ot_size_t M = 8*grid_in->grid_depth*8*grid_in->grid_height*8*grid_in->grid_width*grid_in->n;
  for (int c = 0; c < channels; ++c) {
    over_vars_eps[c] = 1.f/FAST_SQRT(vars[c] + EPS);
    grad_vars_over_M[c] = grad_vars[c]*2.f/M;
    grad_avgs_over_M[c] = grad_avgs[c]/M;
  }
  
  // second pass to compute input gradients
  #pragma omp parallel for
  for(int grid_idx = 0; grid_idx < n_blocks; ++grid_idx) {
    ot_tree_t* tree = octree_get_tree(grid_in, grid_idx);
    ot_data_t* in_data = octree_get_data(grid_in, grid_idx);
    ot_data_t* grad_out_data = octree_get_data(grad_out, grid_idx);
    ot_data_t* grad_in_data = octree_get_data(grad_in, grid_idx);
    
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
  } // for grid_idx
  
  // !
  delete[] grad_avgs;
  delete[] grad_vars;
  delete[] over_vars_eps;
  delete[] grad_vars_over_M;
  delete[] grad_avgs_over_M;
}

extern "C"
void octree_bn_ss_bwd_cpu(const octree* grad_out, ot_data_t* gamma, bool inplace, octree* grad_in) {
  if (!inplace) {
    octree_resize_as_cpu(grad_out, grad_in);
    octree_cpy_scalars(grad_out, grad_in);
    octree_cpy_trees_cpu_cpu(grad_out, grad_in);
    octree_cpy_prefix_leafs_cpu_cpu(grad_out, grad_in);
  }

  const ot_size_t channels = grad_out->feature_size;
  
  #pragma omp parallel for
  for(int leaf_idx = 0; leaf_idx < grad_out->n_leafs; ++leaf_idx) {
    for(int c = 0; c < channels; ++c) {
      ot_data_t val = grad_out->data[leaf_idx * channels + c];
      grad_in->data[leaf_idx * channels + c] = gamma[c]*val;
    }
  }
}

extern "C"
void octree_bn_ss_wbwd_cpu(const octree* grid_in, const octree* grad_out, ot_data_t* grad_gamma, ot_data_t* grad_beta) {
  const ot_size_t channels = grad_out->feature_size;  

  #pragma omp parallel for
  for(int leaf_idx = 0; leaf_idx < grad_out->n_leafs; ++leaf_idx) {
    for(int c = 0; c < channels; ++c) {
      ot_data_t grad = grad_out->data[leaf_idx * channels + c];
      ot_data_t val = grid_in->data[leaf_idx * channels + c];
      #pragma omp atomic
      grad_gamma[c] += grad*val;
      #pragma omp atomic
      grad_beta[c] += grad;
    }
  }
}