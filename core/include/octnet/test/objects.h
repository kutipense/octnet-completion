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

#ifndef OCTREE_TEST_OBJECTS
#define OCTREE_TEST_OBJECTS

#include "octnet/cpu/cpu.h"

float randf() {
  return float(rand()) / float(RAND_MAX);
}

/// Creates a random test octree instance with the given shape.
///
/// @param gn batch size, number of shallow octree grids.
/// @param gd number of shallow octrees in depth dimension.
/// @param gh number of shallow octrees in depth dimension.
/// @param gw number of shallow octrees in depth dimension.
/// @param fs feature size.
/// @param sp0 split probability of the shallow octree cell on depth 0.
/// @param sp1 split probability of the shallow octree cell on depth 1.
/// @param sp2 split probability of the shallow octree cell on depth 2.
/// @param min_val minimum value for the random octree data.
/// @param max_val maximum value for the random octree data.
/// @return an octree*.
octree* create_test_octree_rand(int gn, int gd, int gh, int gw, int fs, float sp0, float sp1, float sp2, float min_val=-1.f, float max_val=1.f) {
  octree* grid = octree_new_cpu();
  octree_resize_cpu(gn, gd, gh, gw, fs, 0, grid);
  
  octree_clr_trees_cpu(grid);
  for(int grid_idx = 0; grid_idx < octree_num_blocks(grid); ++grid_idx) {
    ot_tree_t* tree = octree_get_tree(grid, grid_idx);
    if(randf() <= sp0) {
      tree_set_bit(tree, 0);
      for(int bit_idx_l1 = 1; bit_idx_l1 <= 8; ++bit_idx_l1) {
        if(randf() <= sp1) {
          tree_set_bit(tree, bit_idx_l1);
          int bit_idx_l2 = tree_child_bit_idx(bit_idx_l1);
          for(int idx_l2 = 0; idx_l2 < 8; ++idx_l2) {
            if(randf() <= sp2) {
              tree_set_bit(tree, bit_idx_l2);
            }
            bit_idx_l2++; 
          }
        }
      }
    }
  }

  octree_upd_n_leafs_cpu(grid);
  octree_resize_as_cpu(grid, grid);
  octree_upd_prefix_leafs_cpu(grid);

  for(int idx = 0; idx < grid->n_leafs * grid->feature_size; ++idx) {
    grid->data[idx] = randf() * (max_val - min_val) + min_val;
  }
  
  return grid;
}

/// Creates a fixed Octree of size 8x8x8 with 2 channels and batch size 2.
/// This octres is used to test batch normalization forward pass and looks as follows:
/// block   0
/// |- split node [0,8],[0,8],[0,8] 0
///   |- data node [0,4],[0,4],[0,4] 1 -> 0, 0x614cd0: [0.000000, 1.000000]
///   |- data node [0,4],[0,4],[4,8] 2 -> 2, 0x614cd8: [2.000000, 3.000000]
///   |- data node [0,4],[4,8],[0,4] 3 -> 4, 0x614ce0: [4.000000, 5.000000]
///   |- data node [0,4],[4,8],[4,8] 4 -> 6, 0x614ce8: [6.000000, 7.000000]
///   |- data node [4,8],[0,4],[0,4] 5 -> 8, 0x614cf0: [8.000000, 9.000000]
///   |- data node [4,8],[0,4],[4,8] 6 -> 10, 0x614cf8: [10.000000, 11.000000]
///   |- data node [4,8],[4,8],[0,4] 7 -> 12, 0x614d00: [12.000000, 13.000000]
///   |- data node [4,8],[4,8],[4,8] 8 -> 14, 0x614d08: [14.000000, 15.000000]
/// block   1
/// |- split node [0,8],[0,8],[0,8] 0
///   |- split node [0,4],[0,4],[0,4] 1
///     |- split node [0,2],[0,2],[0,2] 9
///         |- data node [0,1],[0,1],[0,1] 73 -> 28, 0x614d80: [44.000000, 45.000000]
///         |- data node [0,1],[0,1],[1,2] 74 -> 30, 0x614d88: [46.000000, 47.000000]
///         |- data node [0,1],[1,2],[0,1] 75 -> 32, 0x614d90: [48.000000, 49.000000]
///         |- data node [0,1],[1,2],[1,2] 76 -> 34, 0x614d98: [50.000000, 51.000000]
///         |- data node [1,2],[0,1],[0,1] 77 -> 36, 0x614da0: [52.000000, 53.000000]
///         |- data node [1,2],[0,1],[1,2] 78 -> 38, 0x614da8: [54.000000, 55.000000]
///         |- data node [1,2],[1,2],[0,1] 79 -> 40, 0x614db0: [56.000000, 57.000000]
///         |- data node [1,2],[1,2],[1,2] 80 -> 42, 0x614db8: [58.000000, 59.000000]
///     |- data node [0,2],[0,2],[2,4] 10 -> 14, 0x614d48: [30.000000, 31.000000]
///     |- data node [0,2],[2,4],[0,2] 11 -> 16, 0x614d50: [32.000000, 33.000000]
///     |- data node [0,2],[2,4],[2,4] 12 -> 18, 0x614d58: [34.000000, 35.000000]
///     |- data node [2,4],[0,2],[0,2] 13 -> 20, 0x614d60: [36.000000, 37.000000]
///     |- data node [2,4],[0,2],[2,4] 14 -> 22, 0x614d68: [38.000000, 39.000000]
///     |- data node [2,4],[2,4],[0,2] 15 -> 24, 0x614d70: [40.000000, 41.000000]
///     |- data node [2,4],[2,4],[2,4] 16 -> 26, 0x614d78: [42.000000, 43.000000]
///   |- data node [0,4],[0,4],[4,8] 2 -> 0, 0x614d10: [16.000000, 17.000000]
///   |- data node [0,4],[4,8],[0,4] 3 -> 2, 0x614d18: [18.000000, 19.000000]
///   |- data node [0,4],[4,8],[4,8] 4 -> 4, 0x614d20: [20.000000, 21.000000]
///   |- data node [4,8],[0,4],[0,4] 5 -> 6, 0x614d28: [22.000000, 23.000000]
///   |- data node [4,8],[0,4],[4,8] 6 -> 8, 0x614d30: [24.000000, 25.000000]
///   |- data node [4,8],[4,8],[0,4] 7 -> 10, 0x614d38: [26.000000, 27.000000]
///   |- data node [4,8],[4,8],[4,8] 8 -> 12, 0x614d40: [28.000000, 29.000000]
/// @return octree
octree* create_test_bn_octree_2x8x8x8x2_fixed() {
  octree* grid = octree_new_cpu();
  octree_resize_cpu(2, 1, 1, 1, 2, 0, grid);
  
  octree_clr_trees_cpu(grid);
  for(int grid_idx = 0; grid_idx < octree_num_blocks(grid); ++grid_idx) {
    ot_tree_t* tree = octree_get_tree(grid, grid_idx);
    tree_set_bit(tree, 0);
    if (grid_idx == 1) {
      for(int bit_idx_l1 = 1; bit_idx_l1 <= 1; ++bit_idx_l1) {
        tree_set_bit(tree, bit_idx_l1);
        int bit_idx_l2 = tree_child_bit_idx(bit_idx_l1);
        for(int idx_l2 = 0; idx_l2 < 1; ++idx_l2) {
          tree_set_bit(tree, bit_idx_l2);
          bit_idx_l2++; 
        }
      }
    }
  }

  octree_upd_n_leafs_cpu(grid);
  octree_resize_as_cpu(grid, grid);
  octree_upd_prefix_leafs_cpu(grid);

  for(int idx = 0; idx < grid->n_leafs; ++idx) {
    for (int c = 0; c < 2; ++c) {
      grid->data[idx*2 + c] = 2*idx + c;
    }
  }
  
  return grid;
}

/// Creates the same octree structure as create_test_bn_octree_2x8x8x8x2_fixed
/// but filled with ones.
/// @return octree
octree* create_test_bn_octree_2x8x8x8x2_value(ot_data_t value) {
  octree* grid = octree_new_cpu();
  octree_resize_cpu(2, 1, 1, 1, 2, 0, grid);
  
  octree_clr_trees_cpu(grid);
  for(int grid_idx = 0; grid_idx < octree_num_blocks(grid); ++grid_idx) {
    ot_tree_t* tree = octree_get_tree(grid, grid_idx);
    tree_set_bit(tree, 0);
    if (grid_idx == 1) {
      for(int bit_idx_l1 = 1; bit_idx_l1 <= 1; ++bit_idx_l1) {
        tree_set_bit(tree, bit_idx_l1);
        int bit_idx_l2 = tree_child_bit_idx(bit_idx_l1);
        for(int idx_l2 = 0; idx_l2 < 1; ++idx_l2) {
          tree_set_bit(tree, bit_idx_l2);
          bit_idx_l2++; 
        }
      }
    }
  }

  octree_upd_n_leafs_cpu(grid);
  octree_resize_as_cpu(grid, grid);
  octree_upd_prefix_leafs_cpu(grid);

  for(int idx = 0; idx < grid->n_leafs; ++idx) {
    for (int c = 0; c < 2; ++c) {
      grid->data[idx*2 + c] = value;
    }
  }
  
  return grid;
}

#endif 
