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

#include "octnet/gpu/col2oc.h"
#include "octnet/gpu/gpu.h"
#include "octnet/gpu/buffer.h"
#include "octnet/core/z_curve.h"

#define SHARED_N_FEAT 8

__device__ 
inline bool col2oc_in_vol(const octree* in, const int d, const int h, const int w) {
  return d >= 0 && h >= 0 && w >= 0 && d < 8 * in->grid_depth && h < 8 * in->grid_height && w < 8 * in->grid_width;
}


__device__
inline void col2oc_leaf(const ot_data_t* out, const ot_tree_t* leaf_tree, const int leaf_leaf_idx, const int leaf_grid_idx, const int leaf_bit_idx,
    const int n, const int ds, const int hs, const int ws, const int size, 
    ot_data_t* shared, octree* in) {

  ot_data_t factor;

  int d,h,w, kidx, grid_idx, bit_idx, data_idx, leaf_idx, data_cnt, data_cnt_e1, data_cnt_e2;
  ot_tree_t* tree;
  ot_data_t* data_in;
  ot_data_t val;

  data_idx = tree_data_idx(leaf_tree, leaf_bit_idx, in->feature_size);
  // data_in = in->data_ptrs[leaf_grid_idx] + data_idx;
  data_in = octree_get_data(in, leaf_grid_idx) + data_idx;

  for(int rep_f = 0; rep_f < (in->feature_size + SHARED_N_FEAT - 1) / SHARED_N_FEAT; ++rep_f) {
    int from_f = rep_f * SHARED_N_FEAT;
    int to_f = IMIN(SHARED_N_FEAT, in->feature_size - rep_f * SHARED_N_FEAT);

    for(int f = 0; f < to_f; ++f) {
      leaf_idx = leaf_leaf_idx;
      factor = 1.f / (size * size * size);

      //leaf data
      //1-center
      // (1,1,1)=13
      val = size*size*size * factor;
      shared[f] = val * out[(leaf_idx * in->feature_size + from_f + f) * K333 + 13];

      //6 
      val = (size-1)*size*size * factor;
      //(0,1,1)=4, (2,1,1)=22, (1,0,1)=10, (1,2,1)=16, (1,1,0)=12, (1,1,2)=14
      shared[f] += val * out[(leaf_idx * in->feature_size + from_f + f) * K333 + 4];
      shared[f] += val * out[(leaf_idx * in->feature_size + from_f + f) * K333 + 10];
      shared[f] += val * out[(leaf_idx * in->feature_size + from_f + f) * K333 + 12];
      shared[f] += val * out[(leaf_idx * in->feature_size + from_f + f) * K333 + 14];
      shared[f] += val * out[(leaf_idx * in->feature_size + from_f + f) * K333 + 16];
      shared[f] += val * out[(leaf_idx * in->feature_size + from_f + f) * K333 + 22];

      //8
      val = (size-1)*(size-1)*(size-1) * factor;
      //(0,0,0)=0, (0,0,2)=2, (0,2,0)=6, (0,2,2)=8, 
      //(2,0,0)=18, (2,0,2)=20, (2,2,0)=24, (2,2,2)=26
      shared[f] += val * out[(leaf_idx * in->feature_size + from_f + f) * K333 + 0];
      shared[f] += val * out[(leaf_idx * in->feature_size + from_f + f) * K333 + 2];
      shared[f] += val * out[(leaf_idx * in->feature_size + from_f + f) * K333 + 6];
      shared[f] += val * out[(leaf_idx * in->feature_size + from_f + f) * K333 + 8];
      shared[f] += val * out[(leaf_idx * in->feature_size + from_f + f) * K333 + 18];
      shared[f] += val * out[(leaf_idx * in->feature_size + from_f + f) * K333 + 20];
      shared[f] += val * out[(leaf_idx * in->feature_size + from_f + f) * K333 + 24];
      shared[f] += val * out[(leaf_idx * in->feature_size + from_f + f) * K333 + 26];

      //12
      val = (size-1)*(size-1)*(size) * factor;
      //(0,0,1)=1,  (0,1,0)=3,  (0,1,2)=5,  (0,2,1)=7
      //(1,0,0)=9,  (1,0,2)=11, (1,2,0)=15, (1,2,2)=17
      //(2,0,1)=19, (2,1,0)=21, (2,1,2)=23, (2,2,1)=25
      shared[f] += val * out[(leaf_idx * in->feature_size + from_f + f) * K333 + 1];
      shared[f] += val * out[(leaf_idx * in->feature_size + from_f + f) * K333 + 3];
      shared[f] += val * out[(leaf_idx * in->feature_size + from_f + f) * K333 + 5];
      shared[f] += val * out[(leaf_idx * in->feature_size + from_f + f) * K333 + 7];
      shared[f] += val * out[(leaf_idx * in->feature_size + from_f + f) * K333 + 9];
      shared[f] += val * out[(leaf_idx * in->feature_size + from_f + f) * K333 + 11];
      shared[f] += val * out[(leaf_idx * in->feature_size + from_f + f) * K333 + 15];
      shared[f] += val * out[(leaf_idx * in->feature_size + from_f + f) * K333 + 17];
      shared[f] += val * out[(leaf_idx * in->feature_size + from_f + f) * K333 + 19];
      shared[f] += val * out[(leaf_idx * in->feature_size + from_f + f) * K333 + 21];
      shared[f] += val * out[(leaf_idx * in->feature_size + from_f + f) * K333 + 23];
      shared[f] += val * out[(leaf_idx * in->feature_size + from_f + f) * K333 + 25];
    }


    //corner data
    for(int cd = 0; cd < 2; ++cd) { 
      for(int ch = 0; ch < 2; ++ch) { 
        for(int cw = 0; cw < 2; ++cw) { 
          d = ds + (cd*(size+1)-1); h = hs + (ch*(size+1)-1); w = ws + (cw*(size+1)-1);
          if(col2oc_in_vol(in, d,h,w)) {
            grid_idx = octree_grid_idx(in, n, d / 8, h / 8, w / 8);
            tree = octree_get_tree(in, grid_idx);
            bit_idx = tree_bit_idx(tree, d % 8, h % 8, w % 8);
            data_idx = tree_data_idx(tree, bit_idx, 1);
            // leaf_idx = n_leafs_upto(in, grid_idx) + data_idx;
            leaf_idx = in->prefix_leafs[grid_idx] + data_idx;
            factor = 1.f / (bit_idx == 0 ? 512 : (bit_idx < 9 ? 64 : (bit_idx < 73 ? 8 : 1)));

            kidx = ((1-cd)*2*3 + (1-ch)*2)*3 + (1-cw)*2;
            for(int f = 0; f < to_f; ++f) {
              shared[f] += factor * out[(leaf_idx * in->feature_size + from_f + f) * K333 + kidx];
            }
          }
        }
      }
    }


    //along the edges
    //d
    for(int ch = 0; ch < 2; ++ch) { 
      for(int cw = 0; cw < 2; ++cw) { 
        d = ds; h = hs + (ch*(size+1)-1); w = ws + (cw*(size+1)-1);
        if(col2oc_in_vol(in, d,h,w)) {
          grid_idx = octree_grid_idx(in, n, d / 8, h / 8, w / 8);
          tree = octree_get_tree(in, grid_idx);
          int e = 0;
          while(e < size) {
            d = ds + e;

            bit_idx = tree_bit_idx(tree, d % 8, h % 8, w % 8);
            data_cnt = bit_idx == 0 ? 8 : (bit_idx < 9 ? 4 : (bit_idx < 73 ? 2 : 1));
            factor = 1.f / (data_cnt * data_cnt * data_cnt);
            data_cnt = IMIN(size - e, data_cnt);
            data_idx = tree_data_idx(tree, bit_idx, 1);
            // leaf_idx = n_leafs_upto(in, grid_idx) + data_idx;
            leaf_idx = in->prefix_leafs[grid_idx] + data_idx;
            
            for(int f = 0; f < to_f; ++f) {
              kidx = ((2) * 3 + ((1-ch)*2)) * 3 + ((1-cw)*2);
              shared[f] += (data_cnt - (e+data_cnt >= size)) * factor * out[(leaf_idx * in->feature_size + from_f + f) * K333 + kidx];
              kidx = ((1) * 3 + ((1-ch)*2)) * 3 + ((1-cw)*2);
              shared[f] += data_cnt * factor * out[(leaf_idx * in->feature_size + from_f + f) * K333 + kidx];
              kidx = ((0) * 3 + ((1-ch)*2)) * 3 + ((1-cw)*2);
              shared[f] += (data_cnt - (e == 0)) * factor * out[(leaf_idx * in->feature_size + from_f + f) * K333 + kidx];
            }

            e += data_cnt;
          }
        }
      }
    }

    //h
    for(int cd = 0; cd < 2; ++cd) { 
      for(int cw = 0; cw < 2; ++cw) { 
        d = ds + (cd*(size+1)-1); h = hs; w = ws + (cw*(size+1)-1);
        if(col2oc_in_vol(in, d,h,w)) {
          grid_idx = octree_grid_idx(in, n, d / 8, h / 8, w / 8);
          tree = octree_get_tree(in, grid_idx);
          int e = 0;
          while(e < size) {
            h = hs + e;

            bit_idx = tree_bit_idx(tree, d % 8, h % 8, w % 8);
            data_cnt = bit_idx == 0 ? 8 : (bit_idx < 9 ? 4 : (bit_idx < 73 ? 2 : 1));
            factor = 1.f / (data_cnt * data_cnt * data_cnt);
            data_cnt = IMIN(size - e, data_cnt);
            data_idx = tree_data_idx(tree, bit_idx, 1);
            // leaf_idx = n_leafs_upto(in, grid_idx) + data_idx;
            leaf_idx = in->prefix_leafs[grid_idx] + data_idx;

            for(int f = 0; f < to_f; ++f) {
              kidx = (((1-cd)*2) * 3 + (2)) * 3 + ((1-cw)*2);
              shared[f] += (data_cnt - (e+data_cnt >= size)) * factor * out[(leaf_idx * in->feature_size + from_f + f) * K333 + kidx];
              kidx = (((1-cd)*2) * 3 + (1)) * 3 + ((1-cw)*2);
              shared[f] += data_cnt * factor * out[(leaf_idx * in->feature_size + from_f + f) * K333 + kidx];
              kidx = (((1-cd)*2) * 3 + (0)) * 3 + ((1-cw)*2);
              shared[f] += (data_cnt - (e == 0)) * factor * out[(leaf_idx * in->feature_size + from_f + f) * K333 + kidx];
            }

            e += data_cnt;
          }
        }
      }
    }

    //w
    for(int cd = 0; cd < 2; ++cd) { 
      for(int ch = 0; ch < 2; ++ch) { 
        d = ds + (cd*(size+1)-1); h = hs + (ch*(size+1)-1); w = ws;
        if(col2oc_in_vol(in, d,h,w)) {
          grid_idx = octree_grid_idx(in, n, d / 8, h / 8, w / 8);
          tree = octree_get_tree(in, grid_idx);
          int e = 0;
          while(e < size) {
            w = ws + e;

            bit_idx = tree_bit_idx(tree, d % 8, h % 8, w % 8);
            data_cnt = bit_idx == 0 ? 8 : (bit_idx < 9 ? 4 : (bit_idx < 73 ? 2 : 1));
            factor = 1.f / (data_cnt * data_cnt * data_cnt);
            data_cnt = IMIN(size - e, data_cnt);
            data_idx = tree_data_idx(tree, bit_idx, 1);
            // leaf_idx = n_leafs_upto(in, grid_idx) + data_idx;
            leaf_idx = in->prefix_leafs[grid_idx] + data_idx;

            for(int f = 0; f < to_f; ++f) {
              kidx = (((1-cd)*2) * 3 + ((1-ch)*2)) * 3 + (2);
              shared[f] += (data_cnt - (e+data_cnt >= size)) * factor * out[(leaf_idx * in->feature_size + from_f + f) * K333 + kidx];
              kidx = (((1-cd)*2) * 3 + ((1-ch)*2)) * 3 + (1);
              shared[f] += data_cnt * factor * out[(leaf_idx * in->feature_size + from_f + f) * K333 + kidx];
              kidx = (((1-cd)*2) * 3 + ((1-ch)*2)) * 3 + (0);
              shared[f] += (data_cnt - (e == 0)) * factor * out[(leaf_idx * in->feature_size + from_f + f) * K333 + kidx];
            }

            e += data_cnt;
          }
        }
      }
    }


    //along the faces
    //d
    for(int fd = 0; fd < 2; ++fd) {
      d = ds + (fd*(size+1)-1); h = hs; w = ws;
      if(col2oc_in_vol(in, d,h,w)) {
        grid_idx = octree_grid_idx(in, n, d / 8, h / 8, w / 8);
        tree = octree_get_tree(in, grid_idx);
        int z = 0;
        while(z < size * size) {
          const int e1 = z_curve_x(z);
          const int e2 = z_curve_y(z);
          h = hs + e1;
          w = ws + e2;

          bit_idx = tree_bit_idx(tree, d % 8, h % 8, w % 8);
          data_idx = tree_data_idx(tree, bit_idx, 1);
          // leaf_idx = n_leafs_upto(in, grid_idx) + data_idx;
          leaf_idx = in->prefix_leafs[grid_idx] + data_idx;
          data_cnt = bit_idx == 0 ? 8 : (bit_idx < 9 ? 4 : (bit_idx < 73 ? 2 : 1));
          data_cnt_e1 = IMIN(size - e1, data_cnt);
          data_cnt_e2 = IMIN(size - e2, data_cnt);
          factor = 1.f / (data_cnt * data_cnt * data_cnt);
          data_cnt = IMIN(size * size - z, data_cnt * data_cnt);

          for(int f = 0; f < to_f; ++f) {
            kidx = (((1-fd)*2) * 3 + (2)) * 3 + (2);
            shared[f] += (data_cnt_e1 - (e1+data_cnt_e1 >= size)) * (data_cnt_e2 - (e2+data_cnt_e2 >= size)) * factor * out[(leaf_idx * in->feature_size + from_f + f) * K333 + kidx];
            kidx = (((1-fd)*2) * 3 + (2)) * 3 + (1);
            shared[f] += (data_cnt_e1 - (e1+data_cnt_e1 >= size)) * (data_cnt_e2) * factor * out[(leaf_idx * in->feature_size + from_f + f) * K333 + kidx];
            kidx = (((1-fd)*2) * 3 + (2)) * 3 + (0);
            shared[f] += (data_cnt_e1 - (e1+data_cnt_e1 >= size)) * (data_cnt_e2 - (e2 == 0)) * factor * out[(leaf_idx * in->feature_size + from_f + f) * K333 + kidx];
            kidx = (((1-fd)*2) * 3 + (1)) * 3 + (2);
            shared[f] += (data_cnt_e1) * (data_cnt_e2 - (e2+data_cnt_e2 >= size)) * factor * out[(leaf_idx * in->feature_size + from_f + f) * K333 + kidx];
            kidx = (((1-fd)*2) * 3 + (1)) * 3 + (1);
            shared[f] += data_cnt * factor * out[(leaf_idx * in->feature_size + from_f + f) * K333 + kidx];
            kidx = (((1-fd)*2) * 3 + (1)) * 3 + (0);
            shared[f] += (data_cnt_e1) * (data_cnt_e2 - (e2 == 0)) * factor * out[(leaf_idx * in->feature_size + from_f + f) * K333 + kidx];
            kidx = (((1-fd)*2) * 3 + (0)) * 3 + (2);
            shared[f] += (data_cnt_e1 - (e1 == 0)) * (data_cnt_e2 - (e2+data_cnt_e2 >= size)) * factor * out[(leaf_idx * in->feature_size + from_f + f) * K333 + kidx];
            kidx = (((1-fd)*2) * 3 + (0)) * 3 + (1);
            shared[f] += (data_cnt_e1 - (e1 == 0)) * (data_cnt_e2) * factor * out[(leaf_idx * in->feature_size + from_f + f) * K333 + kidx];
            kidx = (((1-fd)*2) * 3 + (0)) * 3 + (0);
            shared[f] += (data_cnt_e1 - (e1 == 0)) * (data_cnt_e2 - (e2 == 0)) * factor * out[(leaf_idx * in->feature_size + from_f + f) * K333 + kidx];          
          }

          z += data_cnt;
        }
      }
    }

    //h
    for(int fh = 0; fh < 2; ++fh) {
      d = ds; h = hs + (fh*(size+1)-1); w = ws;
      if(col2oc_in_vol(in, d,h,w)) {
        grid_idx = octree_grid_idx(in, n, d / 8, h / 8, w / 8);
        tree = octree_get_tree(in, grid_idx);
        int z = 0;
        while(z < size * size) {
          const int e1 = z_curve_x(z);
          const int e2 = z_curve_y(z);
          d = ds + e1;
          w = ws + e2;

          bit_idx = tree_bit_idx(tree, d % 8, h % 8, w % 8);
          data_idx = tree_data_idx(tree, bit_idx, 1);
          // leaf_idx = n_leafs_upto(in, grid_idx) + data_idx;
          leaf_idx = in->prefix_leafs[grid_idx] + data_idx;
          data_cnt = bit_idx == 0 ? 8 : (bit_idx < 9 ? 4 : (bit_idx < 73 ? 2 : 1));
          data_cnt_e1 = IMIN(size - e1, data_cnt);
          data_cnt_e2 = IMIN(size - e2, data_cnt);
          factor = 1.f / (data_cnt * data_cnt * data_cnt);
          data_cnt = IMIN(size * size - z, data_cnt * data_cnt);

          for(int f = 0; f < to_f; ++f) {
            kidx = ((2) * 3 + ((1-fh)*2)) * 3 + (2);
            shared[f] += (data_cnt_e1 - (e1+data_cnt_e1 >= size)) * (data_cnt_e2 - (e2+data_cnt_e2 >= size)) * factor * out[(leaf_idx * in->feature_size + from_f + f) * K333 + kidx];
            kidx = ((2) * 3 + ((1-fh)*2)) * 3 + (1);
            shared[f] += (data_cnt_e1 - (e1+data_cnt_e1 >= size)) * (data_cnt_e2) * factor * out[(leaf_idx * in->feature_size + from_f + f) * K333 + kidx];
            kidx = ((2) * 3 + ((1-fh)*2)) * 3 + (0);
            shared[f] += (data_cnt_e1 - (e1+data_cnt_e1 >= size)) * (data_cnt_e2 - (e2 == 0)) * factor * out[(leaf_idx * in->feature_size + from_f + f) * K333 + kidx];
            kidx = ((1) * 3 + ((1-fh)*2)) * 3 + (2);
            shared[f] += (data_cnt_e1) * (data_cnt_e2 - (e2+data_cnt_e2 >= size)) * factor * out[(leaf_idx * in->feature_size + from_f + f) * K333 + kidx];
            kidx = ((1) * 3 + ((1-fh)*2)) * 3 + (1);
            shared[f] += data_cnt * factor * out[(leaf_idx * in->feature_size + from_f + f) * K333 + kidx];
            kidx = ((1) * 3 + ((1-fh)*2)) * 3 + (0);
            shared[f] += (data_cnt_e1) * (data_cnt_e2 - (e2 == 0)) * factor * out[(leaf_idx * in->feature_size + from_f + f) * K333 + kidx];
            kidx = ((0) * 3 + ((1-fh)*2)) * 3 + (2);
            shared[f] += (data_cnt_e1 - (e1 == 0)) * (data_cnt_e2 - (e2+data_cnt_e2 >= size)) * factor * out[(leaf_idx * in->feature_size + from_f + f) * K333 + kidx];
            kidx = ((0) * 3 + ((1-fh)*2)) * 3 + (1);
            shared[f] += (data_cnt_e1 - (e1 == 0)) * (data_cnt_e2) * factor * out[(leaf_idx * in->feature_size + from_f + f) * K333 + kidx];
            kidx = ((0) * 3 + ((1-fh)*2)) * 3 + (0);
            shared[f] += (data_cnt_e1 - (e1 == 0)) * (data_cnt_e2 - (e2 == 0)) * factor * out[(leaf_idx * in->feature_size + from_f + f) * K333 + kidx];          
          }
          
          z += data_cnt;
        }
      }
    }

    //w
    for(int fw = 0; fw < 2; ++fw) {
      d = ds; h = hs; w = ws + (fw*(size+1)-1); 
      if(col2oc_in_vol(in, d,h,w)) {
        grid_idx = octree_grid_idx(in, n, d / 8, h / 8, w / 8);
        tree = octree_get_tree(in, grid_idx);
        int z = 0;
        while(z < size * size) {
          const int e1 = z_curve_x(z);
          const int e2 = z_curve_y(z);
          d = ds + e1;
          h = hs + e2;

          bit_idx = tree_bit_idx(tree, d % 8, h % 8, w % 8);
          data_idx = tree_data_idx(tree, bit_idx, 1);
          // leaf_idx = n_leafs_upto(in, grid_idx) + data_idx;
          leaf_idx = in->prefix_leafs[grid_idx] + data_idx;
          data_cnt = bit_idx == 0 ? 8 : (bit_idx < 9 ? 4 : (bit_idx < 73 ? 2 : 1));
          data_cnt_e1 = IMIN(size - e1, data_cnt);
          data_cnt_e2 = IMIN(size - e2, data_cnt);
          factor = 1.f / (data_cnt * data_cnt * data_cnt);
          data_cnt = IMIN(size * size - z, data_cnt * data_cnt);

          for(int f = 0; f < to_f; ++f) {
            kidx = ((2) * 3 + (2)) * 3 + ((1-fw)*2);
            shared[f] += (data_cnt_e1 - (e1+data_cnt_e1 >= size)) * (data_cnt_e2 - (e2+data_cnt_e2 >= size)) * factor * out[(leaf_idx * in->feature_size + from_f + f) * K333 + kidx];
            kidx = ((2) * 3 + (1)) * 3 + ((1-fw)*2);
            shared[f] += (data_cnt_e1 - (e1+data_cnt_e1 >= size)) * (data_cnt_e2) * factor * out[(leaf_idx * in->feature_size + from_f + f) * K333 + kidx];
            kidx = ((2) * 3 + (0)) * 3 + ((1-fw)*2);
            shared[f] += (data_cnt_e1 - (e1+data_cnt_e1 >= size)) * (data_cnt_e2 - (e2 == 0)) * factor * out[(leaf_idx * in->feature_size + from_f + f) * K333 + kidx];
            kidx = ((1) * 3 + (2)) * 3 + ((1-fw)*2);
            shared[f] += (data_cnt_e1) * (data_cnt_e2 - (e2+data_cnt_e2 >= size)) * factor * out[(leaf_idx * in->feature_size + from_f + f) * K333 + kidx];
            kidx = ((1) * 3 + (1)) * 3 + ((1-fw)*2);
            shared[f] += data_cnt * factor * out[(leaf_idx * in->feature_size + from_f + f) * K333 + kidx];
            kidx = ((1) * 3 + (0)) * 3 + ((1-fw)*2);
            shared[f] += (data_cnt_e1) * (data_cnt_e2 - (e2 == 0)) * factor * out[(leaf_idx * in->feature_size + from_f + f) * K333 + kidx];
            kidx = ((0) * 3 + (2)) * 3 + ((1-fw)*2);
            shared[f] += (data_cnt_e1 - (e1 == 0)) * (data_cnt_e2 - (e2+data_cnt_e2 >= size)) * factor * out[(leaf_idx * in->feature_size + from_f + f) * K333 + kidx];
            kidx = ((0) * 3 + (1)) * 3 + ((1-fw)*2);
            shared[f] += (data_cnt_e1 - (e1 == 0)) * (data_cnt_e2) * factor * out[(leaf_idx * in->feature_size + from_f + f) * K333 + kidx];
            kidx = ((0) * 3 + (0)) * 3 + ((1-fw)*2);
            shared[f] += (data_cnt_e1 - (e1 == 0)) * (data_cnt_e2 - (e2 == 0)) * factor * out[(leaf_idx * in->feature_size + from_f + f) * K333 + kidx];          
          }
          
          z += data_cnt;
        }
      }
    }

    //copy shared mem
    for(int f = 0; f < to_f; ++f) {
      data_in[from_f + f] = shared[f];
    }
  }
}




__global__ void kernel_col2oc_leafs(octree in, int n_leafs, const ot_data_t* col_buffer) {
  extern __shared__ ot_data_t out_shared[];

  CUDA_KERNEL_LOOP(leaf_idx, n_leafs) {
    const int grid_idx = in.data[leaf_idx * in.feature_size];
    const ot_tree_t* tree = octree_get_tree(&in, grid_idx);

    // const int cum_n_leafs = n_leafs_upto(&in, grid_idx);
    const int cum_n_leafs = in.prefix_leafs[grid_idx];
    const int data_idx = leaf_idx - cum_n_leafs;
    const int bit_idx = data_idx_to_bit_idx(tree, data_idx);

    int n,d,h,w;
    const int depth = octree_ind_to_dense_ind(&in, grid_idx, bit_idx, &n, &d,&h,&w);
    const int size = width_from_depth(depth);
    
    col2oc_leaf(col_buffer, tree, leaf_idx, grid_idx, bit_idx, n,d,h,w,size, out_shared + threadIdx.x * SHARED_N_FEAT, &in);
  }
}


void col2oc_gpu(const ot_data_t* col_buffer, octree* in) {
  const int n_blocks = octree_num_blocks(in);

  octree_leaf_idx_to_grid_idx_gpu(in, in->feature_size, in->data_capacity, in->data);

  // CUDA_CHECK( cudaFuncSetCacheConfig(&kernel_oc2col_leafs, cudaFuncCachePreferShared); );
  kernel_col2oc_leafs<<<GET_BLOCKS_T(in->n_leafs, 256), 256, 256 * SHARED_N_FEAT * sizeof(ot_data_t)>>>(*in, in->n_leafs, col_buffer);
  CUDA_POST_KERNEL_CHECK;
}
