/// Example of creating and convolving an OctNet in 2D.
/// 
/// @author David Stutz
/// @file 2d_create_conv

#define N_THREADS 1 ///< number of threads to use

#include <iostream>
#include "octnet/core/types.h"
#include "octnet/cpu/cpu.h"
#include "octnet/cpu/conv.h"
#include "octnet/cpu/pool.h"
#include "octnet/cpu/dense.h"
#include "octnet/create/create.h"

/// Easy access for DHWC data format.
///
/// @param array
/// @param depth
/// @param height
/// @param width
/// @param channels
/// @param d
/// @param h
/// @param w
/// @param c
template<typename T>
T& dhwc_access(T*array, int depth, int height, int width, int channels, int d, int h, int w, int c) {
  return array[((d*height + h)*width + w)*channels + c];
}

/// Utility for printing volumes.
///
/// @param array
/// @param depth
/// @param height
/// @param width
template<typename T>
void print_volume(T* array, int depth, int height, int width, int channels = 1) {
  for (int d = 0; d < depth; d++) {
    std::cout << "-- " << d << " --" << std::endl;
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        for (int c = 0; c < channels; c++) {
          std::cout << array[((d*height + h)*width + w)*channels + c] << ",";
        }
      }
      std::cout << std::endl;
    }
  }
}

/// Example of convolving an OctNet in 2D.
///
/// @param argc
/// @param argv
/// @return 
int main(int argc, char** argv) {
  
  // build dense data
  ot_size_t depth = 16;
  ot_size_t height = 16;
  ot_size_t width = 16;
  ot_size_t channels = 1;
  
  ot_data_t* dense = new ot_data_t[depth*height*width];

  for (int d = 0; d < depth; d++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        // see is_occupied in create_dense.cpp for data alignment
        dhwc_access(dense, depth, height, width, channels, d, h, w, 0) = 0.f;

        if (h%2 == 0) {
          dhwc_access(dense, depth, height, width, channels, d, h, w, 0) = 1.f;
        }
      }
    }
  }
  

  // ranges, everything within the ranges will be labeled as "occupied".
  int n_ranges = 1;
  ot_data_t* ranges = new ot_data_t[2];
  ranges[0] = 0.5f;
  ranges[1] = 1.5f;

  octree* grid = octree_create_from_dense_cpu(dense, depth, height, width, n_ranges, ranges, false, 0, false, N_THREADS);
  
  ot_data_t* weights = new ot_data_t[3*3*3];
  dhwc_access(weights, 3, 3, 3, 1, 1, 1, 1, 0) = 1.5f;
  
  ot_data_t* bias = new ot_data_t[1];
  bias[0] = 0;
  
  octree* conv_grid = octree_new_cpu();
  octree_conv3x3x3_sum_cpu(grid, weights, bias, 1, conv_grid);
  
  // regular pool only pools, but does not adapt the size (one could simply
  // take every second value ...)
  // gridpool also changes the size, however requires the volume to be at least
  // 16x16x16 in contrast to pool
  octree* pool_grid = octree_new_cpu();
  //octree_pool2x2x2_max_cpu(conv_grid, false, false, true, pool_grid);
  octree_gridpool2x2x2_max_cpu(conv_grid, pool_grid);
  
  ot_data_t* conv_dense = new ot_data_t[conv_grid->grid_depth*8*conv_grid->grid_height*8*conv_grid->grid_width*8];
  octree_to_dhwc_cpu(conv_grid, conv_grid->grid_depth*8, conv_grid->grid_height*8, conv_grid->grid_width*8, conv_dense);
  
  ot_data_t* pool_dense = new ot_data_t[pool_grid->grid_depth*8*pool_grid->grid_height*8*pool_grid->grid_width*8];
  octree_to_dhwc_cpu(pool_grid, pool_grid->grid_depth*8, pool_grid->grid_height*8, pool_grid->grid_width*8, pool_dense);
  
  print_volume(dense, grid->grid_depth*8, grid->grid_height*8, grid->grid_width*8);
  print_volume(conv_dense, conv_grid->grid_depth*8, conv_grid->grid_height*8, conv_grid->grid_width*8);
  print_volume(pool_dense, pool_grid->grid_depth*8, pool_grid->grid_height*8, pool_grid->grid_width*8);
  
  return 0;
}

