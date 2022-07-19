/// Example of creating and printing an OctNet in 2D:
///
/// @author David Stutz
/// @file 2d_create_print.cpp

#define N_THREADS 4 ///< number of threads to use

#include "octnet/core/types.h"
#include "octnet/cpu/cpu.h"
#include "octnet/create/create.h"

/// Short example for convolution on 2d grid.
///
/// @param argc
/// @param argv
/// @return 
int main(int argc, char** argv) {
    
    // build dense data
    ot_size_t depth = 1;
    ot_size_t height = 64;
    ot_size_t width = 64;
    
    ot_data_t* dense = new ot_data_t[depth*height*width];
    
    int d = 0;
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        // see is_occupied in create_dense.cpp for data alignment
        dense[(d*height + h)*width + w] = 0.f;
      }
    }
    
    for (int h = 32; h < 40; h++) {
      for (int w = 32; w < 40; w++) {
        dense[(d*height + h)*width + w] = 1.f;
      }
    }
    
    // ranges, everything within the ranges will be labeled as "occupied".
    int n_ranges = 1;
    ot_data_t* ranges = new ot_data_t[2];
    ranges[0] = 0.5f;
    ranges[1] = 1.5f;
    
    octree* grid = octree_create_from_dense_cpu(dense, depth, height, width, n_ranges, ranges, false, 0, false, N_THREADS);
    octree_print_cpu(grid);
    
    return 0;
}

