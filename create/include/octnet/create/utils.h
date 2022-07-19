#ifndef OCTREE_CREATE_UTILS_H
#define OCTREE_CREATE_UTILS_H

#include "octnet/core/core.h"

extern "C" {

void octree_scanline_fill(octree* grid, ot_data_t fill_value);

void octree_occupancy_to_surface(octree* in, octree* out);

}

#endif
