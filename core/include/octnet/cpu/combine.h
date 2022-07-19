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

#ifndef OCTREE_COMBINE_CPU_H
#define OCTREE_COMBINE_CPU_H

#include "octnet/core/core.h"

extern "C" {

void octree_concat_cpu(const octree* grid_in1, const octree* grid_in2, bool check, octree* grid_out);
void octree_concat_bwd_cpu(const octree* in1, const octree* in2, const octree* grad_out, bool do_grad_in2, octree* grad_in1, octree* grad_in2);

void octree_concat_dense_cpu(const octree* in1, const ot_data_t* in2, ot_size_t feature_size2, octree* out);
void octree_concat_dense_bwd_cpu(const octree* in1, const ot_data_t* in2, ot_size_t feature_size2, const octree* grad_out, bool do_grad_in2, octree* grad_in1, ot_data_t* grad_in2);

void octree_combine_n_cpu(octree** in, const int n, octree* out);

void octree_extract_n_cpu(const octree* in, int from, int to, octree* out);
void octree_extract_feature_cpu(const octree* in, int from, int to, octree* out);

} //extern "C"

#endif
