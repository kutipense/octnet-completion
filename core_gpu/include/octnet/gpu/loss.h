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

#ifndef OCTREE_LOSS_GPU_H
#define OCTREE_LOSS_GPU_H

#include "octnet/core/core.h"

extern "C" {

ot_data_t octree_mse_loss_gpu(const octree* input, const octree* target, bool size_average, bool check);
void octree_mse_loss_bwd_gpu(const octree* input, const octree* target, bool size_average, bool check, octree* grad);

void octree_nll_loss_gpu(const octree* input, const octree* target, const ot_data_t* weights, int class_base, bool size_average, bool check, ot_data_t* output, ot_data_t* total_weight);
void octree_nll_loss_bwd_gpu(const octree* input, const octree* target, const ot_data_t* weights, const ot_data_t total_weight, int class_base, bool size_average, bool check, octree* grad);



void octree_bce_loss_gpu(const octree* input, const octree* target, bool size_average, bool check, ot_data_t* output, ot_data_t* total_weight);
void octree_bce_loss_bwd_gpu(const octree* input, const octree* target, bool size_average, bool check, octree* grad);

void octree_bce_dense_loss_gpu(const octree* input, const ot_data_t* target, bool size_average, ot_data_t* output, ot_data_t* total_weight);
void octree_bce_dense_loss_bwd_gpu(const octree* input, const ot_data_t* target, bool size_average, octree* grad);

void octree_bce_ds_loss_gpu(const octree* input, const octree* target, const octree* weights, bool size_average, ot_data_t* output, ot_data_t* total_weight);
void octree_bce_ds_loss_bwd_gpu(const octree* input, const octree* target, const octree* weights, bool size_average, ot_data_t total_weight, octree* grad);

void dense_bce_loss_gpu(const ot_data_t* input, const ot_data_t* target, const ot_data_t* weights, ot_size_t N, ot_data_t* output, ot_data_t* total_weight);
void dense_bce_loss_bwd_gpu(const ot_data_t* input, const ot_data_t* target, const ot_data_t* weights, ot_size_t N, ot_data_t total_weight, ot_data_t* grad); 
}

#endif 
