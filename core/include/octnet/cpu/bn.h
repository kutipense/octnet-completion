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

#ifndef OCTREE_BN_CPU_H
#define OCTREE_BN_CPU_H

#include "octnet/core/core.h"

extern "C" {
  /// The normalization part of batch normalization, i.e. normalizes the input
  /// by mean and standard deviation per octree channel (i.e. mean and standard
  /// deviation computes over the batch and channel, for each channel separately).
  /// @param grid_in input octree
  /// @param avgs array to write channel averages to, array of size grid_in->feature_size initialized with zero
  /// @param vars array to write channel standard deviations to, array of size grid_in->feature_size initialized with zero
  /// @param grid output octree with normalized values
  void octree_bn_norm_cpu(const octree* grid_in, ot_data_t* avgs, ot_data_t* vars, octree* grid);
  
  /// Scales and shifts the normalized input, i.e. is called after octree_bn_norm_cpu.
  /// \param grid_in input octree
  /// \param gamma array of size grid_in->feature_size to scale channels with
  /// \param beta array of size grid_in->feature_size to shift channels with
  /// \param inplace whether to do computation inplace, then grid_out should point to grid_in
  /// \param grid_out output octree
  void octree_bn_ss_cpu(const octree* grid_in, ot_data_t *gamma, ot_data_t *beta, bool inplace, octree* grid_out);
  
  /// Backward pass of normalization part of batch normalization.
  /// \param grid_in input octree of original forward pass
  /// \param grad_out output gradients as octree, i.e. gradients of top layer
  /// \param avgs array with averages for each of the grid_in->feature_size channels
  /// \param vars array with standard deviations for each of the grid_in->feature_size channels
  /// \param grad_in gradients of this layer as octree
  void octree_bn_norm_bwd_cpu(const octree* grid_in, const octree* grad_out, ot_data_t* avgs, ot_data_t* vars, octree* grad_in);
  
  /// Backward pass of scaling and shifting.
  /// \param grad_out output gradients as octree, i.e. gradients of top layer
  /// \param gamma array of gammas of size grad_out->feature_size the channels were scaled with
  /// \param inplace whether to do computation
  /// \param grad_in gradients of this layer as octree
  void octree_bn_ss_bwd_cpu(const octree* grad_out, ot_data_t* gamma, bool inplace, octree* grad_in);
  
  /// Gradients of gamma and beta.
  /// \param grid_in input octree of original forward pass
  /// \param grad_out output gradients as octree, i.e. gradients of top layer
  /// \param grad_gamma gradients with respect to gamma, array of size grid_in->feature_size initialized with zero
  /// \param grad_beta gradient with respect to beta, array of size grid_in->feature_size initialized with zero
  void octree_bn_ss_wbwd_cpu(const octree* grid_in, const octree* grad_out, ot_data_t* grad_gamma, ot_data_t* grad_beta);
}

#endif // OCTREE_BN_CPU_H