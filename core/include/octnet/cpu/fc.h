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

#ifndef OCTREE_FC_CPU_H
#define OCTREE_FC_CPU_H

#include "octnet/core/types.h"

extern "C" {

/// Fully connected layer on an dense array; use one of the functions in dense.h
/// for conversion.
/// 
/// @param input input array
/// @param weights weights as array in format hw, output will be the 
///   matrix-vector product of weights and input plus bias
/// @param bias bias vector to use
/// @param batch_size batch size of input and output
/// @param num_input number of input units
/// @param num_output number of output units
/// @param output output array
void dense_fc_cpu(ot_data_t* input, ot_data_t* weights, ot_data_t* bias, int batch_size, int num_input, int num_output, ot_data_t* output);

/// Fully connected backward pass computing the partial derivative of the loss
/// with respect to the input.
/// 
/// @param weights weights of the layer
/// @param grad_out gradients of the top layer
/// @param batch_size batch size of input and output
/// @param num_input number of input units
/// @param num_output number of output units
/// @param grad_in output gradients of the loss with respect to the layer's input
void dense_fc_bwd_cpu(ot_data_t* weights, ot_data_t* grad_out, int batch_size, int num_input, int num_output, ot_data_t* grad_in);

/// Fully connected backward pass computing the derivative of the loss
/// with respect to the weights and bias.
/// 
/// @param input input of the layer
/// @param grad_out gradients of the top layer
/// @param batch_size batch size of input and output
/// @param num_input number of input units
/// @param num_output number of output units
/// @param scale scale allowing to scale the gradients
/// @param grad_weights gradients of the loss with respect to the weights
/// @param grad_bias gradients of the loss with respect to the bias
void dense_fc_wbwd_cpu(ot_data_t* input, ot_data_t* grad_out, int batch_size, int num_input, int num_output, ot_data_t scale, ot_data_t* grad_weights, ot_data_t* grad_bias);

/// Dense sigmoid layer; computes the point wise sigmoid.
///
/// @param input input of the layer
/// @param batch_size batch size of input and output
/// @param num_input number of input units (= number of output units)
/// @param output output of the layer
void dense_sigmoid_cpu(ot_data_t* input, int batch_size, int num_input, ot_data_t* output);

/// Dense backward pass of sigmoid layer.
///
/// @param output output of the sigmoid layer
/// @param grad_out gradients of the top layer
/// @param batch_size batch size of input and output
/// @param num_input number of units in layer (same for in- and output)
/// @param grad_in gradients of the sigmoid layer with respect to its input
void dense_sigmoid_bwd_cpu(ot_data_t* output, ot_data_t* grad_out, int batch_size, int num_input, ot_data_t* grad_in);

/// Compute the binary cross-entropy loss.
/// 
/// @param input input array
/// @param target target array
/// @param num_input number of input units
/// @return binary cross entropy loss
ot_data_t dense_bce_cpu(ot_data_t* input, ot_data_t* target, int batch_size, int num_input);

/// Compute the gradient of the binary cross-entropy loss with respect to the
/// inputs.
/// 
/// @param input input array
/// @param target target array
/// @param batch_size batch size of input and output
/// @param num_input number of input units
/// @param grad_in gradients with respect to inputs
void dense_bce_bwd_cpu(ot_data_t* input, ot_data_t* target, int batch_size, int num_input, ot_data_t* grad_in);

/// Check for NaN and inf values in the dense array. Will abort if found.
/// 
/// @param array array of length n to check
/// @param n length of array
void dense_check_nan_inf_cpu(ot_data_t* array, int n, const char* identifier);

} // extern "C"

#endif