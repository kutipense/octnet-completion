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

#include "octnet/core/types.h"
#include "octnet/cpu/fc.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>

#define EPS 1e-12

extern "C"
void dense_fc_cpu(ot_data_t* input, ot_data_t* weights, ot_data_t* bias, int batch_size, int num_input, int num_output, ot_data_t* output) {
  for (int n = 0; n < batch_size; ++n) {
    for (int h = 0; h < num_output; ++h) {
      output[n*num_output + h] = bias[h];
      for (int w = 0; w < num_input; ++w) {
        output[n*num_output + h] += weights[h*num_input + w]*input[n*num_input + w];
      }
    }
  }
}

extern "C"
void dense_fc_bwd_cpu(ot_data_t* weights, ot_data_t* grad_out, int batch_size, int num_input, int num_output, ot_data_t* grad_in) {
  for (int n = 0; n < batch_size; ++n) {
    for (int w = 0; w < num_input; ++w) {
      grad_in[n*num_input + w] = 0;
      for (int h = 0; h < num_output; ++h) {
        grad_in[n*num_input + w] += weights[h*num_input + w]*grad_out[n*num_output + h];
      }
    }
  }
}

extern "C"
void dense_fc_wbwd_cpu(ot_data_t* input, ot_data_t* grad_out, int batch_size, int num_input, int num_output, ot_data_t scale, ot_data_t* grad_weights, ot_data_t* grad_bias) {
  for (int h = 0; h < num_output; ++h) {
    for (int w = 0; w < num_input; ++w) {
      for (int n = 0; n < batch_size; ++n) {
        grad_weights[h*num_input + w] += scale*input[n*num_input + w]*grad_out[n*num_output + h];
      }
    }
    
    for (int n = 0; n < batch_size; ++n) {
      grad_bias[h] += grad_out[n*num_output + h];
    }
  }
}

extern "C"
void dense_sigmoid_cpu(ot_data_t* input, int batch_size, int num_input, ot_data_t* output) {
  for (int i = 0; i < num_input*batch_size; i++) {
    output[i] = 1.f/(1.f + exp(-input[i]));
  }
}

extern "C"
void dense_sigmoid_bwd_cpu(ot_data_t* output, ot_data_t* grad_out, int batch_size, int num_input, ot_data_t* grad_in) {
  for (int i = 0; i < num_input*batch_size; i++) {
    grad_in[i] = grad_out[i] * (1.f - output[i]) * output[i];
  }
}

extern "C"
ot_data_t dense_bce_cpu(ot_data_t* input, ot_data_t* target, int batch_size, int num_input) {
  ot_data_t loss = 0;
  for (int n = 0; n < batch_size; ++n) {
    for (int h = 0; h < num_input; ++h) {
      loss -= log(input[n*num_input + h] + EPS)*target[n*num_input + h] + log(1 - input[n*num_input + h] + EPS)*(1 - target[n*num_input + h]);
    }
  }
  
  return loss/batch_size;
}

extern "C"
void dense_bce_bwd_cpu(ot_data_t* input, ot_data_t* target, int batch_size, int num_input, ot_data_t* grad_in) {
  for (int n = 0; n < batch_size; ++n) {
    for (int h = 0; h < num_input; ++h) {
      grad_in[n*num_input + h] = 1.f/ batch_size * (input[n*num_input + h] - target[n*num_input + h]) / (input[n*num_input + h]*(1 - input[n*num_input + h]) + EPS);
    }
  }
}

void dense_check_nan_inf_cpu(ot_data_t* array, int n, const char* identifier) {
  for (int i = 0; i < n; i++) {
    if (std::isnan(array[i])) {
      printf("[ERROR] NaN value in %s\n", identifier);
      exit(-1);
    }
    if (std::isinf(array[i])) {
      printf("[ERROR] Inf value in %s\n", identifier);
      exit(-1);
    }
  }
}