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

#include <cstdlib>
#include <cstdio>
#include <cmath>

#include "octnet/core/types.h"
#include "octnet/cpu/fc.h"

#define EPS 1e-6

void expect(bool should_be_true, const char* message) {
  if (!should_be_true) {
    printf("%s\n", message);
    exit(-1);
  }
}

void test_fc() {
  const int D_in = 3;
  const int D_out = 3;
  ot_data_t* weights = new ot_data_t[D_in*D_out];
  ot_data_t* biases = new ot_data_t[D_out];
  
  for (int i = 0; i < D_in*D_out; i++) {
    weights[i] = 1;
  }
  
  for (int i = 0; i < D_out; i++) {
    biases[i] = 0.5f;
  }
  
  const int B = 2;
  ot_data_t* data = new ot_data_t[B*D_in];
  for (int b = 0; b < B; b++) {
    for (int d = 0; d < D_in; d++) {
      data[b*D_in + d] = (b + 1);
    }
  }
  
  ot_data_t* output = new ot_data_t[B*D_out];
  dense_fc_cpu(data, weights, biases, B, D_in, D_out, output);
  
  char* buffer = new char[100];
  for (int b = 0; b < B; b++) {
    for (int d = 0; d < D_out; d++) {
      sprintf(buffer, "expected %f to be equal to %f", output[b*D_out + d], D_in*b + 0.5f);
      expect(output[b*D_out + d] == D_in*(b + 1) + 0.5f, buffer);
    }
  }
}

void test_sigmoid() {
  const int B = 2;
  const int D = 3;
  ot_data_t* input = new ot_data_t[B*D];
  ot_data_t* output = new ot_data_t[B*D];
  
  for (int i = 0; i < D; i++) {
    input[i] = 1;
  }
  
  dense_sigmoid_cpu(input, B, D, output);
  
  char* buffer = new char[100];
  for (int i = 0; i < D; i++) {
    sprintf(buffer, "expected %f to be equal to %f", output[i], 1.f/(1.f + exp(input[i])));
    expect(abs(output[i] - 1.f/(1.f + exp(-input[i]))) < EPS, buffer);
  }
}

void test_bce() {
  const int B = 2;
  ot_data_t* input = new ot_data_t[B];
  input[0] = std::rand() / ((float) RAND_MAX);
  input[1] = std::rand() / ((float) RAND_MAX);
  
  ot_data_t* target = new ot_data_t[B];
  target[0] = 1;
  target[1] = 0;
  
  ot_data_t loss = dense_bce_cpu(input, target, B, 1);
  
  expect(!std::isnan(loss), "loss is NaN");
  expect(!std::isinf(loss), "loss is inf");
}

int main(int argc, char** argv) {
  test_fc();
  test_sigmoid();
  test_bce();
  return 0;
}

