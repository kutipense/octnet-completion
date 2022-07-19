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

#include "octnet/cpu/cpu.h"
#include "octnet/cpu/io.h"
#include "octnet/create/create.h"

#include <cstring>
#include <iostream>

void test_dense_features() {
  const int depth = 2;
  const int height = 2;
  const int width = 2;
  const int channels = 3;
  
  ot_data_t* data = new ot_data_t[depth*height*width*channels];
  for (int d = 0; d < depth; ++d) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        for (int c = 0; c < channels; ++c) {
          data[((d*height + h)*width + w)*channels + c] = 0.f;
          
          // if (d > 2 && d < 6 && h > 2 && h < 6 && w > 2 && w < 6) {
            data[((d*height + h)*width + w)*channels + c] = c+1;
          // }
        }
      }
    }
  }

  for (int d = 0; d < depth*height*width*channels; ++d){
    std::cout << data[d] << " ";
  }
  std::cout << std::endl;
  
  octree* o = octree_create_from_dense_features_cpu(data, depth, height, width, channels, false, 0, false, 4);
  octree_print_cpu(o);
}

int main(int argc, char** argv) {    
  test_dense_features();
  return 0;
}