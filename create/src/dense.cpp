#include "octnet/create/dense.h"

#include <cstdio>
#include <cstdlib>


void dense_occupancy_to_surface(const ot_data_t* dense, int depth, int height, int width, int n_iter, ot_data_t* surface) {

  if(n_iter != 1) {
    printf("[ERROR] n_iter != 1 not implemented\n");
    exit(-1);
  }

  for(int iter = 0; iter < n_iter; ++iter) {
    for(int in_idx = 0; in_idx < depth*height*width; ++in_idx) {
      if(dense[in_idx] == 0) {
        surface[in_idx] = 0;
        continue;
      }

      int iw = in_idx % (width);
      int id = in_idx / (width * height);
      int ih = ((in_idx - iw) / width) % height;

      bool surf = false;
      for(int od = id-1; od < id+2 && !surf; ++od) {  
        for(int oh = ih-1; oh < ih+2 && !surf; ++oh) {  
          for(int ow = iw-1; ow < iw+2 && !surf; ++ow) {  
            int border_idx = (od * height + oh) * width + ow;
            if((od != id || oh != ih || ow != iw) && 
               (od < 0 || oh < 0 || ow < 0 || od >= depth || oh >= height || ow >= width || dense[border_idx] == 0)) {
              surf = true;
            }
          }
        }
      }
      
      surface[in_idx] = surf ? 1 : 0;
    }
  }
}
