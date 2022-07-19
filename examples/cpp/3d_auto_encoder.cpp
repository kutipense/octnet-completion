/// Example of using OctNet for denoising using a single convolutional layer
/// as auto-encoder.
///
/// @author David Stutz
/// @file 3d_auto_encoder.cpp

#define N_THREADS 4

#include <cstdlib>
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "octnet/core/types.h"
#include "octnet/core/core.h"
#include "octnet/cpu/cpu.h"
#include "octnet/cpu/dense.h"
#include "octnet/cpu/combine.h"
#include "octnet/cpu/conv.h"
#include "octnet/cpu/activations.h"
#include "octnet/cpu/loss.h"
#include "octnet/create/create.h"

/// Simple struct for storing dense volumes.
struct volume {
  ot_data_t *data; ///< underlying data
  ot_size_t depth; ///< depth of volume
  ot_size_t height; ///< height of volume
  ot_size_t width; ///< width of volume
  
  /// Constructor for save initialization and memory management.
  volume() {
    data = 0;
    depth = 0;
    height = 0;
    width = 0;
  }
  
  /// Copy constructor.
  /// 
  /// @param v
  volume(const volume &v) {
    depth = v.depth;
    height = v.height;
    width = v.width;
    
    data = new ot_data_t[depth*height*width];
    
    for (int d = 0; d < depth; d++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          operator()(d, h, w) = v(d, h, w);
        }
      }
    }
  }
  
  /// Access data entries for getting/setting data (by reference!).
  /// 
  /// @param d
  /// @param h
  /// @param w
  ot_data_t& operator()(int d, int h, int w) {
    return data[(d*height + h)*width + w];
    //return data[(n*depth + d)*height + h)*width + w];
  }
  
  const ot_data_t& operator()(int d, int h, int w) const {
    return data[(d*height + h)*width + w];
  }
  
  /// Destructor to not care about deleting memory.
  ~volume() {
    if (data != 0) {
      delete[] data;
    }
  }
};

/// Create a simple dataset of the given sizes.
/// 
/// @param N
/// @param volumes
void create_dataset(int N, std::vector<volume> &volumes) {
  
  const int depth = 8;
  const int height = 8;
  const int width = 8;
  
  volumes.clear();
  for (int i = 0; i < N; i++) {
    volume v;
    v.depth = depth;
    v.height = height;
    v.width = width;
    
    v.data = new ot_data_t[depth*height*width];
    
    for (int d = 0; d < depth; d++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          v(d, h, w) = 0.f;
        }
      }
    }
    
    for (int d = 0; d < depth; d++) {
      for (int h = 0; h < depth/2; h++) {
        for (int w = 0; w < width; w++) {
          v(d, h, w) = 1.f;
        }
      }
    }
    
    volumes.push_back(v);
  }
}

/// Create a noisy dataset from the given dataset.
/// 
/// @param dataset
/// @param noisy
void perturb_dataset(const std::vector<volume> &dataset, std::vector<volume> &noisy) {
  assert(dataset.size() > 0);
  
  const int depth = dataset[0].depth;
  const int height = dataset[0].height;
  const int width = dataset[0].width;
  
  noisy.clear();
  for (unsigned int n = 0; n < dataset.size(); n++) {
    volume v;
    v.depth = depth;
    v.height = height;
    v.width = width;
    
    v.data = new ot_data_t[depth*height*width];
    
    for (int d = 0; d < depth; d++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          v(d, h, w) = dataset[n](d, h, w);
        }
      }
    }
    
    for (int d = 0; d < depth; d++) {
      for (int h = 0; h < height/2; h++) {
        for (int w = 0; w < width; w++) {
          float r = std::rand() / ((float) RAND_MAX);
          if (r < 0.05) {
            v(d, h, w) = 0;
          }
        }
      }
    }
    
    noisy.push_back(v);
  }
}

/// Convert the dataset to individual octrees; does not bundle the volumes into
/// batches - this is done randomly during training.
/// 
/// @param dataset
/// @param octrees
void convert_dataset(const std::vector<volume> &dataset, octree** octrees) {
  
  const int n_ranges = 1;
  ot_data_t* ranges = new ot_data_t[2];
  ranges[0] = 0.5f;
  ranges[1] = 1.5f;
  
  for (unsigned int n = 0 ; n < dataset.size(); n++) {
    octree* o = octree_create_from_dense_cpu(dataset[n].data, dataset[n].depth, dataset[n].height, dataset[n].width, n_ranges, ranges, false, 0, false, N_THREADS);
    octrees[n] = o;
  }
}

/// Initialize the weights for convolutional layers.
/// 
/// @param weights
/// @param n
/// @param neurons_in
/// @param neurons_out
void initialize_weights(ot_data_t* weights, int n, int neurons_in, int neurons_out) {
  for (int i = 0; i < n; i++) {
    // https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
    ot_data_t u_1 = std::rand() / ((float) RAND_MAX);
    ot_data_t u_2 = std::rand() / ((float) RAND_MAX);
    ot_data_t g = std::sqrt(-2*std::log(u_1))*std::cos(2*M_PI*u_2);
    
    // xavier initialization
    weights[i] = g*(2.f/(neurons_in + neurons_out));
  }
}

/// Initialize biases.
/// 
/// @param biases
/// @param n
void initialize_biases(ot_data_t* biases, int n) {
  for (int i = 0; i < n; i++) {
    biases[i] = 0.5f;
  }
}

/// Initialize an array with zero.
/// 
/// @param array
/// @param n
void initialize_zero(ot_data_t* array, int n) {
  for (int i = 0; i < n; i++) {
    array[i] = 0;
  }
}

/// Compute magnitude of weight array.
///
/// @param array
/// @param n
/// @param scale
/// @return 
ot_data_t compute_magnitude(ot_data_t* array, int n, ot_data_t scale = 1.f) {
  ot_data_t mag = 0.f;
  for (int i = 0; i < n; i++) {
    mag += (1.f/scale)*(1.f/scale)*array[i]*array[i];
  }
  return std::sqrt(mag);
}

/// Clip gradients by factor (multiplication).
/// 
/// @param array
/// @param n
/// @param factor
void clip_gradients(ot_data_t* array, int n, ot_data_t factor) {
  for (int i = 0; i < n; i++) {
    array[i] *= factor;
  }
}

/// Display output and target side by side.
/// 
/// @param output
/// @param target
void display_output_target(const octree* output, const octree* target) {
  assert(output->n == target->n);
  assert(output->grid_depth == target->grid_depth);
  assert(output->grid_height == target->grid_height);
  assert(output->grid_height == target->grid_height);
  
  const int depth = 8;
  const int height = 8;
  const int width = 8;
  
  ot_data_t* dense_output = new ot_data_t[output->n*depth*height*width];
  ot_data_t* dense_target = new ot_data_t[output->n*depth*height*width];
  
  octree_to_dhwc_cpu(output, 8, 8, 8, dense_output);
  octree_to_dhwc_cpu(target, 8, 8, 8, dense_target);
  
  for (int n = 0; n < output->n; n++) {
    for (int d = 0; d < depth; d++) {
      std::cout << "-- " << d << std::endl;
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          std::cout << dense_output[(((n*depth + d)*height + h)*width + w)] << ",";
        }
        
        std::cout << "-";
        for (int w = 0; w < width; w++) {
          std::cout << dense_target[(((n*depth + d)*height + h)*width + w)] << ",";
        }
        std::cout << std::endl;
      }
    }
  }
  
  delete[] dense_output;
  delete[] dense_target;
}

/// Auto-encoder training for denoising 8x8x8 volumes.
///
/// @param argc
/// @param argv
/// @return
int main(int argc, char** argv) {

  // note: the below parameters doe currently work (at least when I tried it last).
  // the problem seems to be very sensitive to the parameters and the initialization
  // there is also still some oscillation going on I didn't get rid of and
  // convergence is pretty slow
  
  const int N = 1000; ///< number of training samples
  const int T = 250; ///< number of iterations
  const ot_data_t gradient_clipping = 1e18; ///< clip gradients if the magnitude exceeds this number
  const ot_data_t scale = 1.f; // scale for conv backward passes
  const int conv1_channels = 24; ///< numbe rof channels generated by conv3 layer
  const int conv2_channels = 8; ///< number of channels generated by conv2 layer
  const int batch_size = 16; ///< batch size to use during training
  const int interval = T/10; ///< interval to print a batch and update learning_rate/momentum (currently commented out!)
  ot_data_t learning_rate = 0.05f; ///< learning rate for learning
  ot_data_t momentum = 0.f; ///< momentum term for learning
  
  std::vector<volume> dataset;
  create_dataset(N, dataset);
  
  std::vector<volume> noisy_dataset;
  perturb_dataset(dataset, noisy_dataset);
  
  octree** octrees = new octree*[N]; // !
  convert_dataset(dataset, octrees);
  
  octree** noisy_octrees = new octree*[N];
  convert_dataset(noisy_dataset, noisy_octrees);
  
  // weights for conv1
  const int conv1_weights_size = conv1_channels*1*3*3*3;
  ot_data_t* conv1_weights = new ot_data_t[conv1_weights_size];
  initialize_weights(conv1_weights, conv1_weights_size, 3*3*3, conv1_weights_size);
  ot_data_t* conv1_biases = new ot_data_t[conv1_channels];
  initialize_biases(conv1_biases, conv1_channels);
  
  // gradients for conv1 (are initialized every iteration)
  ot_data_t* grad_conv1_weights = new ot_data_t[conv1_weights_size];
  ot_data_t* grad_conv1_biases = new ot_data_t[conv1_channels];
  
  // momentum for conv1
  ot_data_t* momentum_conv1_weights = new ot_data_t[conv1_weights_size];
  initialize_zero(momentum_conv1_weights, conv1_weights_size);
  ot_data_t* momentum_conv1_biases = new ot_data_t[conv1_channels];
  initialize_zero(momentum_conv1_biases, conv1_channels);
  
  // weights for conv2
  const int conv2_weights_size = conv2_channels*conv1_channels*3*3*3;
  ot_data_t* conv2_weights = new ot_data_t[conv2_weights_size];
  initialize_weights(conv2_weights, conv2_weights_size, conv1_weights_size, conv2_weights_size);
  ot_data_t* conv2_biases = new ot_data_t[conv2_channels];
  initialize_biases(conv2_biases, conv2_channels);
  
  // gradients for conv2 (are initialized every iteration)
  ot_data_t* grad_conv2_weights = new ot_data_t[conv2_weights_size];
  ot_data_t* grad_conv2_biases = new ot_data_t[conv2_channels];
  
  // momentum for conv2
  ot_data_t* momentum_conv2_weights = new ot_data_t[conv2_weights_size];
  initialize_zero(momentum_conv2_weights, conv2_weights_size);
  ot_data_t* momentum_conv2_biases = new ot_data_t[conv2_channels];
  initialize_zero(momentum_conv2_biases, conv2_channels);
  
  // weights for conv3
  const int conv3_weights_size = 1*conv2_channels*3*3*3;
  ot_data_t* conv3_weights = new ot_data_t[conv3_weights_size];
  initialize_weights(conv3_weights, conv3_weights_size, conv2_weights_size, conv3_weights_size);
  ot_data_t* conv3_biases = new ot_data_t[1];
  initialize_biases(conv3_biases, 1);
  
  // gradients for conv3 (are initialized every iteration)
  ot_data_t* grad_conv3_weights = new ot_data_t[conv3_weights_size];
  ot_data_t* grad_conv3_biases = new ot_data_t[1];
  
  // momentum for conv3
  ot_data_t* momentum_conv3_weights = new ot_data_t[conv3_weights_size];
  initialize_zero(momentum_conv3_weights, conv3_weights_size);
  ot_data_t* momentum_conv3_biases = new ot_data_t[1];
  initialize_zero(momentum_conv3_biases, 1);
  
  // these will be intermediate octrees and arrays that are reused
  octree* layer_conv1 = octree_new_cpu();
  octree* layer_conv2 = octree_new_cpu();
  octree* layer_conv3 = octree_new_cpu();
  octree* layer_sigmoid1 = octree_new_cpu();
  octree* layer_sigmoid2 = octree_new_cpu();
  octree* layer_sigmoid3 = octree_new_cpu();
  octree* grad_loss = octree_new_cpu();
  octree* grad_sigmoid1 = octree_new_cpu();
  octree* grad_sigmoid2 = octree_new_cpu();
  octree* grad_sigmoid3 = octree_new_cpu();
  octree* grad_conv1 = octree_new_cpu();
  octree* grad_conv2 = octree_new_cpu();
  octree* grad_conv3 = octree_new_cpu();
  
  for (int t, tt = 0; t < T; t++, tt++) {
    int b = std::rand()%(N - batch_size);
    
    // create a batch of the set size by combining octrees (not completely
    // randomly selected yet)
    octree* batch = octree_new_cpu();
    octree_combine_n_cpu(&octrees[b], batch_size, batch);
    octree* noisy_batch = octree_new_cpu();
    octree_combine_n_cpu(&noisy_octrees[b], batch_size, noisy_batch);
            
    // do the forward pass
    octree_conv3x3x3_avg_cpu(noisy_batch, conv1_weights, conv1_biases, conv1_channels, layer_conv1); octree_check_nan_inf_cpu(layer_conv1, "layer_conv1");
    octree_sigmoid_cpu(layer_conv1, false, layer_sigmoid1); octree_check_nan_inf_cpu(layer_sigmoid1, "layer_sigmoid1");
    
    octree_conv3x3x3_avg_cpu(layer_sigmoid1, conv2_weights, conv2_biases, conv2_channels, layer_conv2); octree_check_nan_inf_cpu(layer_conv2, "layer_conv2");
    octree_sigmoid_cpu(layer_conv2, false, layer_sigmoid2); octree_check_nan_inf_cpu(layer_sigmoid2, "layer_sigmoid2");
    
    octree_conv3x3x3_avg_cpu(layer_sigmoid2, conv3_weights, conv3_biases, 1, layer_conv3); octree_check_nan_inf_cpu(layer_conv3, "layer_conv3");
    octree_sigmoid_cpu(layer_conv3, false, layer_sigmoid3); octree_check_nan_inf_cpu(layer_sigmoid3, "layer_sigmoid3");
    
    const bool size_average = true;
    const bool check = true;
    ot_data_t error = octree_mse_loss_cpu(layer_sigmoid3, batch, size_average, size_average);
    
    // seems like the gradient arrays need to be initialized!
    initialize_zero(grad_conv1_weights, conv1_weights_size);
    initialize_zero(grad_conv1_biases, conv1_channels);
    
    initialize_zero(grad_conv2_weights, conv2_weights_size);
    initialize_zero(grad_conv2_biases, conv2_channels);
    
    initialize_zero(grad_conv3_weights, conv3_weights_size);
    initialize_zero(grad_conv3_biases, 1);
    
    // do the backward pass
    octree_mse_loss_bwd_cpu(layer_sigmoid3, batch, size_average, check, grad_loss); octree_check_nan_inf_cpu(grad_loss, "grad_loss");
    
    octree_sigmoid_bwd_cpu(layer_conv3, layer_sigmoid3, grad_loss, false, grad_sigmoid3); octree_check_nan_inf_cpu(grad_sigmoid3, "grad_sigmoid3");
    octree_conv3x3x3_avg_bwd_cpu(conv3_weights, grad_sigmoid3, conv2_channels, grad_conv3); octree_check_nan_inf_cpu(grad_conv3, "grad_conv3");
    octree_conv3x3x3_avg_wbwd_cpu(layer_sigmoid2, grad_sigmoid3, 1.f, grad_conv3_weights, grad_conv3_biases);
    
    octree_sigmoid_bwd_cpu(layer_conv2, layer_sigmoid2, grad_conv3, false, grad_sigmoid2); octree_check_nan_inf_cpu(grad_sigmoid2, "grad_sigmoid2");
    octree_conv3x3x3_avg_bwd_cpu(conv2_weights, grad_sigmoid2, conv1_channels, grad_conv2); octree_check_nan_inf_cpu(grad_conv2, "grad_conv2");
    octree_conv3x3x3_avg_wbwd_cpu(layer_sigmoid1, grad_sigmoid2, 1.f, grad_conv2_weights, grad_conv2_biases);
    
    octree_sigmoid_bwd_cpu(layer_conv1, layer_sigmoid1, grad_conv2, false, grad_sigmoid1); octree_check_nan_inf_cpu(grad_sigmoid1, "grad_sigmoid1");
    octree_conv3x3x3_avg_bwd_cpu(conv1_weights, grad_sigmoid1, 1, grad_conv1); octree_check_nan_inf_cpu(grad_conv1, "grad_conv1");
    octree_conv3x3x3_avg_wbwd_cpu(noisy_batch, grad_sigmoid1, 1.f, grad_conv1_weights, grad_conv1_biases);
    
    // Adapt learning rate and momentum.
    if (tt >= interval) {
      //learning_rate *= 0.5;
      //momentum = std::min(momentum*1.025, 0.95);
      std::cout << "Learning rate [" << t << "]: " << learning_rate << std::endl;
      std::cout << "Momentum [" << t << "]: " << momentum << std::endl;
      
      display_output_target(layer_sigmoid3, batch);
      tt = 0;
    }
    
    ot_data_t grad_conv1_weights_mag = compute_magnitude(grad_conv1_weights, conv1_weights_size, scale);
    if (grad_conv1_weights_mag > gradient_clipping) {
      clip_gradients(grad_conv1_weights, conv1_weights_size, 1.f/grad_conv1_weights_mag*gradient_clipping);
    }
    
    ot_data_t grad_conv2_weights_mag = compute_magnitude(grad_conv2_weights, conv2_weights_size, scale);
    if (grad_conv2_weights_mag > gradient_clipping) {
      clip_gradients(grad_conv2_weights, conv2_weights_size, 1.f/grad_conv2_weights_mag*gradient_clipping);
    }
    
    ot_data_t grad_conv3_weights_mag = compute_magnitude(grad_conv3_weights, conv3_weights_size, scale);
    if (grad_conv3_weights_mag > gradient_clipping) {
      clip_gradients(grad_conv3_weights, conv3_weights_size, 1.f/grad_conv3_weights_mag*gradient_clipping);
    }
    
    for (int i = 0; i < conv1_weights_size; i++) {
      momentum_conv1_weights[i] = momentum*momentum_conv1_weights[i] - 1.f/scale*learning_rate*grad_conv1_weights[i];
      conv1_weights[i] += momentum_conv1_weights[i];
    }
    
    for (int i = 0; i < conv2_weights_size; i++) {
      momentum_conv2_weights[i] = momentum*momentum_conv2_weights[i] - 1.f/scale*learning_rate*grad_conv2_weights[i];
      conv2_weights[i] += momentum_conv2_weights[i];
    }
    
    for (int i = 0; i < conv3_weights_size; i++) {
      momentum_conv3_weights[i] = momentum*momentum_conv3_weights[i] - 1.f/scale*learning_rate*grad_conv3_weights[i];
      conv3_weights[i] += momentum_conv3_weights[i];
    }
    
    ot_data_t grad_conv1_biases_mag = compute_magnitude(grad_conv1_biases, conv1_channels, scale);
    if (grad_conv1_biases_mag > gradient_clipping) {
      clip_gradients(grad_conv1_biases, conv1_channels, 1.f/grad_conv1_biases_mag*gradient_clipping);
    }
    
    ot_data_t grad_conv2_biases_mag = compute_magnitude(grad_conv2_biases, conv2_channels, scale);
    if (grad_conv2_biases_mag > gradient_clipping) {
      clip_gradients(grad_conv2_biases, conv2_channels, 1.f/grad_conv2_biases_mag*gradient_clipping);
    }
    
    ot_data_t grad_conv3_biases_mag = compute_magnitude(grad_conv3_biases, 1, scale);
    if (grad_conv3_biases_mag > gradient_clipping) {
      clip_gradients(grad_conv3_biases, 1, 1.f/grad_conv3_biases_mag*gradient_clipping);
    }
    
    for (int i = 0; i < conv1_channels; i++) {
      momentum_conv1_biases[i] = momentum*momentum_conv1_biases[i] - 1.f/scale*learning_rate*grad_conv1_biases[i];
      conv1_biases[i] += momentum_conv1_biases[i];
    }
    
    for (int i = 0; i < conv2_channels; i++) {
      momentum_conv2_biases[i] = momentum*momentum_conv2_biases[i] - 1.f/scale*learning_rate*grad_conv2_biases[i];
      conv2_biases[i] += momentum_conv2_biases[i];
    }
    
    momentum_conv3_biases[0] = momentum*momentum_conv3_biases[0] - 1.f/scale*learning_rate*grad_conv3_biases[0];
    conv3_biases[0] += momentum_conv3_biases[0];
    
    std::cout << "Error [" << t << "]: " << error << std::endl;
    std::cout << "Gradient Magnitude [" << t << "]: " << grad_conv1_weights_mag
            << " | " << grad_conv2_weights_mag
            << " | " << grad_conv3_weights_mag << std::endl;
    
    octree_free_cpu(batch);
    octree_free_cpu(noisy_batch);
  }
  
  for (int n = 0; n < N; n++) {
    octree_free_cpu(octrees[n]);
    octree_free_cpu(noisy_octrees[n]);
  }
  
  octree_free_cpu(layer_conv1);
  octree_free_cpu(layer_conv2);
  octree_free_cpu(layer_conv3);
  octree_free_cpu(layer_sigmoid1);
  octree_free_cpu(layer_sigmoid2);
  octree_free_cpu(layer_sigmoid3);
  octree_free_cpu(grad_loss);
  octree_free_cpu(grad_sigmoid1);
  octree_free_cpu(grad_sigmoid2);
  octree_free_cpu(grad_sigmoid3);
  octree_free_cpu(grad_conv1);
  octree_free_cpu(grad_conv2);
  octree_free_cpu(grad_conv3);
  
  delete[] conv1_weights;
  delete[] conv1_biases;
  delete[] grad_conv1_weights;
  delete[] grad_conv1_biases;
  delete[] momentum_conv1_weights;
  delete[] momentum_conv1_biases;
  
  delete[] conv2_weights;
  delete[] conv2_biases;
  delete[] grad_conv2_weights;
  delete[] grad_conv2_biases;
  delete[] momentum_conv2_weights;
  delete[] momentum_conv2_biases;
  
  delete[] conv3_weights;
  delete[] conv3_biases;
  delete[] grad_conv3_weights;
  delete[] grad_conv3_biases;
  delete[] momentum_conv3_weights;
  delete[] momentum_conv3_biases;
  
  // delete octrees !
  return 0;
}

