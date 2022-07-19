/// Simple example to test fc.cpp on a 1d classification problem.
/// 
/// @author David Stutz
/// @file 1d_classification.cpp

#include <cstdlib>
#include <iostream>
#include <cmath>
#include "octnet/core/types.h"
#include "octnet/cpu/fc.h"

/// Create a toy 1d dataset for classification.
/// 
/// @param dataset
/// @param labels
/// @param N
/// @param D
void create_dataset(ot_data_t* dataset, ot_data_t* labels, int N, int D) {
  for (int n = 0; n < N; n++) {
    float r = std::rand() / ((float) RAND_MAX);
    
    if (r > 0.5) {
      labels[n] = 1;
    }
    else {
      labels[n] = 0;
    }
    
    for (int d = 0; d < D; d++) {
      if (r > 0.5 && d >= D/2) {
        dataset[n*D + d] = 0;
      }
      else {
        dataset[n*D + d] = 1;
      }
    }
  }
}

/// Perturb the dataset using salt and pepper noise.
/// 
/// @param dataset
/// @param N
/// @param D
void perturb_dataset(ot_data_t* dataset, int N, int D) {
  for (int n = 0; n < N; n++) {
    for (int d = 0; d < D; d++) {
      float r = std::rand() / ((float) RAND_MAX);
      
      if (r < 0.025) {
        if (dataset[n*D + d] > 0) {
          dataset[n*D + d] = 1;
        }
        else {
          dataset[n*D + d] = 0;
        }
      }
      dataset[n*D + D - 1] = 9;
    }
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

void initialize_biases(ot_data_t* biases, int n) {
  for (int i = 0; i < n; i++) {
    biases[i] = 0.f;
  }
}

void initialize_weights(ot_data_t* weights, int num_inputs, int num_outputs) {
  for (int i = 0; i < num_inputs*num_outputs; i++) {
    // https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
    ot_data_t u_1 = std::rand() / ((float) RAND_MAX);
    ot_data_t u_2 = std::rand() / ((float) RAND_MAX);
    ot_data_t g = std::sqrt(-2*std::log(u_1))*std::cos(2*M_PI*u_2);
    
    // xavier initialization
    weights[i] = g*(2.f/(num_inputs + num_outputs));
  }
}

/// Example of fully connected classification.
///
/// @param argc
/// @param argv
/// @return 
int main(int argc, char** argv) {
  
  const int N = 100;
  const int batch_size = 10;
  const int D = 10;
  const int T = 250;
  const int fc1_outputs = 1;
  ot_data_t learning_rate = 0.1f;
  
  ot_data_t* dataset = new ot_data_t[N*D];
  ot_data_t* labels = new ot_data_t[N];
  
  create_dataset(dataset, labels, N, D);
  //perturb_dataset(dataset, N, D);
  
  // fc1 weights
  const int fc1_weights_size = fc1_outputs*D;
  ot_data_t* fc1_weights = new ot_data_t[fc1_weights_size];
  initialize_weights(fc1_weights, D, fc1_outputs);
  ot_data_t* fc1_biases = new ot_data_t[fc1_outputs];
  initialize_biases(fc1_biases, fc1_outputs);
  
  // fc1 gradients
  ot_data_t* grad_fc1_weights = new ot_data_t[fc1_weights_size];
  ot_data_t* grad_fc1_biases = new ot_data_t[1];
  
  // layer outputs
  ot_data_t* layer_fc1 = new ot_data_t[batch_size*fc1_outputs];
  ot_data_t* layer_sigmoid1 = new ot_data_t[batch_size*fc1_outputs];
  
  // layer gradients
  ot_data_t* grad_fc1 = new ot_data_t[batch_size*D];
  ot_data_t* grad_sigmoid1 = new ot_data_t[batch_size*fc1_outputs];
  ot_data_t* grad_loss = new ot_data_t[batch_size*1];
  
  for (int t = 0; t < T; t++) {
    int b = std::rand()%(N - batch_size);
    
    // batch
    ot_data_t* batch_data = dataset + (b*D);
    ot_data_t* batch_labels = labels + b;
    
    // forward pass:
    dense_fc_cpu(batch_data, fc1_weights, fc1_biases, batch_size, D, fc1_outputs, layer_fc1); dense_check_nan_inf_cpu(layer_fc1, batch_size*fc1_outputs, "layer_fc1");
    dense_sigmoid_cpu(layer_fc1, batch_size, fc1_outputs, layer_sigmoid1); dense_check_nan_inf_cpu(layer_sigmoid1, batch_size*fc1_outputs, "layer_sigmoid1");
    ot_data_t loss = dense_bce_cpu(layer_sigmoid1, batch_labels, batch_size, 1);
    
//    std::cout << "Predictions/targets [" << t << "]: ";
//    for (int b = 0; b < batch_size; b++) {
//      std::cout << layer_sigmoid1[b] << "/" << batch_labels [b] << ",";
//    }
//    std::cout << std::endl;
    
    initialize_zero(grad_fc1_weights, fc1_outputs*D);
    initialize_zero(grad_fc1_biases, fc1_outputs);
    
    // backward pass:
    dense_bce_bwd_cpu(layer_fc1, batch_labels, batch_size, 1, grad_loss); dense_check_nan_inf_cpu(grad_loss, batch_size*1, "grad_loss");
    dense_sigmoid_bwd_cpu(layer_sigmoid1, grad_loss, batch_size, fc1_outputs, grad_sigmoid1);dense_check_nan_inf_cpu(grad_sigmoid1, batch_size*fc1_outputs, "grad_sigmoid1");
    //dense_fc_bwd_cpu(fc1_weights, grad_sigmoid1, batch_size, D, fc1_outputs, grad_fc1); dense_check_nan_inf_cpu(grad_fc1, batch_size*D, "grad_fc1");
    dense_fc_wbwd_cpu(batch_data, grad_sigmoid1, batch_size, D, fc1_outputs, 1.f, grad_fc1_weights, grad_fc1_biases);
    
    for (int i = 0; i < fc1_weights_size; i++) {
      fc1_weights[i] -= learning_rate*grad_fc1_weights[i];
    }
    
    //fc1_biases[0] -= learning_rate*grad_fc1_biases[0];
    
    std::cout << "Error [" << t << "]: " << loss << std::endl;
  }
    
  return 0;
}

