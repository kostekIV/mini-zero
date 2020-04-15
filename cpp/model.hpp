#pragma once

#include <tensorflow/c/c_api.h>
#include <vector>
#include <string> 

class model {
private:
  TF_Session *session;
  TF_Graph *graph;
  TF_Status *status;
  TF_Output input_op;
  std::vector<TF_Output> outputs;
  std::vector<TF_Tensor*> output_tensor;
  // std::vector<std::vector<float>> out_dims;
public:
  model(const char* path, const char* tag);
  std::pair<float, std::vector<float>> predict(float* input_vals);
};
