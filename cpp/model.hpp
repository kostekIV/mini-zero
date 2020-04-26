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
  // TODO: dont hardcode tensor dims
  // and clear resources -.-
  // std::vector<std::vector<float>> out_dims;
public:
  model() {};
  model(const model&) = delete;
  model& operator=(const model&) = delete;
  model(const char* path, const char* tag);
  std::vector<std::pair<float, std::vector<float>>> predict(float* input_vals, int b_size=1);
  ~model();
};
