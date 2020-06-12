#include "model.hpp"
#include <iostream>
#include <cstring>

static void Deallocator(void* data, size_t length, void* arg) {
  free(data);
}

model::model(const char* path, const char* tag) {
  int ntags = 1;

  graph = TF_NewGraph();
  status = TF_NewStatus();
  auto opts = TF_NewSessionOptions();
  TF_Buffer* run_opts = NULL;

  session = TF_LoadSessionFromSavedModel(opts, run_opts, path, &tag, ntags, graph, NULL, status);
  TF_DeleteSessionOptions(opts);

  input_op = TF_Output{TF_GraphOperationByName(graph, "serving_default_input_1"), 0};
  
  auto pol_op = TF_Output{TF_GraphOperationByName(graph, "StatefulPartitionedCall"), 0};
  auto val_op = TF_Output{TF_GraphOperationByName(graph, "StatefulPartitionedCall"), 1};

  outputs.push_back(pol_op);
  outputs.push_back(val_op);
}

std::vector<std::pair<float, std::vector<float>>> model::predict(float* input_vals, int b_size) {

  const std::vector<std::int64_t> input_dims = {b_size, 10, 10, 2};
  auto input_tensor = TF_AllocateTensor(TF_FLOAT, input_dims.data(), 4, b_size * 200 * sizeof(float));

  const std::vector<std::int64_t> pol_dims = {b_size, 100};
  const std::vector<std::int64_t> val_dims = {b_size, 1};
  auto pol_size = b_size * 100 * sizeof(float);
  auto val_size = b_size * sizeof(float);

  std::vector<TF_Tensor*> output_tensor;
  output_tensor.push_back(TF_AllocateTensor(TF_FLOAT, pol_dims.data(), 2, pol_size));
  output_tensor.push_back(TF_AllocateTensor(TF_FLOAT, val_dims.data(), 2, val_size));
  auto data = TF_TensorData(input_tensor);
  std::memcpy(data, input_vals, TF_TensorByteSize(input_tensor));
  TF_SessionRun(session,
                NULL, // Run options.
                &input_op, &input_tensor, 1, // Input tensors, input tensor values, number of inputs.
                &outputs[0], &output_tensor[0], 2, // Output tensors, output tensor values, number of outputs.
                NULL, 0, // Target operations, number of targets.
                NULL, // Run metadata.
                status // Output status.
                );

  std::vector<std::pair<float, std::vector<float>>> result;
  auto pol = static_cast<float*>(TF_TensorData(output_tensor[0]));
  auto val = static_cast<float*>(TF_TensorData(output_tensor[1]));
  for (int j = 0; j < b_size; j++) {
    std::vector<float> pol_r(100, 0);
    for (int i = 0; i < 100; i++) pol_r[i] = pol[j*100 + i];
    result.push_back(std::make_pair(val[j], pol_r));
  }

  TF_DeleteTensor(output_tensor[0]);
  TF_DeleteTensor(output_tensor[1]);
  TF_DeleteTensor(input_tensor);
  return result;
}

model::~model() {
  TF_DeleteGraph(graph);
  TF_DeleteSession(session, status);
  TF_DeleteStatus(status);
}
