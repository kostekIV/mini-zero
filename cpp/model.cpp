#include "model.hpp"
#include <iostream>

static void Deallocator(void* data, size_t length, void* arg) {
}

model::model(const char* path, const char* tag) {
  int ntags = 1;

  graph = TF_NewGraph();
  status = TF_NewStatus();
  auto opts = TF_NewSessionOptions();
  TF_Buffer* run_opts = NULL;

  session = TF_LoadSessionFromSavedModel(opts, run_opts, path, &tag, ntags, graph, NULL, status);
  input_op = TF_Output{TF_GraphOperationByName(graph, "serving_default_input_1"), 0};
  
  auto pol_op = TF_Output{TF_GraphOperationByName(graph, "StatefulPartitionedCall"), 0};
  auto val_op = TF_Output{TF_GraphOperationByName(graph, "StatefulPartitionedCall"), 1};

  outputs.push_back(pol_op);
  outputs.push_back(val_op);

  const std::vector<std::int64_t> pol_dims = {1, 81};
  const std::vector<std::int64_t> val_dims = {1, 1};
  auto pol_size = 100 * sizeof(float);
  auto val_size = sizeof(float);

  output_tensor.push_back(TF_AllocateTensor(TF_FLOAT, pol_dims.data(), 2, pol_size));
  output_tensor.push_back(TF_AllocateTensor(TF_FLOAT, val_dims.data(), 2, val_size));
}

std::pair<float, std::vector<float>> model::predict(float* input_vals) {
  const std::vector<std::int64_t> input_dims = {1, 9, 9, 2};
  auto input_tensor = TF_NewTensor(TF_FLOAT, input_dims.data(), 4, input_vals, 81*2*sizeof(float), &Deallocator, 0);

  TF_SessionRun(session,
                NULL, // Run options.
                &input_op, &input_tensor, 1, // Input tensors, input tensor values, number of inputs.
                &outputs[0], &output_tensor[0], 2, // Output tensors, output tensor values, number of outputs.
                NULL, 0, // Target operations, number of targets.
                NULL, // Run metadata.
                status // Output status.
                );

  auto pol = static_cast<float*>(TF_TensorData(output_tensor[0]));
  auto val = static_cast<float*>(TF_TensorData(output_tensor[1]));

  std::vector<float> pol_r(81);
  for (int i = 0; i < 81; i++) pol_r[i] = pol[i];

  return std::make_pair(val[0], pol_r);
}