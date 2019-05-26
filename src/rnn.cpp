#include <vector>
#include <iostream>
#include <cassert>

extern "C" {
#include <stdio.h>
#include <stdlib.h>
#include <tensorflow/c/c_api.h>
}

//https://stackoverflow.com/questions/44305647/segmentation-fault-when-using-tf-sessionrun-to-run-tensorflow-graph-in-c-not-c
TF_Graph * graph = nullptr;
TF_Status* status = nullptr;
TF_SessionOptions* sess_opts = nullptr;
TF_Session* session = nullptr;

TF_Buffer* read_file(const char* file);

void free_buffer(void* data, size_t length) {
        free(data);
}

static void Deallocator(void* data, size_t length, void* arg) {
  //free(data);
  // *reinterpret_cast<bool*>(arg) = true;
}

void compute_rnn(float* gains, float* vad, float* input_data) {
  const int num_bytes_in = 42 * sizeof(float);
  const int num_bytes_out = 22 * sizeof(float);

  int64_t in_dims[] = {1, 1, 42};
  int64_t out_dims[] = {1, 22};

  std::vector<TF_Output> inputs;
  std::vector<TF_Tensor*> input_values;

  // Pass the graph and a string name of your input operation
  // (make sure the operation name is correct)
  TF_Operation* input_op = TF_GraphOperationByName(graph, "main_input");
  TF_Output input_opout = {input_op, 0};
  inputs.push_back(input_opout);

  // Create the input tensor using the dimension (in_dims) and size (num_bytes_in)
  // variables created earlier
  TF_Tensor* input = TF_NewTensor(TF_FLOAT, in_dims, 3, input_data, num_bytes_in, &Deallocator, 0);
  input_values.push_back(input);

  // Optionally, you can check that your input_op and input tensors are correct
  // by using some of the functions provided by the C API.
  //std::cout << "Input op info: " << TF_OperationNumOutputs(input_op) << "\n";
  //std::cout << "Input data info: " << TF_Dim(input, 0) << "\n";

  // ######################
  // Set up graph outputs (similar to setting up graph inputs)
  // ######################

  // Create vector to store graph output operations
  std::vector<TF_Output> outputs;
  TF_Operation* output_op = TF_GraphOperationByName(graph, "output_node0");
  TF_Output output_opout = {output_op, 0};
  outputs.push_back(output_opout);

  // Create TF_Tensor* vector
  std::vector<TF_Tensor*> output_values(outputs.size(), nullptr);

  // Similar to creating the input tensor, however here we don't yet have the
  // output values, so we use TF_AllocateTensor()
  TF_Tensor* output_value = TF_AllocateTensor(TF_FLOAT, out_dims, 2, num_bytes_out);
  output_values.push_back(output_value);

  // As with inputs, check the values for the output operation and output tensor
  //std::cout << "Output: " << TF_OperationName(output_op) << "\n";
  //std::cout << "Output info: " << TF_Dim(output_value, 0) << "\n";

  // ######################
  // Run graph
  // ######################
  //fprintf(stdout, "Running session...\n");

  // Call TF_SessionRun
  TF_SessionRun(session, nullptr,
                &inputs[0], &input_values[0], inputs.size(),
                &outputs[0], &output_values[0], outputs.size(),
                nullptr, 0, nullptr, status);

  float* out_vals = static_cast<float*>(TF_TensorData(output_values[0]));

  std::cout << "Gains: ";
  for(int i= 0; i < 22; ++i) {
    gains[i] = out_vals[i];
    std::cout << gains[i] << " ";
  }
  std::cout << "\n";

  //fprintf(stdout, "Successfully run session\n");

}

int init_rnn() {
  // Graph definition from unzipped https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
  // which is used in the Go, Java and Android examples
  TF_Buffer* graph_def = read_file("../model/rnn_noise_new.h5.pb");
  graph = TF_NewGraph();

  // Import graph_def into graph
  status = TF_NewStatus();
  TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
  TF_GraphImportGraphDef(graph, graph_def, opts, status);
  TF_DeleteImportGraphDefOptions(opts);
  if (TF_GetCode(status) != TF_OK) {
          fprintf(stderr, "ERROR: Unable to import graph %s", TF_Message(status));
          return 1;
  }
  fprintf(stdout, "Successfully imported graph\n");
  sess_opts = TF_NewSessionOptions();
  session = TF_NewSession(graph, sess_opts, status);
  assert(TF_GetCode(status) == TF_OK);
  TF_DeleteBuffer(graph_def);

#if 0
  for(int i = 0; i < 440; ++i) {
    size_t pos = i;
    printf("%s\n", TF_OperationName(TF_GraphNextOperation(graph, &pos)));
  }
#endif
  return 0;
}

void deinit_rnn() {
  TF_CloseSession(session, status);
  TF_DeleteSession(session, status);
  TF_DeleteSessionOptions(sess_opts);
  TF_DeleteStatus(status);
  TF_DeleteGraph(graph);
}

TF_Buffer* read_file(const char* file) {
  FILE *f = fopen(file, "rb");
  fseek(f, 0, SEEK_END);
  long fsize = ftell(f);
  fseek(f, 0, SEEK_SET);  //same as rewind(f);

  void* data = malloc(fsize);
  fread(data, fsize, 1, f);
  fclose(f);

  TF_Buffer* buf = TF_NewBuffer();
  buf->data = data;
  buf->length = fsize;
  buf->data_deallocator = free_buffer;
  return buf;
}
