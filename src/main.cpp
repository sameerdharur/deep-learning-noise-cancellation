#include <iostream>
#include <cstring>

#include "WavUtils.h"
#include "rnn.h"
#include "rnnoise.h"

void ProcessData(int16_t* buffer, uint32_t buffer_len) {
  const int frame_size = 480;
  DenoiseState *st;
  st = rnnoise_create();
  int16_t patch_buffer[frame_size];
  if (st != NULL) {
    uint32_t frames = buffer_len / frame_size;
    uint32_t lastFrame = buffer_len % frame_size;
    for (int i = 0; i < frames; ++i) {
      //std::cout << "Processing Frame " << i << "out of " << frames << "\n";
      rnnoise_process_frame(st, buffer, buffer);
      buffer += frame_size;
    }
    if (lastFrame != 0) {
      memset(patch_buffer, 0, frame_size * sizeof(int16_t));
      memcpy(patch_buffer, buffer, lastFrame * sizeof(int16_t));
      rnnoise_process_frame(st, patch_buffer, patch_buffer);
      memcpy(buffer, patch_buffer, lastFrame * sizeof(int16_t));
    }
  }
  rnnoise_destroy(st);
}

void Denoise(char* in_file, char* out_file) {
  uint32_t in_sampleRate = 0;
  uint64_t in_size = 0;
  int16_t * data_in = WavUtils::Read(in_file, &in_sampleRate, &in_size);
  uint32_t out_sampleRate = 48000;
  auto out_size = (uint32_t) (in_size * ((float) out_sampleRate / in_sampleRate));
  auto *data_out = new int16_t[out_size];
  if (data_in != nullptr && data_out != nullptr) {
    WavUtils::ResampleData(data_in, in_sampleRate, (uint32_t) in_size, data_out, out_sampleRate);
    ProcessData(data_out, out_size);
    WavUtils::ResampleData(data_out, out_sampleRate, (uint32_t) out_size, data_in, in_sampleRate);
    WavUtils::Write(out_file, data_in, in_sampleRate, (uint32_t) in_size);
    free(data_in);
    free(data_out);
  } else {
    if (data_in) free(data_in);
    if (data_out) free(data_out);
  }
}

int main(int argc, char **argv) {
  int status = -1;
  if (argc == 3) {
    char* in_file = argv[1];
    char* out_file = argv[2];
    status = 0;
    init_rnn();
    Denoise(in_file, out_file);
    deinit_rnn();
  } else {
    std::cout << "Usage: danc <inputfile> <outputfile>";
  }
  return status;
}
