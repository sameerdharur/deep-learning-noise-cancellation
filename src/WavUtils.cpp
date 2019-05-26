//
// Created by shantanu on 7/15/18.
//

#include "WavUtils.h"

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

int16_t* WavUtils::Read(char* filename, uint32_t* sampleRate, uint64_t* sampleCount) {
  unsigned int channels;
  int16_t *buffer = drwav_open_and_read_file_s16(filename, &channels, sampleRate, sampleCount);
  if (channels != 1) {
    drwav_free(buffer);
    buffer = nullptr;
    *sampleRate = 0;
    *sampleCount = 0;
  }
  return buffer;
}


bool WavUtils::Write(char* filename, int16_t* buffer, uint32_t sampleRate, uint64_t sampleCount) {
  bool status = false;
  drwav_data_format format;
  format.container = drwav_container_riff;
  format.format = DR_WAVE_FORMAT_PCM;
  format.channels = 1;
  format.sampleRate = (drwav_uint32) sampleRate;
  format.bitsPerSample = 16;
  drwav *pWav = drwav_open_file_write(filename, &format);
  if (pWav) {
    drwav_uint64 samplesWritten = drwav_write(pWav, sampleCount, buffer);
    drwav_uninit(pWav);
    if (samplesWritten == sampleCount) {
      status = true;
    }
  }

  return status;
}

void WavUtils::ResampleData(int16_t const sourceData[], int32_t sampleRate, uint32_t srcSize, int16_t destinationData[],
                  int32_t newSampleRate) {
  if (sampleRate == newSampleRate) {
    std::copy(sourceData, sourceData + srcSize * sizeof(float), destinationData);
    return;
  }
  uint32_t last_pos = srcSize - 1;
  uint32_t dstSize = (uint32_t) (srcSize * ((float) newSampleRate / sampleRate));
  for (uint32_t idx = 0; idx < dstSize; idx++) {
    float index = ((float) idx * sampleRate) / (newSampleRate);
    uint32_t p1 = (uint32_t) index;
    float coef = index - p1;
    uint32_t p2 = (p1 == last_pos) ? last_pos : p1 + 1;
    destinationData[idx] = (int16_t) ((1.0f - coef) * sourceData[p1] + coef * sourceData[p2]);
  }
}

