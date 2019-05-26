//
// Created by shantanu on 7/15/18.
//

#ifndef DANC_WAVUTILS_H
#define DANC_WAVUTILS_H


#include <string>

class WavUtils {
private:
public:
  static int16_t* Read(char* filename, uint32_t* sampleRate, uint64_t* sampleCount);
  static bool Write(char* filename, int16_t* buffer, uint32_t sampleRate, uint64_t sampleCount);
  static void ResampleData(int16_t const sourceData[], int32_t sampleRate, uint32_t srcSize, int16_t destinationData[],
                    int32_t newSampleRate);

};


#endif //DANC_WAVUTILS_H
