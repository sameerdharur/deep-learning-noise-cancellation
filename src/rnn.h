//
// Created by shantanu on 7/29/18.
//

#ifndef DANC_RNN_H
#define DANC_RNN_H
int init_rnn(void);
void deinit_rnn(void);
void compute_rnn(float* gains, float* vad, float* input);

#endif //DANC_RNN_H
