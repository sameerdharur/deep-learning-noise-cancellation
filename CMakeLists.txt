cmake_minimum_required(VERSION 3.10)
project(danc)

set(CMAKE_CXX_STANDARD 17)

link_directories(/home/shantanu/dev/ml/noise/tfc/lib)
add_executable(danc src/main.cpp src/WavUtils.cpp src/kiss_fft.cpp src/rnn_vad_weights.cpp src/rnn.cpp src/rnnoise.cpp)
target_include_directories(danc PRIVATE /home/shantanu/dev/ml/noise/tfc/include)
target_link_libraries(danc tensorflow)
