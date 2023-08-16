cmake_minimum_required(VERSION 3.12.4)
cmake_policy(VERSION 3.12.4)
cmake_policy(SET CMP0003 NEW)
cmake_policy(SET CMP0074 NEW)
project(fft CXX C ASM)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fno-pie -g -no-pie -Wno-attributes --fast-math")
find_package(HPX REQUIRED NO_CMAKE_PACKAGE_REGISTRY)

set(fft_source_files
    src/fft.cpp
    src/fft_server.cpp
    src/main.cpp
    src/permuted.cpp
    src/permuted.S
    src/twiddles.cpp
    src/fft.S
    src/scramble.S
    src/transpose.S
 )      

set(fft_include_files
    fft/fft.hpp
    fft/permuted.hpp
    fft/timer.hpp
)

add_hpx_executable(
    fft
  DEPENDENCIES
    simd fftw3_threads fftw3
  SOURCES
    ${fft_source_files}
  HEADERS
    ${fft_header_files}
 )
 
target_include_directories(fft PUBLIC ${PROJECT_SOURCE_DIR})
 