cmake_minimum_required(VERSION 3.12.4)
cmake_policy(VERSION 3.12.4)
cmake_policy(SET CMP0003 NEW)
cmake_policy(SET CMP0074 NEW)
project(fft CXX C ASM)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fno-pie -no-pie -Wno-attributes --fast-math")
find_package(HPX REQUIRED NO_CMAKE_PACKAGE_REGISTRY)

set(fft_source_files
    src/bit_reverse.S
    src/fft_1d_dit.S
    src/fft_1d_dif.S
    src/fft.cpp
    src/fft_3d_local.cpp
    src/fft_server.cpp
    src/fft_2d_dit.S
    src/fft_2d_dif.S
    src/main.cpp
    src/scramble.S
    src/scramble_hi.S
    src/transpose_zy.S
    src/transpose_zyx.S
    src/transpose_hi.S
    src/twiddles.cpp
 )      

set(fft_include_files
    fft/fft.hpp
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
set_property(TARGET fft PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
set_property(TARGET fft PROPERTY CUDA_SEPARABLE_COMPILATION ON)
 