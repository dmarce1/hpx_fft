cmake_minimum_required(VERSION 3.12.4)
cmake_policy(VERSION 3.12.4)
cmake_policy(SET CMP0003 NEW)
cmake_policy(SET CMP0074 NEW)
project(fft CXX C ASM)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fno-pie -g -no-pie -Wno-attributes --fast-math")
find_package(HPX REQUIRED NO_CMAKE_PACKAGE_REGISTRY)
enable_language(CUDA)
set(CUDA_PROPAGATE_HOST_FLAGS FALSE)
execute_process(COMMAND nvcc -lcuda ${PROJECT_SOURCE_DIR}/src/cuda_detect.cu -o cuda_detect)
execute_process(COMMAND ./cuda_detect OUTPUT_VARIABLE CUDA_ARCH)
message(STATUS "CUDA Architecture: ${CUDA_ARCH}")

set(fft_source_files
    src/fft.cpp
    src/fft_server.cpp
    src/main.cpp
    src/transpose.S
)      

set(fft_include_files
    fft/fft.hpp
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
 