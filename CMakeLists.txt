cmake_minimum_required (VERSION 3.8)
cmake_policy(SET CMP0074 NEW)

project(cudaMPCD LANGUAGES CXX CUDA)

find_package(HDF5 REQUIRED COMPONENTS CXX)
include_directories(include PUBLIC ${HDF5_INCLUDE_DIRS})

if(DEFINED ENV{CUDA})
    include_directories(PUBLIC $ENV{CUDA}/include) 
endif()

add_executable(main
    src/main.cpp 
    src/h5cpp.cpp 
    src/cuda_allocator.cpp 
    src/simulation_context.cu 
    src/gpu_functions.cu 
    src/extended_collision.cu 
    src/gpu_constants.cu
)

set_property(TARGET main PROPERTY CUDA_ARCHITECTURES 61 72)
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET main PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_features(main PUBLIC cxx_std_17)
target_link_libraries(main PUBLIC ${HDF5_LIBRARIES})
