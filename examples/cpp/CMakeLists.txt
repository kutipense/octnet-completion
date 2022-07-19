cmake_minimum_required(VERSION 2.8)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

find_package(OctNetCore REQUIRED)
find_package(OctNetCreate REQUIRED)
#find_package(OctNetCoreCPU REQUIRED)
find_package(CUDA 6.5 REQUIRED)

# example for creating and printing 2D OctNets
include_directories(${OctNetCreate_INCLUDE_DIR} ${OctNetCore_INCLUDE_DIR})
add_executable(2d_create_print 2d_create_print.cpp)
target_link_libraries(2d_create_print ${OctNetCreate_LIBRARY} ${OctNetCore_LIBRARY})

# example of 2D convolution on OctNets
include_directories(${OctNetCreate_INCLUDE_DIR} ${OctNetCore_INCLUDE_DIR})
add_executable(2d_conv_pool 2d_conv_pool.cpp)
target_link_libraries(2d_conv_pool ${OctNetCreate_LIBRARY} ${OctNetCore_LIBRARY})

# 3d denoising of squares and spheres
include_directories(${OctNetCreate_INCLUDE_DIR} ${OctNetCore_INCLUDE_DIR})
add_executable(3d_auto_encoder 3d_auto_encoder.cpp)
target_link_libraries(3d_auto_encoder ${OctNetCreate_LIBRARY} ${OctNetCore_LIBRARY})

# 1d classification to test fully connected stuff
include_directories(${OctNetCreate_INCLUDE_DIR} ${OctNetCore_INCLUDE_DIR})
add_executable(1d_classification 1d_classification.cpp)
target_link_libraries(1d_classification ${OctNetCreate_LIBRARY} ${OctNetCore_LIBRARY})

# 3d classification of squares
include_directories(${OctNetCreate_INCLUDE_DIR} ${OctNetCore_INCLUDE_DIR})
add_executable(3d_classification 3d_classification.cpp)
target_link_libraries(3d_classification ${OctNetCreate_LIBRARY} ${OctNetCore_LIBRARY})
