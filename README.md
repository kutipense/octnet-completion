# OctNet Batch Normalization

This repository contains a modified version of [griegler/octnet](https://github.com/griegler/octnet) that includes batch normalization.

The code corresponds to the CVPR'17 paper [1], the modifcations are described in [this blog article](https://davidstutz.de/convolutional-batch-normalization-for-octnets/).

    [1] Gernot Riegler, Ali Osman Ulusoy, Andreas Geiger. OctNet: Learning Deep 3D Representations at High Resolutions. CVPR, 2017.

## Code Overview

- `core` - This directory includes the core code for the hybrid grid-octree data structure (`include/octnet/core`), the CPU code for the network operations on this data structure (`include/octnet/cpu`), as well as some code to create test objects (`include/octnet/test`). 
- `core_gpu` - GPU (CUDA) code for the network operations.
- `create` - Code to pre-process 3D data (point clouds, meshes, dense volumes) and convert it to the grid-octree structure.
- `geometry` - Simple geometry routines mainly used in the `create` package.
- `py` - This directory a small python wrapper to the `create` package and some `core` functions. 
- `th` - A full featured torch wrapper for all network operations. 
- [`example`](example/01_classification_modelnet/) - Contains an example to create data and train a network on ModelNet10.


## Requirements

To build the individual projects you will need:

- `cmake` to setup the projects
- `gcc`, or `clang` to build the core project
- `nvcc` (CUDA) to compile the GPU network operations
- `cython` to compile the Python wrapper
- `torch` to setup the torch wrapper

Optional:

- `OpenMP` for the parallelization of the CPU functions

## Build

All packages, except the Python wrapper `py`, are cmake projects. Therefore, you can create a `build` directory in the individual package folder and call `cmake` and `make`. For example, to build the `core` package:

    cd core
    mkdir build
    cd build
    cmake ..
    make -j

To build the Python wrapper just do

    cd py
    python setup.py build_ext --inplace

If you do not want to repeat this for all the packages, we provide two simple bash scripts that automate this process:

- `build_cpu.sh` - builds all the CPU code for OctNet
- `build_all.sh` - same as above, but also builds the GPU network functions and the GPU wrapper code for torch

