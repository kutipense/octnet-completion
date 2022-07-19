# - Try to find OctNetCoreGPU
# Once done, this will define
#
#  OctNetCoreGPU_FOUND - system has OctNetCoreGPU
#  OctNetCoreGPU_INCLUDE_DIRS - the OctNetCoreGPU include directories
#  OctNetCoreGPU_LIBRARIES - link these to use OctNetCoreGPU

include(LibFindMacros)

# Include dir
find_path(OctNetCoreGPU_INCLUDE_DIR
  NAMES octnet/core/core.h
  PATHS ./common/core_gpu/include ../core_gpu/include ../../core_gpu/include
)

# Finally the library itself
find_library(OctNetCoreGPU_LIBRARY
  NAMES octnet_core
  PATHS ./common/core_gpu/build ../core_gpu/build ../../core_gpu/build
)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(OctNetCoreGPU_PROCESS_INCLUDES OctNetCoreGPU_INCLUDE_DIR)
set(OctNetCoreGPU_PROCESS_LIBS OctNetCoreGPU_LIBRARY)
libfind_process(OctNetCoreGPU)
