
# - Try to find OctNetCore
# Once done, this will define
#
#  OctNetCore_FOUND - system has OctNetCore
#  OctNetCore_INCLUDE_DIRS - the OctNetCore include directories
#  OctNetCore_LIBRARIES - link these to use OctNetCore

include(LibFindMacros)

# Include dir
find_path(OctNetCore_INCLUDE_DIR
  NAMES octnet/core/core.h
  PATHS ./common/core/include ../core/include ../../core/include
)

# Finally the library itself
find_library(OctNetCore_LIBRARY
  NAMES octnet_core
  PATHS ./common/core/build ../core/build ../../core/build
)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(OctNetCore_PROCESS_INCLUDES OctNetCore_INCLUDE_DIR)
set(OctNetCore_PROCESS_LIBS OctNetCore_LIBRARY)
libfind_process(OctNetCore)
