# - Try to find OctNetCreate
# Once done, this will define
#
#  OctNetCreate_FOUND - system has OctNetCreate
#  OctNetCreate_INCLUDE_DIRS - the OctNetCreate include directories
#  OctNetCreate_LIBRARIES - link these to use OctNetCreate

include(LibFindMacros)

# Include dir
find_path(OctNetCreate_INCLUDE_DIR
  NAMES octnet/create/create.h
  PATHS ./common/create/include../create/include ../../create/include
)

# Finally the library itself
find_library(OctNetCreate_LIBRARY
  NAMES octnet_create
  PATHS ./common/create/include ../create/build ../../create/build
)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(OctNetCreate_PROCESS_INCLUDES OctNetCreate_INCLUDE_DIR)
set(OctNetCreate_PROCESS_LIBS OctNetCreate_LIBRARY)
libfind_process(OctNetCreate)
