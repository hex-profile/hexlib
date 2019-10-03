include(${HEXLIB_CMAKE_DIR}/common.cmake)

#================================================================
#
# Library of GPU processing modules.
#
#================================================================

hexlibProjectTemplate(moduleLibGpu STATIC . "imgLibGpu;moduleToolsGpu" TRUE hexlib)
