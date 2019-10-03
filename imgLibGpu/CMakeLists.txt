include(${HEXLIB_CMAKE_DIR}/common.cmake)

#================================================================
#
# Library of signal/image processing functions on CPU.
#
#================================================================

hexlibProjectTemplate(imgLibGpu STATIC . "imgBaseGpu;imgLib" TRUE hexlib)
