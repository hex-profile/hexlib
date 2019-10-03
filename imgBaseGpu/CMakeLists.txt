include(${HEXLIB_CMAKE_DIR}/common.cmake)

#================================================================
#
# Tools for a signal/image processing library on GPU.
#
#================================================================

hexlibProjectTemplate(imgBaseGpu STATIC . "imgBase" FALSE hexlib)
