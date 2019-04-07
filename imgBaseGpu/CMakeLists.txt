include(${HEXLIB_CMAKE_DIR}/common.cmake)

#================================================================
#
# Tools for a signal/image processing library on GPU.
#
#================================================================

hexlibProjectTemplate(hexlibImgBaseGpu STATIC . "hexlibImgBase" FALSE hexlib)
