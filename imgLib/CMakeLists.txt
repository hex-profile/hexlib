include(${HEXLIB_CMAKE_DIR}/common.cmake)

#================================================================
#
# Library of signal/image processing functions on CPU.
#
#================================================================

hexlibProjectTemplate(hexlibImgLib STATIC . "hexlibImgBase" FALSE hexlib)
