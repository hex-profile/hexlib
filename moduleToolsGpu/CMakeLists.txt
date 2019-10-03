include(${HEXLIB_CMAKE_DIR}/common.cmake)

#================================================================
#
# Tools for an advanced image/signal processing module on GPU.
#
#================================================================

hexlibProjectTemplate(moduleToolsGpu STATIC . "imgBaseGpu;moduleTools" FALSE hexlib)
