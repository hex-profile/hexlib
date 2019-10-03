include(${HEXLIB_CMAKE_DIR}/common.cmake)

#================================================================
#
# Common app implementations for GPU-based shells.
#
#================================================================

set(dependencies imgBaseGpu moduleTools moduleToolsGpu shellCommon gpuCompilerInterface)
hexlibProjectTemplate(shellCommonGpu STATIC . "${dependencies}" TRUE hexlib)
