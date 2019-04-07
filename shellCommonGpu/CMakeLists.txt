include(${HEXLIB_CMAKE_DIR}/common.cmake)

#================================================================
#
# Common app implementations for GPU-based shells.
#
#================================================================

set(dependencies hexlibImgBaseGpu hexlibImgLibGpu hexlibModuleTools hexlibModuleToolsGpu hexlibShellCommon hexlibGpuCompilerInterface)
hexlibProjectTemplate(hexlibShellCommonGpu STATIC . "${dependencies}" TRUE hexlib)
