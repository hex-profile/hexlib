#================================================================
#
# Common app implementations for GPU-based shells.
#
#================================================================

set(dependencies hexlibImgBaseGpu hexlibImgLibGpu hexlibModuleTools hexlibModuleToolsGpu hexlibShellCommon hexlibGpuCompilerInterface)
hexlibProjectTemplate(hexlibShellCommonGpu hexlib::shellCommonGpu "${dependencies}" TRUE)
