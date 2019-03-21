#================================================================
#
# Common app implementations for GPU-based shells.
#
#================================================================

set(usedProjects hexlibImgBaseGpu hexlibImgLibGpu hexlibModuleTools hexlibModuleToolsGpu hexlibShellCommon hexlibGpuCompilerInterface)
hexlibProjectTemplate(hexlibShellCommonGpu hexlib::shellCommonGpu "${usedProjects}")
