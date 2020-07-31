#================================================================
#
# Common app implementations for GPU-based shells.
#
#================================================================

set(dependencies imgBaseGpu moduleTools moduleToolsGpu moduleLibGpu shellCommon gpuCompilerInterface)
hexlibProjectTemplate(shellCommonGpu STATIC . "${dependencies}" TRUE hexlib)
