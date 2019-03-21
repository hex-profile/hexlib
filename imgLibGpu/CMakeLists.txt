#================================================================
#
# Library of signal/image processing functions on CPU.
#
#================================================================

hexlibProjectTemplate(hexlibImgLibGpu hexlib::imgLibGpu "hexlibImgBaseGpu;hexlibImgLib;hexlibModuleToolsGpu")

# target_compile_definitions(hexlibImgLibGpu PRIVATE HEXLIB_PLATFORM=1 CUDA_ARCH=$ENV{HEXLIB_CUDA_ARCH})
target_compile_definitions(hexlibImgLibGpu PRIVATE HOSTCODE=1 DEVCODE=1)
add_dependencies(hexlibImgLibGpu hexlibGpuCompiler)

# set(CMAKE_CXX_COMPILER "hexlibGpuCompiler")
