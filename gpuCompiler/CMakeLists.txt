addSourcesRecursive(sources .)

add_executable(hexlibGpuCompiler ${sources})
target_include_directories(hexlibGpuCompiler PRIVATE .)
target_link_libraries(hexlibGpuCompiler PUBLIC hexlibCppBase hexlibShellCommon)

add_library(hexlibGpuCompilerInterface INTERFACE)
target_include_directories(hexlibGpuCompilerInterface INTERFACE .)
