include(${HEXLIB_CMAKE_DIR}/common.cmake)

#----------------------------------------------------------------
#
# Compiler binary
#
#----------------------------------------------------------------

addSourcesRecursive(sources .)
add_executable(hexlibGpuCompiler ${sources})

target_include_directories(hexlibGpuCompiler PRIVATE .)

target_link_libraries(hexlibGpuCompiler PUBLIC hexlibCppBase hexlibShellCommon)

#----------------------------------------------------------------
#
# Interface headers.
#
#----------------------------------------------------------------

add_library(hexlibGpuCompilerInterface INTERFACE)
target_include_directories(hexlibGpuCompilerInterface INTERFACE .)

#----------------------------------------------------------------
#
# Copy binary to known path.
#
#----------------------------------------------------------------

set_target_properties(hexlibGpuCompiler PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
