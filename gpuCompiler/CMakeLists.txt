include(${HEXLIB_CMAKE_DIR}/common.cmake)

#----------------------------------------------------------------
#
# Compiler binary
#
#----------------------------------------------------------------

hexlibProjectTemplate(hexlibGpuCompiler EXECUTABLE . "hexlibCppBase;hexlibShellCommon" FALSE hexlib)

set_target_properties(hexlibGpuCompiler PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

#----------------------------------------------------------------
#
# Interface headers.
#
#----------------------------------------------------------------

add_library(hexlibGpuCompilerInterface INTERFACE)
target_include_directories(hexlibGpuCompilerInterface INTERFACE .)
