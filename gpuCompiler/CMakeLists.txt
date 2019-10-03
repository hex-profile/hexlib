include(${HEXLIB_CMAKE_DIR}/common.cmake)

#----------------------------------------------------------------
#
# Compiler binary
#
#----------------------------------------------------------------

hexlibProjectTemplate(gpuCompiler EXECUTABLE . "cppBase;shellCommon" FALSE hexlib)

set_target_properties(gpuCompiler PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

#----------------------------------------------------------------
#
# Interface headers.
#
#----------------------------------------------------------------

add_library(gpuCompilerInterface INTERFACE)
target_include_directories(gpuCompilerInterface INTERFACE .)
