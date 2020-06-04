include(${HEXLIB_CMAKE_DIR}/common.cmake)

#----------------------------------------------------------------
#
# Compiler binary
#
#----------------------------------------------------------------

hexlibProjectTemplate(gpuCompiler EXECUTABLE . "cppBase;shellCommon" FALSE hexlib)

#----------------------------------------------------------------
#
# Interface headers.
#
#----------------------------------------------------------------

add_library(gpuCompilerInterface INTERFACE)

target_include_directories(gpuCompilerInterface INTERFACE .)
