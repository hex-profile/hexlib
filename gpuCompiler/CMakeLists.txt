cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)
project(gpuCompiler)

#----------------------------------------------------------------
#
# Compiler binary
#
#----------------------------------------------------------------

hexlibProjectTemplate(gpuCompiler EXECUTABLE . "cppBase;shellCommon" FALSE hexlib)
set_target_properties(gpuCompiler PROPERTIES OUTPUT_NAME "gpuCC")

#----------------------------------------------------------------
#
# Interface headers.
#
#----------------------------------------------------------------

add_library(gpuCompilerInterface INTERFACE)

target_include_directories(gpuCompilerInterface INTERFACE .)
