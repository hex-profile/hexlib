cmake_minimum_required(VERSION 3.23.0 FATAL_ERROR)
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

hexlibProjectTemplate(gpuCompilerInterface INTERFACE . "" FALSE hexlib)
