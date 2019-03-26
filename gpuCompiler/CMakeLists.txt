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

add_custom_command(
    TARGET hexlibGpuCompiler POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
         $<TARGET_FILE:hexlibGpuCompiler>
        ${CMAKE_BINARY_DIR}/$<TARGET_FILE_NAME:hexlibGpuCompiler>)
