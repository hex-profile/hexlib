cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)

#================================================================
#
# HEXLIB_CMAKE_DIR
#
#================================================================

set(HEXLIB_CMAKE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

#================================================================
#
# Tree
#
#================================================================

add_subdirectory(baseLib)

add_subdirectory(cppBase)
add_subdirectory(imgBase)
add_subdirectory(imgLib)
add_subdirectory(moduleTools)
add_subdirectory(shellCommon)

add_subdirectory(gpuCompiler)

add_subdirectory(imgBaseGpu)
add_subdirectory(moduleToolsGpu)
add_subdirectory(imgLibGpu)
add_subdirectory(moduleLibGpu)
add_subdirectory(shellCommonGpu)

add_subdirectory(testbedAt)

if ($ENV{HEXLIB_ARCHIVE_PROJECTS})
    add_subdirectory(archive)
endif()
