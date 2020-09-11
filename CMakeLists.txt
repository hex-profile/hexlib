cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)

#================================================================
#
# Common tools.
#
#================================================================

include(common.cmake)

#================================================================
#
# Tree.
#
#================================================================

add_subdirectory(extLib)

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

if (WIN32)
    add_subdirectory(testbedAt)
endif()

if ($ENV{HEXLIB_ARCHIVE_PROJECTS})
    add_subdirectory(archive)
endif()
