cmake_minimum_required(VERSION 3.13.2 FATAL_ERROR)

#================================================================
#
# HEXLIB_PLATFORM
#
# 0: CPU emulation backend (slow for GPU functions).
# 1: CUDA Driver API backend.
#
#================================================================

if (NOT DEFINED HEXLIB_PLATFORM)
    message(WARNING "HEXLIB_PLATFORM is not defined, defaulting to 0 (CPU emulation)")
    set(HEXLIB_PLATFORM 0)
endif()

###

if (${HEXLIB_PLATFORM} EQUAL 0)
    # CPU emulation
elseif (${HEXLIB_PLATFORM} EQUAL 1)
    # CUDA Driver API
else()
    message(FATAL_ERROR "HEXLIB_PLATFORM=${HEXLIB_PLATFORM} is not valid")
endif()

###

set(touchUnusedVars ${HEXLIB_CUDA_ARCH} ${HEXLIB_CUDA_BITNESS})

#================================================================
#
# HEXLIB_CUDA_ARCH
#
#================================================================

if (${HEXLIB_PLATFORM} EQUAL 1)

    if (NOT DEFINED HEXLIB_CUDA_ARCH)
        set(HEXLIB_CUDA_ARCH $ENV{HEXLIB_CUDA_ARCH})
    endif()

endif()

#================================================================
#
# HEXLIB_GUARDED_MEMORY
#
#================================================================

if (NOT DEFINED HEXLIB_GUARDED_MEMORY)
    set(HEXLIB_GUARDED_MEMORY 0)
endif()

#================================================================
#
# addSourcesRecursive
#
#================================================================

function (addSourcesRecursive result dirs)

    set (internalSources "")

    foreach (dir ${dirs})
        file (GLOB_RECURSE tmp ${dir}/*.c ${dir}/*.cpp ${dir}/*.cxx)
        list (APPEND internalSources ${tmp})
    endforeach()

    ###

    set (internalHeaders "")

    foreach (dir ${dirs})
        file (GLOB_RECURSE tmp ${dir}/*.h ${dir}/*.hpp)
        list (APPEND internalHeaders ${tmp})
    endforeach()

    ###

    SET_SOURCE_FILES_PROPERTIES(${internalSources} PROPERTIES LANGUAGE CXX)

    ###

    set(resultValue ${${result}} ${internalSources} ${internalHeaders})
    set(${result} ${resultValue} PARENT_SCOPE)

endfunction()

#================================================================
#
# hexlibProjectTemplate
#
#================================================================

function (hexlibProjectTemplate projectName libType dependentProjects requiresGpuCompiler)

    addSourcesRecursive(sources .)

    ###

    add_library(${projectName} ${libType} ${sources})

    ###

    set_target_properties(${projectName} PROPERTIES FOLDER hexlib)

    target_include_directories(${projectName} PUBLIC .)

    ###

    if (dependentProjects)
        target_link_libraries(${projectName} PUBLIC ${dependentProjects})
    endif()

    ###

    target_compile_definitions(${projectName} PRIVATE HEXLIB_PLATFORM=${HEXLIB_PLATFORM} HEXLIB_GUARDED_MEMORY=${HEXLIB_GUARDED_MEMORY})

    ###

    if (MSVC)
        target_compile_definitions(${projectName} PRIVATE _CRT_SECURE_NO_WARNINGS=1)
    endif()

    ###

    if (${requiresGpuCompiler})

        if (${HEXLIB_PLATFORM} EQUAL 0)

            target_compile_definitions(${projectName} PRIVATE HOSTCODE=1 DEVCODE=1)

        elseif(${HEXLIB_PLATFORM} EQUAL 1)

            if (NOT DEFINED HEXLIB_CUDA_ARCH)
                message(FATAL_ERROR "For CUDA hardware target, HEXLIB_CUDA_ARCH should be specified (sm_20, sm_30, ...)")
            endif()

            target_compile_definitions(${projectName} PRIVATE HEXLIB_CUDA_ARCH=${HEXLIB_CUDA_ARCH})

            ###

            if (NOT DEFINED HEXLIB_CUDA_BITNESS)
                message(FATAL_ERROR "For CUDA hardware target, HEXLIB_CUDA_BITNESS should be specified (32 or 64)")
            endif()

            target_compile_definitions(${projectName} PRIVATE HEXLIB_CUDA_BITNESS=${HEXLIB_CUDA_BITNESS})
            
            ###

            target_link_libraries(${projectName} PUBLIC cuda.lib)

            set(CMAKE_CXX_COMPILER "${CMAKE_BINARY_DIR}/hexlibGpuCompiler" PARENT_SCOPE)

        else()

            message(FATAL_ERROR, "Bad HEXLIB_PLATFORM")

        endif()

    endif()

endfunction()

#================================================================
#
# tree
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
add_subdirectory(shellCommonGpu)

add_subdirectory(archive)
