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

    set_source_files_properties(${internalSources} PROPERTIES LANGUAGE CXX)

    ###

    set(resultValue ${${result}} ${internalSources} ${internalHeaders})
    set(${result} ${resultValue} PARENT_SCOPE)

endfunction()

#================================================================
#
# hexlibProjectTemplate
#
#================================================================

function (hexlibProjectTemplate projectName libType sourceDirs dependentProjects requiresGpuCompiler folderName)

    cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)
    
    #----------------------------------------------------------------
    #
    # Target
    #
    #----------------------------------------------------------------

    foreach (dir IN LISTS sourceDirs)
        addSourcesRecursive(sources ${dir})
    endforeach()

    ###

    if (${libType} STREQUAL "EXECUTABLE")
        add_executable(${projectName} ${sources})
    else()
        add_library(${projectName} ${libType} ${sources})
    endif()

    #----------------------------------------------------------------
    #
    # Compiler
    #
    #----------------------------------------------------------------

    if (${HEXLIB_PLATFORM} EQUAL 1)
        set(cppStd 14)
    else()
        set(cppStd 17)
    endif()

    ###

    set_target_properties(${projectName} PROPERTIES CXX_STANDARD ${cppStd})

    ###

    if (MSVC)
        target_compile_options(${projectName} PRIVATE "/wd5040")
        target_compile_options(${projectName} PRIVATE "/we4239")
        target_compile_definitions(${projectName} PRIVATE _CRT_SECURE_NO_WARNINGS=1)
    endif()

    #----------------------------------------------------------------
    #
    # HEXLIB_PLATFORM
    #
    # 0: CPU emulation backend (slow for GPU functions).
    # 1: CUDA Driver API backend.
    #
    #----------------------------------------------------------------

    if (NOT DEFINED HEXLIB_PLATFORM)
        message(FATAL_ERROR "HEXLIB_PLATFORM needs to be defined, use HEXLIB_PLATFORM=0 for CPU emulation.")
    elseif (${HEXLIB_PLATFORM} EQUAL 0)
        # CPU emulation
    elseif (${HEXLIB_PLATFORM} EQUAL 1)
        # CUDA Driver API
    else()
        message(FATAL_ERROR "HEXLIB_PLATFORM=${HEXLIB_PLATFORM} is not valid.")
    endif()

    #----------------------------------------------------------------
    #
    # HEXLIB_GUARDED_MEMORY
    #
    #----------------------------------------------------------------

    if (NOT DEFINED HEXLIB_GUARDED_MEMORY)
        set(HEXLIB_GUARDED_MEMORY 0)
    endif()

    #----------------------------------------------------------------
    #
    # HEXLIB_GPU_BITNESS
    # HEXLIB_CUDA_ARCH
    #
    #----------------------------------------------------------------

    if(${HEXLIB_PLATFORM} EQUAL 1)

        if (NOT DEFINED HEXLIB_GPU_BITNESS)
            message(FATAL_ERROR "For GPU hardware target, HEXLIB_GPU_BITNESS should be specified (32 or 64).")
        endif()

        ###

        if (NOT DEFINED HEXLIB_CUDA_ARCH)
            set(HEXLIB_CUDA_ARCH $ENV{HEXLIB_CUDA_ARCH})
        endif()

        if ((NOT DEFINED HEXLIB_CUDA_ARCH) OR (HEXLIB_CUDA_ARCH STREQUAL ""))
            message(FATAL_ERROR "For CUDA hardware target, HEXLIB_CUDA_ARCH should be specified (sm_20, sm_30, ...).")
        endif()

    endif()

    #----------------------------------------------------------------
    #
    # Target properties
    #
    #----------------------------------------------------------------

    if (folderName)
        set_target_properties(${projectName} PROPERTIES FOLDER ${folderName})
    endif()

    target_include_directories(${projectName} PUBLIC .)

    ###

    if (dependentProjects)
        target_link_libraries(${projectName} PUBLIC ${dependentProjects})
    endif()

    ###

    target_compile_definitions(${projectName} PRIVATE HEXLIB_PLATFORM=${HEXLIB_PLATFORM} HEXLIB_GUARDED_MEMORY=${HEXLIB_GUARDED_MEMORY})
    target_compile_definitions(${projectName} PRIVATE HEXLIB_GPU_BITNESS=${HEXLIB_GPU_BITNESS})

    ###

    if (${requiresGpuCompiler})

        if (${HEXLIB_PLATFORM} EQUAL 0)

            target_compile_definitions(${projectName} PRIVATE HOSTCODE=1 DEVCODE=1)

        elseif(${HEXLIB_PLATFORM} EQUAL 1)

            add_dependencies(${projectName} makeGpuCompiler)

            target_compile_definitions(${projectName} PRIVATE HEXLIB_CUDA_ARCH=${HEXLIB_CUDA_ARCH})

            target_link_libraries(${projectName} PUBLIC cuda)

            set(CMAKE_CXX_COMPILER "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/gpuCompiler" PARENT_SCOPE)

        else()

            message(FATAL_ERROR, "Bad HEXLIB_PLATFORM value.")

        endif()

    endif()

endfunction()
