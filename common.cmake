#================================================================
#
# checkDefs
#
#================================================================

function (checkDefs)

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
    elseif (HEXLIB_PLATFORM EQUAL 0)
        # CPU emulation
    elseif (HEXLIB_PLATFORM EQUAL 1)
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
        message(FATAL_ERROR "HEXLIB_GUARDED_MEMORY needs to be defined, use HEXLIB_GUARDED_MEMORY=0 for normal compilation.")
    endif()

    #----------------------------------------------------------------
    #
    # HEXLIB_GPU_BITNESS
    # HEXLIB_GPU_ARCH
    #
    #----------------------------------------------------------------

    if (HEXLIB_PLATFORM EQUAL 1)

        if (NOT DEFINED HEXLIB_GPU_BITNESS)
            message(FATAL_ERROR "For CUDA hardware target, HEXLIB_GPU_BITNESS should be specified (32 or 64).")
        endif()

        ###

        if (NOT DEFINED HEXLIB_GPU_ARCH)
            message(FATAL_ERROR "For CUDA hardware target, HEXLIB_GPU_ARCH should be specified as a comma-separated list of integers, for example: 20,30,35.")
        endif()

    endif()

endfunction()

#================================================================
#
# setupCuda
#
#================================================================

function (setupCuda)

    checkDefs()

    ###

    if (HEXLIB_PLATFORM EQUAL 1)

        find_program(NVCC_PATH nvcc)
        get_filename_component(cudaRoot "${NVCC_PATH}/../.." ABSOLUTE)

        math(EXPR hostBitness "8*${CMAKE_SIZEOF_VOID_P}")

        if (WIN32)
            if (hostBitness EQUAL 32)
                set(cudaLib ${cudaRoot}/lib/Win32)
            elseif (hostBitness EQUAL 64)
                set(cudaLib ${cudaRoot}/lib/x64)
            endif()
        else()
            if (hostBitness EQUAL 32)
                set(cudaLib ${cudaRoot}/lib)
            elseif (hostBitness EQUAL 64)
                set(cudaLib ${cudaRoot}/lib64)
            endif()
        endif()

        ###

        if (NOT EXISTS "${cudaRoot}/include/cuda.h")
            message(FATAL_ERROR "Problem with cudaRoot path.")
        endif()

        include_directories("${cudaRoot}/include")

        ###

        if (NOT (EXISTS "${cudaLib}/cuda.lib" OR EXISTS "${cudaLib}/libcuda.so" OR EXISTS "${cudaLib}/stubs/libcuda.so"))
            message(FATAL_ERROR "Problem with cudaLib path.")
        endif()

        link_directories(${cudaLib})
        link_directories(${cudaLib}/stubs)

    endif()

endfunction()

#================================================================
#
# "Make GPU compiler" target.
#
#================================================================

function (defineMakeGpuCompiler)

    checkDefs()

    if (NOT TARGET makeGpuCompiler)

        if (WIN32)
            set(scriptExt .cmd)
            set(binaryExt .exe)
        else()
            set(scriptExt .sh)
            set(binaryExt "")
        endif()

        ###

        set(commonCmakeDir ${CMAKE_CURRENT_LIST_DIR}) 

        ###

        add_custom_command(
            OUTPUT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/gpuCompilerBuild/gpuCC${binaryExt}"

            COMMAND 
                "${commonCmakeDir}/gpuCompiler/makeGpuCompiler${scriptExt}"
                "${CMAKE_CURRENT_SOURCE_DIR}"
                "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/gpuCompilerBuild"
                "${CMAKE_GENERATOR}"
        )

        ###

        add_custom_target(makeGpuCompiler DEPENDS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/gpuCompilerBuild/gpuCC${binaryExt})

    endif()

endfunction()

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
    
    checkDefs()

    #----------------------------------------------------------------
    #
    # Define the target.
    #
    #----------------------------------------------------------------

    foreach (dir IN LISTS sourceDirs)
        addSourcesRecursive(sources ${dir})
    endforeach()

    ###

    if (libType STREQUAL "EXECUTABLE")
        add_executable(${projectName} ${sources})
    else()
        add_library(${projectName} ${libType} ${sources})
    endif()

    #----------------------------------------------------------------
    #
    # Compiler specifics.
    #
    #----------------------------------------------------------------

    if (MSVC)
        target_compile_options(${projectName} PRIVATE "/wd5040")
        target_compile_options(${projectName} PRIVATE "/we4239")
        target_compile_definitions(${projectName} PRIVATE _CRT_SECURE_NO_WARNINGS=1)
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

    target_compile_definitions(${projectName} PRIVATE HEXLIB_PLATFORM=${HEXLIB_PLATFORM})
    target_compile_definitions(${projectName} PRIVATE HEXLIB_GUARDED_MEMORY=${HEXLIB_GUARDED_MEMORY})
    target_compile_definitions(${projectName} PRIVATE HEXLIB_GPU_BITNESS=${HEXLIB_GPU_BITNESS})
    target_compile_definitions(${projectName} PRIVATE HEXLIB_GPU_DISASM=${HEXLIB_GPU_DISASM})

    ###

    if (${requiresGpuCompiler})

        if (HEXLIB_PLATFORM EQUAL 0)

            target_compile_definitions(${projectName} PRIVATE HOSTCODE=1 DEVCODE=1)

        elseif(HEXLIB_PLATFORM EQUAL 1)

            add_dependencies(${projectName} makeGpuCompiler)

            target_compile_definitions(${projectName} PRIVATE HEXLIB_GPU_ARCH=${HEXLIB_GPU_ARCH})

            target_link_libraries(${projectName} PRIVATE cuda)

            set(CMAKE_CXX_COMPILER "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/gpuCompilerBuild/gpuCC" PARENT_SCOPE)

        else()

            message(FATAL_ERROR, "Bad HEXLIB_PLATFORM value.")

        endif()

    endif()

endfunction()

#================================================================
#
# main
#
#================================================================

setupCuda()
defineMakeGpuCompiler()
