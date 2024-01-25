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
            message(FATAL_ERROR "For CUDA hardware target, HEXLIB_GPU_ARCH should be specified as a colon-separated list of integers, for example: 20:30:35.")
        endif()

    endif()

endfunction()

#================================================================
#
# setupCuda
#
#================================================================

function (setupCuda projectName)

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

        target_compile_definitions(${projectName} PRIVATE HEXLIB_GPU_ARCH=${HEXLIB_GPU_ARCH})

        ###

        if (NOT EXISTS "${cudaRoot}/include/cuda.h")
            message(FATAL_ERROR "Problem with cudaRoot path.")
        endif()

        target_include_directories(${projectName} PRIVATE "${cudaRoot}/include")

        ###

        if (EXISTS "${cudaLib}/cuda.lib")
            target_link_libraries(${projectName} PRIVATE "${cudaLib}/cuda.lib")
        elseif (EXISTS "${cudaLib}/libcuda.so")
            target_link_libraries(${projectName} PRIVATE "${cudaLib}/libcuda.so")
        elseif (EXISTS "${cudaLib}/stubs/libcuda.so")
            target_link_libraries(${projectName} PRIVATE "${cudaLib}/stubs/libcuda.so")
        else()
            message(FATAL_ERROR "Cannot find CUDA library. Problem with cudaLib path.")
        endif()

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
# addHeadersRecursive
#
#================================================================

function (addHeadersRecursive result dirs)

    set (internalHeaders "")

    foreach (dir ${dirs})
        file (GLOB_RECURSE tmp ${dir}/*.h ${dir}/*.hpp)
        list (APPEND internalHeaders ${tmp})
    endforeach()

    set_source_files_properties(${internalHeaders} PROPERTIES LANGUAGE CXX)

    set(resultValue ${${result}} ${internalHeaders})
    set(${result} ${resultValue} PARENT_SCOPE)

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

    set_source_files_properties(${internalSources} PROPERTIES LANGUAGE CXX)

    ###

    set(resultValue ${${result}} ${internalSources})
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

    if (libType STREQUAL "EXECUTABLE")
        add_executable(${projectName})
    else()
        add_library(${projectName} ${libType})
    endif()

    #----------------------------------------------------------------
    #
    # Find sources and headers.
    #
    #----------------------------------------------------------------

    set(sources "")
    set(headers "")

    foreach (dir IN LISTS sourceDirs)
        addSourcesRecursive(sources ${dir})
        addHeadersRecursive(headers ${dir})
    endforeach()

    #----------------------------------------------------------------
    #
    # RVISION build support:
    # target headers and install config.
    #
    #----------------------------------------------------------------

    if (headers)
        target_sources(${projectName} PUBLIC
            FILE_SET headers TYPE HEADERS BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/ FILES "${headers}"
        )
    endif()

    if (DEFINED RVISION_PLATFORM_BUILD AND DEFINED namespace)
        install(TARGETS ${projectName} EXPORT "${namespace}-targets"
            COMPONENT ${projectName}
            FILE_SET headers DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hexbase
            INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        )
    endif()

    #----------------------------------------------------------------
    #
    # Target sources.
    #
    #----------------------------------------------------------------

    target_sources(${projectName} PRIVATE ${sources})

    #----------------------------------------------------------------
    #
    # Target include dirs.
    #
    #----------------------------------------------------------------

    set(visibility PUBLIC)

    if (libType STREQUAL "INTERFACE")
        set(visibility INTERFACE)
    endif()

    target_include_directories(${projectName} ${visibility} $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

    if (libType STREQUAL "INTERFACE")
        return()
    endif()

    #----------------------------------------------------------------
    #
    # Namespace (for old platform).
    #
    #----------------------------------------------------------------

    if (folderName)
        set_target_properties(${projectName} PROPERTIES FOLDER ${folderName})
    endif()

    #----------------------------------------------------------------
    #
    # Link dependencies.
    #
    #----------------------------------------------------------------

    if (dependentProjects)
        target_link_libraries(${projectName} PUBLIC ${dependentProjects})
    endif()

    #----------------------------------------------------------------
    #
    # Compiler options and defines.
    #
    #----------------------------------------------------------------

    target_compile_definitions(${projectName} PRIVATE HEXLIB_PLATFORM=${HEXLIB_PLATFORM})
    target_compile_definitions(${projectName} PRIVATE HEXLIB_ERROR_MODE=${HEXLIB_ERROR_MODE})
    target_compile_definitions(${projectName} PRIVATE HEXLIB_GUARDED_MEMORY=${HEXLIB_GUARDED_MEMORY})
    target_compile_definitions(${projectName} PRIVATE HEXLIB_GPU_BITNESS=${HEXLIB_GPU_BITNESS})

    if (DEFINED HEXLIB_GPU_DISASM)
        target_compile_definitions(${projectName} PRIVATE HEXLIB_GPU_DISASM=${HEXLIB_GPU_DISASM})
    endif()

    ###

    if (MSVC)
        target_compile_options(${projectName} PRIVATE "/wd5040")
        target_compile_options(${projectName} PRIVATE "/we4239")
        target_compile_definitions(${projectName} PRIVATE _CRT_SECURE_NO_WARNINGS=1)
        target_compile_definitions(${projectName} PRIVATE _SCL_SECURE_NO_WARNINGS=1)
    endif()

    ###

    target_compile_features(${projectName} PRIVATE cxx_std_17)

    #----------------------------------------------------------------
    #
    # GPU support.
    #
    #----------------------------------------------------------------

    if (${requiresGpuCompiler})

        if (HEXLIB_PLATFORM EQUAL 0)

            target_compile_definitions(${projectName} PRIVATE HOSTCODE=1 DEVCODE=1)

        elseif(HEXLIB_PLATFORM EQUAL 1)

            add_dependencies(${projectName} makeGpuCompiler)
            set(CMAKE_CXX_COMPILER "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/gpuCompilerBuild/gpuCC" PARENT_SCOPE)

            setupCuda(${projectName})

        else()

            message(FATAL_ERROR, "Bad HEXLIB_PLATFORM value.")

        endif()

    endif()

endfunction()

#================================================================
#
# linkOpenCv
#
#================================================================

function (linkOpenCv targetName opencvVersion opencvModules)

    set(libs "")

    ###

    math(EXPR bitness "8*${CMAKE_SIZEOF_VOID_P}")

    ###

    if (CMAKE_BUILD_TYPE MATCHES Debug)
        set(libd "d")
    elseif (CMAKE_BUILD_TYPE MATCHES Release)
        set(libd "")
    elseif (CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
        set(libd "")
    else()
        message(FATAL_ERROR "Unrecognized CMAKE_BUILD_TYPE.")
    endif()

    ###

    if (WIN32)

        if (NOT DEFINED ENV{HEXFLOW_OPENCV_DIR})
           message(WARNING "Could not find HEXFLOW_OPENCV_DIR environment variable")
        endif()

        set(opencvDir $ENV{HEXFLOW_OPENCV_DIR})

        target_include_directories(${targetName} PRIVATE ${opencvDir}/include)

        set(opencvLib ${opencvDir}/x${bitness}/vc14/lib)

        foreach(lib ${opencvModules})
            set(libs ${libs} "${opencvLib}/${lib}${opencvVersion}${libd}.lib")
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /DELAYLOAD:${lib}${opencvVersion}${libd}.dll")
        endforeach()

        add_custom_command(TARGET ${targetName} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${opencvDir}/x${bitness}/vc14/bin"
                $<TARGET_FILE_DIR:${targetName}>)

        set(libs ${libs} delayimp.lib) # delay-load OpenCV to avoid detecting its memory leaks

    else()

        foreach(lib ${opencvModules})
            set(libs ${libs} "${lib}${libd}")
        endforeach()

    endif()

    ###

    target_link_libraries(${targetName} PRIVATE ${libs})

endfunction()

#================================================================
#
# main
#
#================================================================

defineMakeGpuCompiler()
