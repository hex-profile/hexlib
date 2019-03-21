cmake_minimum_required(VERSION 3.13.2 FATAL_ERROR)

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

function (hexlibProjectTemplate projectName projectAlias dependentProjects)

    addSourcesRecursive(sources .)

    add_library(${projectName} OBJECT ${sources})
    add_library(${projectAlias} ALIAS ${projectName})
    set_target_properties(${projectName} PROPERTIES FOLDER hexlib)

    target_include_directories(${projectName} PUBLIC .)

    if (dependentProjects)
        target_link_libraries(${projectName} PUBLIC ${dependentProjects})
    endif()

    # target_compile_definitions(${projectName} PRIVATE HEX_PLATFORM=0 HOSTCODE=1 DEVCODE=1)

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

add_subdirectory(imgBaseGpu)
add_subdirectory(moduleToolsGpu)
add_subdirectory(imgLibGpu)
add_subdirectory(shellCommonGpu)

add_subdirectory(gpuCompiler)
