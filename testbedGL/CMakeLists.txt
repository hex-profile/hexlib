#================================================================
#
# OpenGL testbed lib.
#
#================================================================

hexlibProjectTemplate(testbedGL STATIC . "imgBase;shellCommon;shellCommonGpu" TRUE "")

#================================================================
#
# Pthreads.
#
#================================================================

if (UNIX)
    find_package(Threads REQUIRED)
    target_link_libraries(testbedGL PRIVATE ${CMAKE_THREAD_LIBS_INIT})
endif()

#================================================================
#
# OpenGL.
#
#================================================================

find_package(OpenGL REQUIRED)
target_include_directories(testbedGL PRIVATE ${OPENGL_INCLUDE_DIRS})
target_link_libraries(testbedGL PRIVATE ${OPENGL_LIBRARIES})

#================================================================
#
# GLEW.
#
#================================================================

if (DEFINED ENV{GLEW_ROOT})
    target_include_directories(testbedGL PRIVATE $ENV{GLEW_ROOT})
    target_sources(testbedGL PRIVATE $ENV{GLEW_ROOT}/GL/glew.c)
else()
    find_package(GLEW REQUIRED)
    target_include_directories(testbedGL PRIVATE ${GLEW_INCLUDE_DIRS})
    target_link_libraries(testbedGL PRIVATE ${GLEW_LIBRARIES})
endif()

#================================================================
#
# GLFW.
#
#================================================================

if (DEFINED ENV{GLFW_ROOT})
    math(EXPR BITNESS "8*${CMAKE_SIZEOF_VOID_P}")
    target_include_directories(testbedGL PRIVATE $ENV{GLFW_ROOT}/include)
    target_link_libraries(testbedGL PRIVATE $ENV{GLFW_ROOT}/lib/${CMAKE_BUILD_TYPE}${BITNESS}/glfw3.lib)
else()
    find_package(glfw3 REQUIRED)
    target_include_directories(testbedGL PRIVATE ${GLFW_INCLUDE_DIRS})
    target_link_libraries(testbedGL PRIVATE glfw)
endif()
