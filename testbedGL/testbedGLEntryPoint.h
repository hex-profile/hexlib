#pragma once

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
    #include <winuser.h>
#endif

#include <stdlib.h>

#include "testbedGL/testbedGL/testbedGL.h"

//================================================================
//
// TESTBED_GL_ENTRY_POINT (Linux)
//
//================================================================

#if defined(__linux__)

    #define TESTBED_GL_ENTRY_POINT(factory) \
        \
        int main(int argCount, const char* argStr[]) \
        { \
            return testbedGL::mainEntry(argCount, argStr, factory) ? EXIT_SUCCESS : EXIT_FAILURE; \
        } \

#endif

//================================================================
//
// TESTBED_GL_ENTRY_POINT (Win32)
//
//================================================================

#if defined(_WIN32)

    #define TESTBED_GL_ENTRY_POINT(factory) \
        \
        int WINAPI WinMain \
        ( \
            HINSTANCE hInstance, \
            HINSTANCE hPrevInstance, \
            LPSTR lpCmdLine, \
            int nCmdShow \
        ) \
        { \
            CharType** argv = __argv; \
            \
            return testbedGL::mainEntry(__argc, (const CharType**) argv, factory) ? EXIT_SUCCESS : EXIT_FAILURE; \
        } \

#endif
