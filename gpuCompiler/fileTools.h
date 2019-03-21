#pragma once

#ifdef _WIN32
#include <windows.h>
#include "stlString/stlString.h"
#endif

//================================================================
//
// fileExist
//
//================================================================

inline bool fileExist(const StlString& filename)
{

#if defined(_WIN32)

    //
    // only FindFirstFile - GetFileAttributes sees not all files
    //

    WIN32_FIND_DATA tmp;
    HANDLE handle = FindFirstFile(filename.c_str(), &tmp);
    bool yes = (handle != INVALID_HANDLE_VALUE);
    if (handle != INVALID_HANDLE_VALUE) FindClose(handle);
    return yes;

#else

    #error Implement

#endif

}
