#pragma once

//================================================================
//
// FileToolsImpl
//
//================================================================

#if defined(_WIN32)

    #include "fileToolsWin32.h"

    using FileToolsImpl = FileToolsWin32;

#elif defined(__linux__)

    #include "fileToolsLinux.h"

    using FileToolsImpl = FileToolsLinux;

#else

    #error

#endif
