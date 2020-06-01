#pragma once

//================================================================
//
// BinaryFileImpl
//
//================================================================

#if defined(_WIN32)

    #include "binaryFileWin32.h"
    using BinaryFileImpl = BinaryFileWin32;

#elif defined(__linux__)

    #include "binaryFileLinux.h"
    using BinaryFileImpl = BinaryFileLinux;

#else

    #error

#endif
