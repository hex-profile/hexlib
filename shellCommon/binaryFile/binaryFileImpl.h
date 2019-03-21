#pragma once

//================================================================
//
// BinaryFileImpl
//
//================================================================

#if defined(_WIN32)

    #include "binaryFileWin32.h"
    using BinaryFileImpl = BinaryFileWin32;

#else

    #error

#endif
