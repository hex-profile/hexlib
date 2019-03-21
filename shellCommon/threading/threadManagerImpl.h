#pragma once

//================================================================
//
// ThreadManagerImpl
//
//================================================================

#if defined(_WIN32)

    #include "threadManagerWin32.h"

    using ThreadManagerImpl = ThreadManagerWin32;

#endif


#if defined(__linux__)

    #include "threadManagerLinux.h"

    using ThreadManagerImpl = ThreadManagerLinux;

#endif
