#pragma once

#if defined(_WIN32)

    #include "emuMultiWin32.h"

    namespace emuMultiProc
    {
        using namespace emuMultiWin32;
        using EmuMultiProc = EmuMultiWin32;
    }

#elif defined(__arm__) || defined (__aarch64__) || defined (__linux__)

    #include "emuMultiProcFake.h"

    namespace emuMultiProc
    {
        using namespace emuMultiProcFake;
        using EmuMultiProc = EmuMultiProcFake;
    }

#else

    #error

#endif
