#pragma once

//================================================================
//
// MsgBoxImpl
//
//================================================================

#if defined(_WIN32)

    #include "msgBoxWin32.h"
    using MsgBoxImpl = MsgBoxWin32;

#elif defined(__linux__)

    #include "msgBoxLinux.h"
    using MsgBoxImpl = MsgBoxLinux;

#else

    #error

#endif
