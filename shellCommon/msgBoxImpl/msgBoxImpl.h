#pragma once

//================================================================
//
// TimerImpl
//
//================================================================

#if defined(_WIN32)

    #include "msgBoxWin32.h"
    using MsgBoxImpl = MsgBoxWin32;

#else

    #error

#endif
