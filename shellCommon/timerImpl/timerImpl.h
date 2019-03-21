#pragma once

//================================================================
//
// TimerImpl
//
//================================================================

#if defined(_WIN32) || defined(__linux__)

    #include "timerStdChrono.h"
    using TimerImpl = TimerStdChrono;

#else

    #error Attention required

#endif
