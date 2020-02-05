#pragma once

#include <stdarg.h>

#include "timer/timerKit.h"
#include "charType/charType.h"
#include "numbers/float/floatBase.h"
#include "timer/timer.h"

//================================================================
//
// StatusStdout
//
//================================================================

class StatusStdout
{

public:

    inline bool printf(const TimerKit& kit, const CharType* format, ...)
    {
        va_list args;
        va_start(args, format);
        bool ok = this->vprintf(kit, format, args);
        va_end(args);
        return ok;
    }

    bool vprintf(const TimerKit& kit, const CharType* format, va_list args);
    bool clear();

public:

    void resetUpdateTime() {lastOutputInit = false;}

public:

    float32 progressValue = 0;

private:

    static const int maxPrefixSize = 15;
    static const int maxMsgSize = 60;

    int phase = 0;

    ////

    bool lastOutputInit = false;
    TimeMoment lastOutput;

};
