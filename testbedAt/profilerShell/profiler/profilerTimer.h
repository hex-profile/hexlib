#pragma once

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#endif

#include "profilerShell/profiler/profilerStruct.h"
#include "numbers/interface/numberInterface.h"
#include "numbers/int/intType.h"
#include "numbers/float/floatType.h"

//================================================================
//
// ProfilerTimer
//
//================================================================

class ProfilerTimer
{

public:

    sysinline ProfilerMoment moment();

    sysinline float32 ticksPerSec()
        {return freq;}

    sysinline float32 divTicksPerSec()
        {return divFreq;}

public:

    sysinline ProfilerTimer();

public:

    float32 freq;
    float32 divFreq;

};

//================================================================
//
// ProfilerTimer::ProfilerTimer
//
//================================================================

sysinline ProfilerTimer::ProfilerTimer()
{
    freq = 0;
    divFreq = 0;

    LARGE_INTEGER f = {0};
    if (!QueryPerformanceFrequency(&f)) return;

    freq = float32(f.QuadPart);
    divFreq = 1 / freq;
}

//================================================================
//
// ProfilerTimer::moment
//
// Speed optimization: sysinline and no error checks.
//
//================================================================

#ifdef _WIN32

sysinline ProfilerMoment ProfilerTimer::moment()
{
    LARGE_INTEGER moment = {0};
    QueryPerformanceCounter(&moment);
    return moment.QuadPart;
}

#endif

//================================================================
//
//
//
//================================================================



//================================================================
//
// ProfilerTimerEx
//
// Adds freezing support
//
//================================================================

class ProfilerTimerEx
{

public:

    sysinline ProfilerMoment moment()
    {
        ProfilerMoment rawMoment = timer.moment();
        return timeBase + rawMoment;
    }

    sysinline float32 divTicksPerSec()
        {return timer.divTicksPerSec();}

public:

    ProfilerTimerEx() {reset();}

    void reset()
    {
        timeBase = 0;
    }

private:

    COMPILE_ASSERT(!TYPE_IS_SIGNED(ProfilerMoment));
    ProfilerTimer timer;
    ProfilerMoment timeBase;
    float32 maxTimeInt32;

};
