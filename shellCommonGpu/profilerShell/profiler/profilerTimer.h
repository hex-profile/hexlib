#pragma once

//----------------------------------------------------------------

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#elif defined(__linux__)
    #include <time.h>
#else
    #error
#endif

//----------------------------------------------------------------

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

#if defined(_WIN32)

    LARGE_INTEGER f = {0};
    if (!QueryPerformanceFrequency(&f)) return;
    freq = float32(f.QuadPart);

#elif defined(__linux__)

    freq = 1e9f;

#else

    #error

#endif


    divFreq = 1 / freq;
}

//================================================================
//
// ProfilerTimer::moment
//
// Speed optimization: sysinline and no error checks.
//
//================================================================

sysinline ProfilerMoment ProfilerTimer::moment()
{

#if defined(_WIN32)

    LARGE_INTEGER moment = {0};
    QueryPerformanceCounter(&moment);
    return moment.QuadPart;

#elif defined(__linux__)

    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return ProfilerMoment(time.tv_sec) * ProfilerMoment(1000000000) + time.tv_nsec;

#else

    #error

#endif

}

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
