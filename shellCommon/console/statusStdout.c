#include "statusStdout.h"

#include <stdio.h>

//================================================================
//
// VSNPRINTF
//
//================================================================

#if defined(_MSC_VER)

    #define VSNPRINTF CHARTYPE_SELECT(_snprintf, _snwprintf)

#elif defined(__GNUC__)

    #define VSNPRINTF CHARTYPE_SELECT(vsnprintf, vsnwprintf)

#else

    #error Implementation required

#endif

//================================================================
//
// StatusStdout::clear
//
//================================================================

bool StatusStdout::clear()
{
    const int clearSize = maxPrefixSize + maxMsgSize;
    CharType buffer[clearSize+1];

    ensure(fputs(CT("\r"), stdout) >= 0);

    for_count (i, clearSize)
        buffer[i] = ' ';

    buffer[clearSize] = 0;

    ensure(fputs(buffer, stdout) >= 0);
    ensure(fputs(CT("\r"), stdout) >= 0);

    return true;
}

//================================================================
//
// StatusStdout::printf
//
//================================================================

bool StatusStdout::vprintf(const TimerKit& kit, const CharType* format, va_list args)
{
    TimeMoment currentMoment = kit.timer.moment();
    bool skipOutput = lastOutputInit && (kit.timer.diff(lastOutput, currentMoment) < 0.5f);

    if (skipOutput)
        return true;

    lastOutput = currentMoment;
    lastOutputInit = true;

    ////

    ensure(clear());

    ////

    CharType buffer[maxMsgSize+1];
    buffer[0] = 0;
    VSNPRINTF(buffer, maxMsgSize, format, args);
    buffer[maxMsgSize] = 0;

    ////

    {
        phase = (phase+1) % 4;
        const CharType* phaseChars = CT("|/-\\");

        ensure(fprintf(stdout, CT("%c %.0f%%%s%s "), phaseChars[phase], progressValue * 100.f, *buffer ? CT(": ") : CT(""), buffer) >= 0);
    }

    ////

    return true;
}
