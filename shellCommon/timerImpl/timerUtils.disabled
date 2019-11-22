#include "timerUtils.h"

#include <time.h>
#include <string.h>

namespace timerUtils {

using namespace std;

//================================================================
//
// TIMER_ENSURE
//
//================================================================

#define TIMER_ENSURE(cond) \
    if (cond) ; else return false

//================================================================
//
// power10
//
//================================================================

template <typename Int>
inline Int power10(int digits)
{
    Int result = 1;

    for (int i = 0; i < digits; ++i)
        result *= 10;

    return result;
}

//================================================================
//
// formatUint
//
//================================================================

template <typename Uint>
bool formatUint(Uint value, int digits, StrOutput& result)
{
    static_assert(is_unsigned<Uint>::value, "");

    ////

    constexpr int maxDigits = 64;
    TIMER_ENSURE(digits >= 0 && digits <= maxDigits);

    ////

    char tmpBuffer[maxDigits];
    char* end = tmpBuffer + digits - 1;

    for (int i = 0; i < digits; ++i)
    {
        Uint valueDiv10 = value / 10;
        Uint digit = value - valueDiv10 * 10;
        *end-- = '0' + int(digit);
        value = valueDiv10;
    }

    ////

    result.add(tmpBuffer, digits);

    ////

    return true;
}

//================================================================
//
// originalDigits
//
//================================================================

constexpr int originalDigits = 6;

//================================================================
//
// formatLocalTimeMoment
//
//================================================================

bool formatLocalTimeMoment(TimeMicroseconds time, int fracDigits, StrOutput& result)
{
    static_assert(is_unsigned<TimeMicroseconds>::value, "");

    //----------------------------------------------------------------
    //
    // Round to smallest units.
    //
    // This should be done first, because rounding influences
    // integer seconds value.
    //
    //----------------------------------------------------------------

    TIMER_ENSURE(fracDigits >= 0 && fracDigits <= originalDigits);

    int ignoreDigits = originalDigits - fracDigits;

    if (ignoreDigits)
    {
        TimeMicroseconds ignoreFactor = power10<TimeMicroseconds>(ignoreDigits);
        time = (time + (ignoreFactor >> 1)) / ignoreFactor; // round to the nearest unit
    }

    //----------------------------------------------------------------
    //
    // Split into the integer seconds part and fractional seconds part.
    //
    //----------------------------------------------------------------

    TimeMicroseconds fracFactor = power10<TimeMicroseconds>(fracDigits);

    ////

    TimeMicroseconds timeInt = time / fracFactor;
    TimeMicroseconds timeFrac = time - timeInt * fracFactor;

    time_t clockTime = time_t(timeInt);
    TIMER_ENSURE(TimeMicroseconds(clockTime) == timeInt);

    //----------------------------------------------------------------
    //
    // Convert to tm struct
    //
    //----------------------------------------------------------------

    struct tm tmp;

#if defined(_MSC_VER)
    localtime_s(&tmp, &clockTime);
#elif defined(__MINGW32__)
    tmp = *localtime(&clockTime);
#elif defined(__GNUC__)
    localtime_r(&clockTime, &tmp);
#else
    #error Attention required
#endif

    //----------------------------------------------------------------
    //
    // Format int seconds part
    //
    //----------------------------------------------------------------

    char buf[128];
    TIMER_ENSURE(strftime(buf, 128, "%F %H:%M:%S", &tmp) > 0);

    ////

    result.add(buf, strlen(buf));

    //----------------------------------------------------------------
    //
    // Format fractional seconds part
    //
    //----------------------------------------------------------------

    if (fracDigits)
    {
        result.add(".", 1);
        TIMER_ENSURE(formatUint(timeFrac, fracDigits, result));
    }

    ////

    return true;
}

//================================================================
//
// formatDuration
//
//================================================================

bool formatDuration(TimeMicroseconds time, int fracPrecision, StrOutput& result)
{
    TIMER_ENSURE(fracPrecision >= 0 && fracPrecision <= 3);
    int ignoreDigits = originalDigits - fracPrecision;
    TimeMicroseconds ignoreFactor = power10<TimeMicroseconds>(ignoreDigits);

    TimeMicroseconds t = (time + ignoreFactor/2) / ignoreFactor; // round to the smallest unit

    ////

    TimeMicroseconds fracFactor = power10<TimeMicroseconds>(fracPrecision);
    TimeMicroseconds timeFrac = t % fracFactor;
    t /= fracFactor;

    TimeMicroseconds timeS = t % 60;
    t /= 60;

    TimeMicroseconds timeM = t % 60;
    t /= 60;

    TimeMicroseconds timeH = t % 24;
    t /= 24;

    TimeMicroseconds timeD = t;

    if (timeD)
    {
        TIMER_ENSURE(formatUint(timeD, 2, result));
        result.add(":", 1);
    }

    if (timeD || timeH)
    {
        TIMER_ENSURE(formatUint(timeH, 2, result));
        result.add(":", 1);
    }

    if (timeD || timeH || timeM)
    {
        TIMER_ENSURE(formatUint(timeM, 2, result));
        result.add(":", 1);
    }

    formatUint(timeS, 2, result);

    if (fracPrecision)
    {
        result.add(".", 1);
        TIMER_ENSURE(formatUint(timeFrac, fracPrecision, result));
    }

    return true;
}

//----------------------------------------------------------------

}
