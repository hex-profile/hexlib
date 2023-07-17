#include "timer/timer.h"
#include "charType/charType.h"

namespace timerUtils {

//================================================================
//
// StrOutput
//
// To decouple from specific string implementation.
//
//================================================================

struct StrOutput
{
    virtual void add(const CharType* bufArray, size_t bufSize) =0;
};

//================================================================
//
// StdStringOutput
//
// For STL-like strings.
//
//================================================================

template <typename Str>
class StdStringOutput : public StrOutput
{

public:

    StdStringOutput(Str& str)
        : str(str) {}

    void add(const CharType* bufArray, size_t bufSize)
        {str.append(bufArray, bufSize);}

private:

    Str& str;

};

//----------------------------------------------------------------

template <typename Str>
inline StdStringOutput<Str> stdStringOutput(Str& str)
    {return StdStringOutput<Str>(str);}

//================================================================
//
// formatLocalTimeMoment
//
//================================================================

bool formatLocalTimeMoment(TimeMicroseconds time, int fracDigits, StrOutput& output);

//================================================================
//
// formatDuration
//
//================================================================

bool formatDuration(TimeMicroseconds time, int fracPrecision, StrOutput& result);

//----------------------------------------------------------------

template <typename String>
inline String formatDuration(TimeMicroseconds time, int fracPrecision)
{
    String result;
    StdStringOutput<String> output(result);
    formatDuration(time, fracPrecision, output);
    return result;
}

//----------------------------------------------------------------

}
