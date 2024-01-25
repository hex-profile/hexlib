#include "paramMsgOutput.h"

#include "numbers/int/intType.h"

//================================================================
//
// formatOutput<ParamMsg>
//
//================================================================

void ParamMsg::outputFunc(const void* value, FormatOutputStream& outputStream)
{

    const ParamMsg& v = * (const ParamMsg*) value;
    auto specialChar = v.specialChar;

    ////

    auto paramPtr = v.paramPtr;
    auto paramSize = v.paramSize;

    auto formatPtr = v.format.ptr;
    auto formatEnd = v.format.ptr + v.format.size;

    ////

    size_t oneDigitParamSize = clampMax<size_t>(paramSize, 10);

    size_t currentIndex = 0;

    ////

    for (; ;)
    {

        const CharType* searchStart = formatPtr;

        while (formatPtr != formatEnd && *formatPtr != specialChar)
            ++formatPtr;

        if (formatPtr != searchStart)
            outputStream.write(searchStart, formatPtr - searchStart);

        if (formatPtr == formatEnd)
            break;

        ++formatPtr;

        //
        // received %n
        //

        if (formatPtr != formatEnd)
        {
            int32 n = *formatPtr - '0';

            if (uint32(n) < oneDigitParamSize)
            {
                const FormatOutputAtom& p = paramPtr[n];
                p.func(p.value, outputStream);
                ++formatPtr;
                currentIndex = n + 1;
                continue;
            }
        }

        //
        // received %dontKnowWhat, consider it as "next parameter"
        //

        if (currentIndex < paramSize)
        {
            const FormatOutputAtom& p = paramPtr[currentIndex];
            p.func(p.value, outputStream);
            ++currentIndex;
        }
        else
        {
            outputStream.write(&specialChar, 1);
        }

    }
}

//================================================================
//
// formatOutput<ParamMsg>
//
//================================================================

template <>
void formatOutput(const ParamMsg& value, FormatOutputStream& outputStream)
{
    value.func(value.value, outputStream);
}

