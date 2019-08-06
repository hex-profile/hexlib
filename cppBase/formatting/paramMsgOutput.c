#include "paramMsgOutput.h"

#include "numbers/int/intType.h"

//================================================================
//
// paramMsgSpecChar
//
//================================================================

static const CharType paramMsgSpecChar = '%';

//================================================================
//
// formatOutput<ParamMsg>
//
//================================================================

void ParamMsg::outputFunc(const void* value, FormatOutputStream& outputStream)
{

    const ParamMsg& v = * (const ParamMsg*) value;

    ////

    const FormatOutputAtom* paramPtr = v.paramPtr;
    size_t paramSize = v.paramSize;

    const CharType* formatPtr = v.format.ptr;
    const CharType* formatEnd = v.format.ptr + v.format.size;

    ////

    size_t oneDigitParamSize = clampMax<size_t>(paramSize, 10);

    size_t currentIndex = 0;

    ////

    for (;;)
    {

        const CharType* searchStart = formatPtr;

        while (formatPtr != formatEnd && *formatPtr != paramMsgSpecChar)
            ++formatPtr;

        if (formatPtr != searchStart)
            outputStream.write(searchStart, formatPtr - searchStart);

        if (formatPtr == formatEnd)
            break;

        ++formatPtr;

        //
        // here: received %, what's next?
        //

        if (formatPtr == formatEnd)
        {
            outputStream.write(&paramMsgSpecChar, 1);
            break;
        }

        //
        // received %%, output %
        //

        if (*formatPtr == paramMsgSpecChar)
        {
            outputStream.write(&paramMsgSpecChar, 1);
            ++formatPtr;
            continue;
        }

        //
        // received %n
        //

        {
            int32 n = *formatPtr - '0';

            if (uint32(n) < oneDigitParamSize)
            {
                const FormatOutputAtom& p = paramPtr[n];
                p.func(p.value, outputStream);
                ++formatPtr;
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
            outputStream.write(&paramMsgSpecChar, 1);
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

