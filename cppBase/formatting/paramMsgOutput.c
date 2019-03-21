#include "paramMsgOutput.h"

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

    for (;;)
    {

        const CharType* searchStart = formatPtr;

        while (formatPtr != formatEnd && *formatPtr != paramMsgSpecChar)
            ++formatPtr;

        if (formatPtr != searchStart)
            outputStream.write(searchStart, formatPtr - searchStart);

        if (formatPtr == formatEnd) break;

        ++formatPtr;

        //
        // here: received %, what's next?
        //

        //
        // received %%, output %
        //

        if (formatPtr != formatEnd && *formatPtr == paramMsgSpecChar)
        {
            outputStream.write(&paramMsgSpecChar, 1);
            ++formatPtr;
            continue;
        }

        //
        // received %n
        //

        if (formatPtr != formatEnd)
        {
            int32 n = *formatPtr - '0';

            if (n >= 0 && n <= 9)
            {
                if (uint32(n) < paramSize)
                {
                    const FormatOutputAtom& p = paramPtr[n];
                    p.func(p.value, outputStream);
                    ++formatPtr;
                    continue;
                }
            }
        }

        //
        // received %dontKnowWhat
        //

        outputStream.write(&paramMsgSpecChar, 1);

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

