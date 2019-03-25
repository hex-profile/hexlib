#define _CRT_SECURE_NO_WARNINGS

#include "formatStreamStdio.h"

#include <string.h>
#include <stdio.h>

#include "numbers/int/intType.h"
#include "numbers/float/floatType.h"

//================================================================
//
// SNPRINTF
//
//================================================================

#if defined(_MSC_VER)

    #define SNPRINTF CHARTYPE_SELECT(_snprintf, _snwprintf)

#elif defined(__GNUC__)

    #define SNPRINTF CHARTYPE_SELECT(snprintf, snwprintf)

#else

    #error Implementation required

#endif

//================================================================
//
// FrmStreamStdio::write
//
//================================================================

void FrmStreamStdio::write(const CharType* bufferPtr, size_t bufferSize)
{
    if_not (theOk) return;

    size_t availSize = theBufferCapacity - theBufferSize;
    size_t actualSize = clampRange(bufferSize, size_t{0}, availSize);

    memcpy(theBufferArray + theBufferSize, bufferPtr, actualSize * sizeof(bufferPtr[0]));
    theBufferSize += actualSize;
}

//================================================================
//
// FrmStreamStdio::printIntFloat
//
//================================================================

template <typename Type>
inline void FrmStreamStdio::printIntFloat(Type value, const FormatNumberOptions& options)
{
    CharType formatBuf[16];
    CharType* formatPtr = formatBuf;

    *formatPtr++ = '%';

    if (options.alignIsLeft())
        *formatPtr++ = '-';

    if (options.plusIsOn())
        *formatPtr++ = '+';

    if (options.fillIsZero())
        *formatPtr++ = '0';

    *formatPtr++ = '*'; // width

    CharType letter = 'd';

    if (TYPE_IS_BUILTIN_FLOAT(Type))
    {
        if (options.fformIsF())
            letter = 'f';

        if (options.fformIsE())
            letter = 'e';

        if (options.fformIsG())
            letter = 'g';
    }
    else
    {
        if (options.baseIsHex())
            letter = 'X';
        else
            letter = TYPE_IS_SIGNED(Type) ? 'd' : 'u';
    }

    if (TYPE_IS_BUILTIN_FLOAT(Type))
    {
        *formatPtr++ = '.';
        *formatPtr++ = '*'; // precision
    }

    *formatPtr++ = letter;
    *formatPtr++ = 0;

    ////

    size_t availSize = theBufferCapacity - theBufferSize;

    int result = 0;

    if (TYPE_IS_BUILTIN_FLOAT(Type))
    {
        if_not (def(value))
            write(CT("NAN"), 3);
        else
        {
            result = SNPRINTF(theBufferArray + theBufferSize, availSize,
                formatBuf, int(options.getWidth()), int(options.getPrecision()), value);
        }
    }
    else
    {
        result = SNPRINTF(theBufferArray + theBufferSize, availSize,
            formatBuf, int(options.getWidth()), value);
    }

    ////

    if_not (result >= 0 && size_t(result) <= size_t(availSize))
        theOk = false;
    else
        theBufferSize += result;
}

//================================================================
//
// FrmStreamStdio::write<BuiltinInt>
//
//================================================================

#define TMP_MACRO(Type, o) \
    \
    void FrmStreamStdio::write(Type value) \
        {printIntFloat(value, FormatNumberOptions());} \
    \
    void FrmStreamStdio::write(const FormatNumber<Type>& value) \
        {printIntFloat(value.value, value.options);} \

BUILTIN_INT_FOREACH(TMP_MACRO, o)
BUILTIN_FLOAT_FOREACH(TMP_MACRO, o)

#undef TMP_MACRO
