#include "messageFormatterStdio.h"

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
// MessageFormatterStdio::write
//
//================================================================

void MessageFormatterStdio::write(const CharType* bufferPtr, size_t bufferSize)
{
    if_not (ok)
        return;

    size_t availSize = memorySize - usedSize;

    if_not (bufferSize <= availSize)
        {ok = false; return;}

    auto dstPtr = memoryArray + usedSize;

    memcpy(dstPtr, bufferPtr, bufferSize * sizeof(bufferPtr[0]));
    dstPtr[bufferSize] = 0; // use reserved space

    usedSize += bufferSize;
}

//================================================================
//
// MessageFormatterStdio::printIntFloat
//
//================================================================

template <typename Type>
inline void MessageFormatterStdio::printIntFloat(Type value, const FormatNumberOptions& options)
{
    if_not (ok)
        return;

    ////

    CharType formatBuf[16];
    CharType* formatPtr = formatBuf;

    *formatPtr++ = '%';

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

    size_t availSize = memorySize - usedSize;

    int result = 0;

    if (TYPE_IS_BUILTIN_FLOAT(Type))
    {
        if_not (def(value))
            write(CT("NAN"), 3);
        else
        {
            result = SNPRINTF(memoryArray + usedSize, availSize + 1,
                formatBuf, int(options.getWidth()), int(options.getPrecision()), value);
        }
    }
    else if (TYPE_EQUAL(Type, bool))
    {
        result = SNPRINTF(memoryArray + usedSize, availSize + 1,
            value ? CT("ON") : CT("OFF"));
    }
    else
    {
        result = SNPRINTF(memoryArray + usedSize, availSize + 1,
            formatBuf, int(options.getWidth()), value);
    }

    ////

    if_not (result >= 0 && size_t(result) <= availSize)
        {ok = false; return;}

    ////

    usedSize += result;
}

//================================================================
//
// MessageFormatterStdio::write<BuiltinInt>
//
//================================================================

#define TMP_MACRO(Type, o) \
    \
    void MessageFormatterStdio::write(Type value) \
        {printIntFloat(value, FormatNumberOptions());} \
    \
    void MessageFormatterStdio::write(const FormatNumber<Type>& value) \
        {printIntFloat(value.value, value.options);} \

BUILTIN_INT_FOREACH(TMP_MACRO, o)
BUILTIN_FLOAT_FOREACH(TMP_MACRO, o)

#undef TMP_MACRO
