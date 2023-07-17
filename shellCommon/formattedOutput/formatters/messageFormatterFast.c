#include "messageFormatterFast.h"

#include "numbers/int/intType.h"
#include "numbers/float/floatType.h"
#include "parseTools/writeTools.h"
#include "errorLog/debugBreak.h"

//================================================================
//
// MessageFormatterFast::writeFunc
//
//================================================================

sysinline void MessageFormatterFast::writeFunc(const CharType* bufferPtr, size_t bufferSize)
{
    if_not (ok)
        return;

    auto availSize = memorySize - usedSize;

    if_not (bufferSize <= availSize)
    {
        ok = false;
        return;
    }

    ////

    auto dstPtr = memoryArray + usedSize;

    for_count (i, bufferSize)
        *dstPtr++ = *bufferPtr++;

    ////

    *dstPtr = 0; // use reserved space!

    ////

    usedSize += bufferSize;
}

//================================================================
//
// MessageFormatterFast::write
//
//================================================================

void MessageFormatterFast::write(const CharType* bufferPtr, size_t bufferSize)
{
    writeFunc(bufferPtr, bufferSize);
}

//================================================================
//
// MessageFormatterFast::printInt
//
//================================================================

template <typename Type>
inline void MessageFormatterFast::printInt(const Type& value, const FormatNumberOptions& options)
{
    if_not (ok)
        return;

    ////

    if (TYPE_EQUAL(Type, bool))
    {
        if (value)
            writeFunc(CT("ON"), 2);
        else
            writeFunc(CT("OFF"), 3);

        return;
    }

    ////

    using namespace writeToolsImpl;

    IntOptions opt;

    ////

    if (options.plusIsOn())
        opt.forceSign = true;

    if (options.fillIsZero())
        opt.fill = Fill::Zero;

    opt.minWidth = options.getWidth();

    ////

    if (options.baseIsHex())
    {
        opt.base = Base::Hex;
        opt.uppercase = true;
    }

    ////

    auto writer = [&] (auto* ptr, auto size)
        {writeFunc(ptr, size);};

    writeInt<CharType>(value, writer, opt);
}

//================================================================
//
// MessageFormatterFast::printFloat
//
//================================================================

template <typename Type>
inline void MessageFormatterFast::printFloat(const Type& value, const FormatNumberOptions& options)
{
    if_not (ok)
        return;

    ////

    using namespace writeToolsImpl;

    FloatOptions opt;

    if (options.plusIsOn())
        opt.forceBodySign = true;

    if (options.fillIsZero())
        opt.fill = Fill::Zero;

    opt.precision = options.getPrecision();
    opt.minWidth = options.getWidth();

    ////

    if (options.fformIsF())
        opt.format = FloatFormat::Fixed;

    if (options.fformIsE())
    {
        opt.format = FloatFormat::Exponential;
        opt.forceExpoSign = true;
        opt.forceExponent = true;
        opt.alignExponent = true;
    }

    if (options.fformIsG())
    {
        opt.format = FloatFormat::Flexible;
    }

    ////

    if_not (def(value))
        opt.uppercase = true;

    ////

    auto writer = [&] (auto* ptr, auto size)
        {writeFunc(ptr, size);};

    if_not (DEBUG_BREAK_CHECK(writeFloat<CharType>(value, writer, opt)))
        ok = false;
}

//================================================================
//
// MessageFormatterFast::write<*>
//
//================================================================

#define TMP_MACRO(Type, o) \
    \
    void MessageFormatterFast::write(Type value) \
        {printFloat(value, FormatNumberOptions{});} \
    \
    void MessageFormatterFast::write(const FormatNumber<Type>& value) \
        {printFloat(value.value, value.options);} \

BUILTIN_FLOAT_FOREACH(TMP_MACRO, o)

#undef TMP_MACRO

////

#define TMP_MACRO(Type, o) \
    \
    void MessageFormatterFast::write(Type value) \
        {printInt(value, FormatNumberOptions{});} \
    \
    void MessageFormatterFast::write(const FormatNumber<Type>& value) \
        {printInt(value.value, value.options);} \

BUILTIN_INT_FOREACH(TMP_MACRO, o)

#undef TMP_MACRO
