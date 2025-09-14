#pragma once

#include "compileTools/compileTools.h"
#include "numbers/int/intType.h"
#include "numbers/divRound.h"
#include "errorLog/debugBreak.h"

namespace writeToolsImpl {

//================================================================
//
// writeZero
//
//================================================================

template <typename Char, typename Writer>
sysinline void writeZeros(int n, const Writer& writer)
{
    if_not (n > 0)
        return;

    ////

    static const Char packArray[] = {'0', '0', '0', '0', '0', '0', '0', '0'};
    constexpr int packSize = COMPILE_ARRAY_SIZE(packArray);

    ////

    while (n >= packSize)
    {
        writer(packArray, packSize);
        n -= packSize;
    }

    if (n > 0)
        writer(packArray, n);
}

//================================================================
//
// writeSpaces
//
//================================================================

template <typename Char, typename Writer>
sysinline void writeSpaces(int n, const Writer& writer)
{
    if_not (n > 0)
        return;

    ////

    static const Char packArray[] = {' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '};
    constexpr int packSize = COMPILE_ARRAY_SIZE(packArray);

    while (n >= packSize)
    {
        writer(packArray, packSize);
        n -= packSize;
    }

    if (n > 0)
        writer(packArray, n);
}

//================================================================
//
// Limits.
//
//================================================================

constexpr int maxIntBits = 64;

constexpr int maxDecimalIntCharsWithoutSign = 20; // ceil(maxIntBits * log(2) / log(10))

constexpr int maxDecimalIntCharsWithSign = maxDecimalIntCharsWithoutSign + 1;

//================================================================
//
// IntOptions
//
//================================================================

enum class Fill: char {Space, Zero};
enum class Base: char {Dec, Hex};

////

struct IntOptions
{
    bool forceSign = false;
    Fill fill = Fill::Space;
    Base base = Base::Dec;
    bool uppercase = false;
    int minWidth = 0;
};

//================================================================
//
// writeUnsignedInt
//
//================================================================

template <typename Char, typename Uint, typename Writer>
sysinline void writeUnsignedInt(Uint value, const Writer& writer, const IntOptions& options, Char signBuf = '+')
{
    COMPILE_ASSERT(TYPE_IS_BUILTIN_INT(Uint));
    COMPILE_ASSERT(!TYPE_IS_SIGNED(Uint));

    //----------------------------------------------------------------
    //
    // Body.
    //
    //----------------------------------------------------------------

    COMPILE_ASSERT(TYPE_BIT_COUNT(Uint) <= maxIntBits);

    Char bodyArray[maxDecimalIntCharsWithoutSign];

    ////

    auto* bodyEnd = bodyArray + COMPILE_ARRAY_SIZE(bodyArray);
    auto* bodyPtr = bodyEnd;

    if (options.base == Base::Dec)
    {
        do
        {
            auto valueDiv10 = value / 10;
            auto digit = value - valueDiv10 * 10;
            *--bodyPtr = '0' + char(digit);
            value = valueDiv10;
        }
        while (value != 0);
    }
    else
    {
        auto* table = options.uppercase ? "0123456789ABCDEF" : "0123456789abcdef";

        do
        {
            auto digit = value & 0xF;
            *--bodyPtr = table[digit];
            value >>= 4;
        }
        while (value != 0);
    }

    ////

    int bodySize = bodyEnd - bodyPtr;

    //----------------------------------------------------------------
    //
    // Sign and padding.
    //
    //----------------------------------------------------------------

    int signChars = options.forceSign;

    if (options.minWidth == 0)
    {
        writer(&signBuf, signChars);
    }
    else
    {
        int usedChars = signChars + bodySize;

        int paddingChars = clampMin(options.minWidth, usedChars) - usedChars;

        if (options.fill == Fill::Zero)
        {
            writer(&signBuf, signChars);
            writeZeros<Char>(paddingChars, writer);
        }
        else
        {
            writeSpaces<Char>(paddingChars, writer);
            writer(&signBuf, signChars);
        }
    }

    //----------------------------------------------------------------
    //
    // Body.
    //
    //----------------------------------------------------------------

    writer(bodyPtr, bodySize);
}

//================================================================
//
// writeSignedInt
//
//================================================================

template <typename Char, typename Writer, typename Int>
sysinline void writeSignedInt(Int value, const Writer& writer, const IntOptions& options)
{
    COMPILE_ASSERT(TYPE_IS_BUILTIN_INT(Int));
    COMPILE_ASSERT(TYPE_IS_SIGNED(Int));

    ////

    using Uint = TYPE_MAKE_UNSIGNED(Int);

    ////

    if (value >= 0)
    {
        writeUnsignedInt<Char>(Uint(value), writer, options, '+');
    }
    else
    {
        // Works also for INT_MIN in 2s complement code.
        auto opt = options;
        opt.forceSign = true;
        writeUnsignedInt<Char>(Uint(-value), writer, opt, '-');
    }
}

//================================================================
//
// writeInt
//
//================================================================

struct WriteUnsignedIntImpl
{
    template <typename Char, typename Writer, typename Int>
    static sysinline void func(Int value, const Writer& writer, const IntOptions& options)
        {writeUnsignedInt<Char>(value, writer, options);}
};

struct WriteSignedIntImpl
{
    template <typename Char, typename Writer, typename Int>
    static sysinline void func(Int value, const Writer& writer, const IntOptions& options)
        {writeSignedInt<Char>(value, writer, options);}
};

struct WriteBoolImpl
{
    template <typename Char, typename Writer, typename Int>
    static sysinline void func(Int value, const Writer& writer, const IntOptions& options)
        {writeUnsignedInt<Char>(unsigned{value}, writer, options);}
};

//----------------------------------------------------------------

template <typename Char, typename Writer, typename Int>
sysinline void writeInt(Int value, const Writer& writer, const IntOptions& options)
{
    COMPILE_ASSERT(TYPE_IS_BUILTIN_INT(Int));

    using WriteAnyInt = TypeSelect<TYPE_IS_SIGNED(Int), WriteSignedIntImpl, WriteUnsignedIntImpl>;
    using Impl = TypeSelect<TYPE_EQUAL(Int, bool), WriteBoolImpl, WriteAnyInt>;

    Impl::template func<Char>(value, writer, options);
}

//================================================================
//
// FloatTraits
//
//================================================================

template <int binaryDigits>
struct FloatTraits;

////

template <>
struct FloatTraits<24>
{
    // Using signed number only because float->signed conversion
    // is much more efficient, at least on x86 and MSVC.
    using BaseInt = int32;
    using BaseUint = uint32;
    static constexpr int bodyPrecision = 7; // actually 7.2, but...
    static constexpr BaseUint bodyFactor = 10000000;
};

////

template <>
struct FloatTraits<53>
{
    using BaseInt = int64;
    using BaseUint = uint64;
    static constexpr int bodyPrecision = 16;
    static constexpr BaseUint bodyFactor = 10000000000000000;
};

//================================================================
//
// FloatOptions
//
// Precision meaning:
// Flexible & Exponential: Max body digits.
// Fixed: Digits after decimal point.
//
//================================================================

enum class FloatFormat: char {Flexible, Fixed, Exponential};

////

struct FloatOptions
{
    FloatFormat format = FloatFormat::Flexible;
    bool forceBodySign = false;
    bool forceExpoSign = false;
    bool forceExponent = false;
    bool alignExponent = true;
    bool uppercase = false;
    Fill fill = Fill::Space;
    int precision = 0; // 0 means "default" (except Fixed mode).
    int minWidth = 0;
};

//================================================================
//
// writeFloat
//
// Returns false only in case of internal error (highly unlikely).
//
//================================================================

template <typename Char, typename Writer, typename Float>
sysinline bool writeFloat(Float value, const Writer& writer, const FloatOptions& options)
{
    COMPILE_ASSERT(TYPE_EQUAL(Float, float) || TYPE_EQUAL(Float, double));
    constexpr int binaryDigits = TYPE_EQUAL(Float, float) ? FLT_MANT_DIG  : DBL_MANT_DIG;
    using Traits = FloatTraits<binaryDigits>;

    ////

    using BaseInt = typename Traits::BaseInt;
    using BaseUint = typename Traits::BaseUint;

    ////

    auto originalValue = value;

    //----------------------------------------------------------------
    //
    // Print sign.
    // Also works for +-INF, but not for NAN.
    //
    //----------------------------------------------------------------

    Char signBuf = ' ';
    int signChars = 0;

    if (options.forceBodySign && value >= 0)
    {
        signBuf = '+';
        signChars = 1;
    }

    if (value < 0)
    {
        value = -value;
        signBuf = '-';
        signChars = 1;
    }

    //----------------------------------------------------------------
    //
    // writeSignAndPadding
    //
    //----------------------------------------------------------------

    auto writeSignAndPadding = [&] (int usedChars, Fill fill)
    {
        if (options.minWidth == 0)
        {
            writer(&signBuf, signChars);
            return;
        }

        ////

        int paddingChars = clampMin(options.minWidth, usedChars) - usedChars;

        ////

        if (fill == Fill::Zero)
        {
            writer(&signBuf, signChars);
            writeZeros<Char>(paddingChars, writer);
        }
        else
        {
            writeSpaces<Char>(paddingChars, writer);
            writer(&signBuf, signChars);
        }
    };

    //----------------------------------------------------------------
    //
    // Support of special numbers.
    //
    //----------------------------------------------------------------

    if_not (def(value))
    {
        constexpr int specialChars = 3;

        ////

        writeSignAndPadding(signChars + specialChars, Fill::Space); // No sense to use zero filling here.

        ////

        if (value != value)
        {
            static const Char bufferLo[] = {'n', 'a', 'n'};
            static const Char bufferUp[] = {'N', 'A', 'N'};
            COMPILE_ASSERT(COMPILE_ARRAY_SIZE(bufferLo) == specialChars);
            COMPILE_ASSERT(COMPILE_ARRAY_SIZE(bufferUp) == specialChars);
            writer(options.uppercase ? bufferUp : bufferLo, specialChars);
        }
        else
        {
            static const Char bufferLo[] = {'i', 'n', 'f'};
            static const Char bufferUp[] = {'I', 'N', 'F'};
            COMPILE_ASSERT(COMPILE_ARRAY_SIZE(bufferLo) == specialChars);
            COMPILE_ASSERT(COMPILE_ARRAY_SIZE(bufferUp) == specialChars);
            writer(options.uppercase ? bufferUp : bufferLo, specialChars);
        }

        return true;
    }

    //----------------------------------------------------------------
    //
    // Now value > 0.
    //
    //----------------------------------------------------------------

    ensure(def(value) && value >= 0);

    //----------------------------------------------------------------
    //
    // Frexp by base 10.
    //
    //----------------------------------------------------------------

    int dexponent = 0;

    ////

    if (value != 0)
    {

        #define TMP_MACRO(n) \
            TMP_MACRO_(n)

        #define TMP_MACRO_(n) \
            while (value >= Float(1e##n)) \
            { \
                value *= Float(1e-##n); \
                dexponent += n; \
            }

        TMP_MACRO(32)
        TMP_MACRO(16)
        TMP_MACRO(8)
        TMP_MACRO(4)
        TMP_MACRO(2)
        TMP_MACRO(1)

        #undef TMP_MACRO
        #undef TMP_MACRO_

        ////

        #define TMP_MACRO(n) \
            TMP_MACRO_(n)

        #define TMP_MACRO_(n) \
            while (value < Float(1e-##n)) \
            { \
                value *= Float(1e##n); \
                dexponent -= n; \
            }

        TMP_MACRO(32)
        TMP_MACRO(16)
        TMP_MACRO(8)
        TMP_MACRO(4)
        TMP_MACRO(2)
        TMP_MACRO(1)

        #undef TMP_MACRO
        #undef TMP_MACRO_

        //
        // Now value >= 0.1 && value < 10
        //

        while (value >= 1)
        {
            value *= Float(0.1);
            dexponent += 1;
        }

        //
        // Now value >= 0.1 && value < 1.
        //
    }

    //----------------------------------------------------------------
    //
    // Calculate desired body digits.
    //
    //----------------------------------------------------------------

    int bodyDigits = Traits::bodyPrecision; // Always in [1, Traits::bodyPrecision]

    ////

    if (options.format == FloatFormat::Fixed)
    {
        auto fracPrecision = clampMin(options.precision, 0); // >= 0

        //
        // bodyDigits = clampRange(dexponent + fracPrecision, -1, Traits::bodyPrecision)
        //
        // Avoid int overflow: "bodyPrecision" and "dexponent" are small, but "fracPrecision" may be big,
        // so move the range by (-dexponent).
        //
        // Range from -1 is needed for zero digits rounding which may increase bodyDigits by 1.
        //

        bodyDigits = dexponent + clampRange(fracPrecision, -1 - dexponent, Traits::bodyPrecision - dexponent);
        DEBUG_BREAK_CHECK(-1 <= bodyDigits && bodyDigits <= Traits::bodyPrecision);

        ////

        if (bodyDigits == 0 && value >= 0.5f) // God damn it.
        {
            value = Float(0.1);
            ++dexponent;
            ++bodyDigits;
        }

        bodyDigits = clampRange(bodyDigits, 0, Traits::bodyPrecision);
    }

    //----------------------------------------------------------------
    //
    // Round to the integer decimal mantissa.
    //
    // Here value is in [0.1, 1).
    //
    //----------------------------------------------------------------

    BaseUint mantissa = 0;

    ////

    if (bodyDigits == 0)
    {
        dexponent = 0;
    }
    else
    {
        COMPILE_ASSERT(Traits::bodyFactor >= 10);
        auto bodyFactorDiv10 = Traits::bodyFactor / 10;

        if (bodyDigits != Traits::bodyPrecision)
        {
            bodyFactorDiv10 = 1;

            auto n = bodyDigits - 1;

            for_count (i, n)
                bodyFactorDiv10 *= 10;
        }

        auto bodyFactor = bodyFactorDiv10 * 10;

        ////

        mantissa = BaseUint(BaseInt(value * Float(bodyFactor) + Float(0.5)));

        ////

        ensure(mantissa >= 0);

        //
        // Mantissa is either zero or in [~bodyFactor/10, ~bodyFactor]
        //
        // Ideally, it is like 100..999, but actually it could be like 099..1000.
        //
        // Max inaccuracy is defined by float, so it's rather precise (24+ bits).
        //

        if (mantissa != 0)
        {
            while (mantissa >= bodyFactor)
            {
                mantissa = (mantissa + 5) / 10; // Divide with rounding.
                dexponent += 1;
            }

            while (mantissa < bodyFactorDiv10)
            {
                mantissa *= 10;
                dexponent -= 1;
            }
        }
    }

    //----------------------------------------------------------------
    //
    // Body digits.
    //
    //----------------------------------------------------------------

    Char bodyArray[maxDecimalIntCharsWithoutSign];
    int bodyChars = 0;

    {
        auto* bodyPtr = bodyArray;

        auto bodyWriter = [&] (const Char* ptr, size_t size)
        {
            for_count (i, size)
                *bodyPtr++ = *ptr++;
        };

        writeUnsignedInt<Char>(BaseUint(mantissa), bodyWriter, {});

        bodyChars = bodyPtr - bodyArray;
    }

    //----------------------------------------------------------------
    //
    // Choose scaling. Either it uses fixed format,
    // or it uses exponential format with exponent which is multiple of 3
    // like nano, micro, milli, 1, kilo, mega, giga.
    //
    // Body in [0.1, 1), so log10(body) in [-1, 0).
    // log10(value) = dexponent - 1 + frac, where frac in [0, 1).
    //
    //----------------------------------------------------------------

    bool useFixed = (dexponent > -3 && dexponent < 6 + 1); // value >= 0.001 && value < 1 000 000

    if (options.format == FloatFormat::Fixed)
        useFixed = true;

    if (options.format == FloatFormat::Exponential)
        useFixed = false;

    ////

    int decimalPoint = 0;

    if (useFixed)
    {
        decimalPoint = dexponent;
        dexponent = 0;
    }
    else
    {
        if (mantissa != 0 && options.alignExponent)
        {
            // Want decimalPoint to be in [1, 4), so the integer part is in [1, 999].
            decimalPoint += 1;
            dexponent -= 1;

            // Now increase decimalPoint and decrease dexponent until (dexponent % 3 == 0).
            auto rem = dexponent - divDown(dexponent, 3) * 3;
            dexponent -= rem;
            decimalPoint += rem;
        }
    }

    //----------------------------------------------------------------
    //
    // Integer part.
    //
    //----------------------------------------------------------------

    int intTrailingZeros = 0;
    int intBodyChars = 0;

    {
        auto intDigits = clampMin(decimalPoint, 0);

        if (intDigits == 0)
            intTrailingZeros = 1;
        else
        {
            intBodyChars = clampMax(intDigits, bodyChars);
            intTrailingZeros = intDigits - intBodyChars;
        }
    }

    //----------------------------------------------------------------
    //
    // Fractional part.
    //
    //----------------------------------------------------------------

    //
    // Leading zeros?
    //

    int fracLeadingZeros = 0;

    if (decimalPoint < 0)
    {
        fracLeadingZeros = -decimalPoint;
        decimalPoint = 0;
    }

    //
    // Skip trailing zeros.
    //

    auto fracPtr = bodyArray + clampMax(decimalPoint, bodyChars);
    int fracChars = 0;

    {
        auto fracEnd = bodyArray + bodyChars;

        while (fracEnd != fracPtr && *(fracEnd - 1) == '0')
            --fracEnd;

        fracChars = fracEnd - fracPtr;
    }

    ////

    int fracTralingZeros = 0;

    if (options.format == FloatFormat::Fixed)
    {
        auto n = fracLeadingZeros + fracChars;

        if (options.precision > n)
            fracTralingZeros = options.precision - n;
    }
    else
    {
        auto totalDigits = intBodyChars + intTrailingZeros + fracChars;

        if (options.precision > totalDigits)
            fracTralingZeros = options.precision - totalDigits;
    }

    //----------------------------------------------------------------
    //
    // Exponent.
    //
    //----------------------------------------------------------------

    Char expoArray[maxDecimalIntCharsWithSign + 1]; // Plus 1 for 'E' char.
    int expoChars = 0;

    ////

    if (dexponent || options.forceExponent)
    {
        auto* expoPtr = expoArray;
        *expoPtr++ = options.uppercase ? 'E' : 'e';

        auto expoWriter = [&] (const Char* ptr, size_t size)
        {
            for_count (i, size)
                *expoPtr++ = *ptr++;
        };

        IntOptions opt;
        opt.forceSign = options.forceExpoSign;
        writeSignedInt<Char>(dexponent, expoWriter, opt);

        expoChars = expoPtr - expoArray;
    }

    //----------------------------------------------------------------
    //
    // * Count everything.
    // * Print sign and padding.
    // * Print the rest.
    //
    //----------------------------------------------------------------

    if (options.minWidth == 0)
    {
        writer(&signBuf, signChars);
    }
    else
    {
        int usedChars = signChars + intBodyChars + intTrailingZeros;

        if (fracChars + fracTralingZeros != 0)
            usedChars += 1 + fracLeadingZeros + (fracChars + fracTralingZeros);

        usedChars += expoChars;

        ////

        writeSignAndPadding(usedChars, options.fill);
    }

    //
    // Integer part.
    //

    writer(bodyArray, intBodyChars);
    writeZeros<Char>(intTrailingZeros, writer);

    //
    // Fractional part.
    //

    if (fracChars + fracTralingZeros != 0)
    {
        Char dot = '.';
        writer(&dot, 1);

        writeZeros<Char>(fracLeadingZeros, writer);

        writer(fracPtr, fracChars);

        writeZeros<Char>(fracTralingZeros, writer);
    }

    //
    // Exponent.
    //

    writer(expoArray, expoChars);

    ////

    return true;
}

//----------------------------------------------------------------

}

using writeToolsImpl::writeInt;
using writeToolsImpl::writeFloat;
