#include "messageFormatterStl.h"

#include "numbers/int/intType.h"
#include "numbers/float/floatType.h"

//================================================================
//
// MessageFormatterStl::write
//
//================================================================

void MessageFormatterStl::write(const CharType* bufferPtr, size_t bufferSize)
{
    using namespace std;

    if_not (theOk) return;
    theOk = false;

    try
    {
        ensurev(bufferSize >= 0);

        streamsize size = 0;
        ensurev(convertExact(bufferSize, size));

        outputStream.write(bufferPtr, bufferSize);
        theOk = true;
    }
    catch (const exception&) {}
}

//================================================================
//
// MessageFormatterStl::printIntFloat
//
//================================================================

template <typename Type>
inline void MessageFormatterStl::printIntFloat(Type value, const FormatNumberOptions& options)
{
    using namespace std;

    if_not (theOk) return;
    theOk = false;

    try
    {

        //
        // Flags
        //

        ios_base::fmtflags flags = ios_base::uppercase;

        ////

        if (options.fillIsZero())
            flags |= ios_base::internal;
        else
            flags |= ios_base::right;

        ////

        if (options.plusIsOn())
            flags |= ios_base::showpos;

        ////

        if (options.baseIsDec())
            flags |= ios_base::dec;

        if (options.baseIsHex())
            flags |= ios_base::hex;

        ////

        if (options.fformIsF())
            flags |= ios_base::fixed;

        if (options.fformIsE())
            flags |= ios_base::scientific;

        ////

        outputStream.flags(flags);

        //
        // Fill / Width / Precision
        //

        outputStream.fill(options.fillIsZero() ? '0' : ' ');

        outputStream.width(options.getWidth());

        outputStream.precision(options.getPrecision());

        //
        //
        //

        if (TYPE_EQUAL(Type, bool))
        {
            outputStream << (value ? CT("ON") : CT("OFF"));
        }
        else if (!TYPE_IS_BUILTIN_FLOAT(Type))
        {
            COMPILE_ASSERT(sizeof(int) == sizeof(unsigned));

            if_not (sizeof(Type) < sizeof(int))
                outputStream << value;
            else
            {
                if (TYPE_IS_SIGNED(Type))
                    outputStream << int(value);
                else
                    outputStream << unsigned(value);
            }
        }
        else
        {
            if_not (def(value))
                outputStream.write(CT("NAN"), 3);
            else
                outputStream << value;
        }

        theOk = true;
    }
    catch (const exception&) {}

}

//================================================================
//
// MessageFormatterStl::write<BuiltinInt>
//
//================================================================

#define TMP_MACRO(Type, o) \
    \
    void MessageFormatterStl::write(Type value) \
        {printIntFloat(value, FormatNumberOptions());} \
    \
    void MessageFormatterStl::write(const FormatNumber<Type>& value) \
        {printIntFloat(value.value, value.options);} \

BUILTIN_INT_FOREACH(TMP_MACRO, o)
BUILTIN_FLOAT_FOREACH(TMP_MACRO, o)

#undef TMP_MACRO
