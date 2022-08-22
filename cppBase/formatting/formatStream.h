#pragma once

#include "numbers/float/floatBase.h"
#include "formatting/formatNumberOptions.h"
#include "charType/charArray.h"

//================================================================
//
// FormatNumber
//
// A numeric value with options.
//
//================================================================

template <typename Type>
struct FormatNumber
{
    Type value;
    FormatNumberOptions options;
};

//----------------------------------------------------------------

template <typename Type>
sysinline FormatNumber<Type> formatNumber(const Type& value, const FormatNumberOptions& options)
{
    return {value, options};
}

//----------------------------------------------------------------

template <typename Type, typename Other>
sysinline FormatNumber<Type> formatNumber(const Type& value, const FormatNumber<Other>& example)
{
    return {value, example.options};
}

//================================================================
//
// FormatOutputStream
//
// Abstract interface: output stream of formatted text.
//
// Includes abstract interface for built-in integers and floats,
// for external implementation.
//
//================================================================

struct FormatOutputStream
{

    //
    // CharType array
    //

    virtual void write(const CharType* bufferPtr, size_t bufferSize) =0;

    inline void write(const CharArray& value)
        {write(value.ptr, value.size);}

    //
    // Output builtin integers & floats
    //

    #define TMP_MACRO(Type, o) \
        virtual void write(Type value) =0; \
        virtual void write(const FormatNumber<Type>& value) =0;

    BUILTIN_INT_FOREACH(TMP_MACRO, o)
    BUILTIN_FLOAT_FOREACH(TMP_MACRO, o)

    #undef TMP_MACRO

};

//================================================================
//
// formatOutput
//
// Core function of formatted output: output a value to the stream.
// The function is additionally defined for any required type.
//
//================================================================

template <typename Type>
void formatOutput(const Type& value, FormatOutputStream& outputStream);

//----------------------------------------------------------------

template <typename Type>
sysinline FormatOutputStream& operator <<(FormatOutputStream& outputStream, const Type& value)
{
    formatOutput(value, outputStream);
    return outputStream;
}

//----------------------------------------------------------------

template <typename Type>
struct FormatOutputFunc
{
    typedef void FuncType(const Type& value, FormatOutputStream& outputStream);
    static sysinline FuncType* get() {return &formatOutput<Type>;}
};
