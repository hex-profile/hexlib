#pragma once

#include "charType/charArray.h"
#include "numbers/int/intType.h"
#include "numbers/float/floatType.h"
#include "storage/adapters/lambdaThunk.h"

//================================================================
//
// Pure virtual=0 interfaces to support storage of the configuration variables in text form.
//
//================================================================

//================================================================
//
// Basic interface tools to convert data to/from text format.
// These tools are passed to a configuration variable when it is required
// to serialize to/from text format.
//
// These interfaces are introduced to decouple cfgvars from the implementation of strings and format I/O.
// They provide basic IO support: strings, built-in integers and floating point numbers.
//
// (Cfg text serialization needs to store data with higher precision than display format output).
//
//================================================================

//================================================================
//
// CfgWriteStream
//
// Basic tools for saving in text format
//
//================================================================

struct CfgWriteStream
{

    //
    // Save character array.
    // Save C null-terminated string.
    //

    virtual bool writeChars(const CharType* arrayPtr, size_t arraySize) =0;

    sysinline bool writeStr(const CharArray& str)
        {return writeChars(str.ptr, str.size);}

    //
    // Save builtin integers
    //

    #define TMP_MACRO(Type, o) \
        virtual bool writeValue(const Type& value) =0;

    BUILTIN_INT_FOREACH(TMP_MACRO, o)

    #undef TMP_MACRO

    //
    // Save builtin floats
    //

    #define TMP_MACRO(Type, o) \
        virtual bool writeValue(const Type& value) =0;

    BUILTIN_FLOAT_FOREACH(TMP_MACRO, o)

    #undef TMP_MACRO

};

//================================================================
//
// CfgOutputString
//
// Interface to return a string,
// which is not bound to specific string implementation.
//
//================================================================

struct CfgOutputString
{
    virtual bool addBuf(const CharType* bufArray, size_t bufSize) =0;

    sysinline bool addStr(const CharArray& str)
        {return addBuf(str.ptr, str.size);}
};

LAMBDA_THUNK
(
    cfgOutputString,
    CfgOutputString,
    bool addBuf(const CharType* bufArray, size_t bufSize),
    lambda(bufArray, bufSize)
)

//================================================================
//
// CfgReadStream
//
// Basic tools for loading from text format.
//
//================================================================

struct CfgReadStream
{

    //
    // Skip spaces and tabs.
    //

    virtual void skipSpaces() =0;

    //
    // Skip the specified text.
    //

    virtual bool skipText(const CharArray& text) =0;

    //
    // Read builtin integers.
    //

    #define TMP_MACRO(Type, o) \
        virtual bool readValue(Type& value) =0;

    BUILTIN_INT_FOREACH(TMP_MACRO, o)
    BUILTIN_FLOAT_FOREACH(TMP_MACRO, o)

    #undef TMP_MACRO

    //
    // Read string: reads everything until the stream's end.
    //

    virtual bool readAll(CfgOutputString& result) =0;

};

//================================================================
//
// CfgFamily
//
//================================================================

template <typename Type>
struct CfgFamily
{
    using T = CONVERT_FAMILY(Type);
};

//================================================================
//
// CfgWrite
//
// Extendable template function to save any type to text format.
// It can be extended for user Type.
//
//================================================================

template <typename Family>
struct CfgWrite;

//----------------------------------------------------------------

template <typename Type>
sysinline bool cfgWrite(CfgWriteStream& s, const Type& value)
    {return CfgWrite<typename CfgFamily<Type>::T>::func(s, value);}

//================================================================
//
// CfgWrite for common types
//
//================================================================

//
// Builtin integers
//

template <>
struct CfgWrite<BuiltinInt>
{
    template <typename Type>
    static sysinline bool func(CfgWriteStream& s, const Type& value)
        {return s.writeValue(value);}
};

//
// Builtin floats
//

template <>
struct CfgWrite<BuiltinFloat>
{
    template <typename Type>
    static sysinline bool func(CfgWriteStream& s, const Type& value)
        {return s.writeValue(value);}
};

//
// CharArray string
//

template <>
struct CfgWrite<CharArray>
{
    static sysinline bool func(CfgWriteStream& s, const CharArray& value)
        {return s.writeStr(value);}
};

//================================================================
//
// CfgRead
//
// Extendable template function to read any type from text format.
// It can be extended for user Type.
//
//================================================================

template <typename Family>
struct CfgRead;

//----------------------------------------------------------------

template <typename Type>
sysinline bool cfgRead(CfgReadStream& s, Type& value)
    {return CfgRead<typename CfgFamily<Type>::T>::func(s, value);}

//================================================================
//
// CfgRead for common types
//
//================================================================

//
// Builtin integers & floats
//

template <>
struct CfgRead<BuiltinInt>
{
    template <typename Type>
    static sysinline bool func(CfgReadStream& s, Type& value)
        {return s.readValue(value);}
};

template <>
struct CfgRead<BuiltinFloat>
{
    template <typename Type>
    static sysinline bool func(CfgReadStream& s, Type& value)
        {return s.readValue(value);}
};
