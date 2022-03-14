#pragma once

#include "formatting/formatStream.h"

//================================================================
//
// MessageFormatter
//
//================================================================

struct MessageFormatter : public FormatOutputStream
{
    virtual void clear() =0;
    virtual bool valid() =0;
    virtual size_t size() =0;
    virtual const CharType* cstr() =0;
    virtual CharArray charArray() =0;
};

//================================================================
//
// MessageFormatterNull
//
//================================================================

struct MessageFormatterNull : public MessageFormatter
{

    virtual void write(const CharType* bufferPtr, size_t bufferSize)
        {}

    ////

    #define TMP_MACRO(Type, o) \
        virtual void write(Type value) {} \
        virtual void write(const FormatNumber<Type>& value) {}

    BUILTIN_INT_FOREACH(TMP_MACRO, o)
    BUILTIN_FLOAT_FOREACH(TMP_MACRO, o)

    #undef TMP_MACRO

    ////

    virtual void clear() 
        {}

    virtual bool valid()
        {return false;}

    virtual size_t size()
        {return 0;}

    virtual const CharType* cstr()
        {return CT("");}

    virtual CharArray charArray()
        {return CharArray();}

};
