#pragma once

#include <ostream>

#include "formatting/formatStream.h"

//================================================================
//
// FormatStreamStlThunk
//
//================================================================

class FormatStreamStlThunk : public FormatOutputStream
{

public:

    void write(const CharType* bufferPtr, size_t bufferSize);

    //
    // Output builtin integers & floats
    //

    #define TMP_MACRO(Type, o) \
        void write(Type value); \
        void write(const FormatNumber<Type>& value);

    BUILTIN_INT_FOREACH(TMP_MACRO, o)
    BUILTIN_FLOAT_FOREACH(TMP_MACRO, o)

    #undef TMP_MACRO

public:

    inline FormatStreamStlThunk(std::basic_ostream<CharType>& outputStream)
        : theOk(true), outputStream(outputStream) {}

public:

    void clear() {theOk = true;}
    inline bool valid() {return theOk;}

private:

    bool theOk;

    std::basic_ostream<CharType>& outputStream;

private:

    template <typename Type>
    inline void printIntFloat(Type value, const FormatNumberOptions& options);

};
