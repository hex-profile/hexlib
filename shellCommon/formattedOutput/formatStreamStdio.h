#pragma once

#include "formatting/formatStream.h"
#include "numbers/int/intType.h"

//================================================================
//
// FormatStreamStdioThunk
//
//================================================================

class FormatStreamStdioThunk : public FormatOutputStream
{

public:

    virtual void write(const CharType* bufferPtr, size_t bufferSize);

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

    inline FormatStreamStdioThunk(CharType* bufferArray, size_t bufferCapacity)
        :
        theBufferArray(bufferArray),
        theBufferCapacity(bufferCapacity),
        theBufferSize(0),
        theOk(true)
    {
    }

public:

    inline bool valid() {return theOk;}
    inline size_t usedSize() {return theBufferSize;}

private:

    bool theOk;

    CharType* const theBufferArray; // not used if theBufferCapacity == 0
    size_t const theBufferCapacity; // >= 0
    size_t theBufferSize; // <= theBufferCapacity

private:

    template <typename Type>
    inline void printIntFloat(Type value, const FormatNumberOptions& options);

};
