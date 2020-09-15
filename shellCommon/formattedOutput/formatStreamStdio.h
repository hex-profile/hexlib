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

    FormatStreamStdioThunk() 
        {}

    FormatStreamStdioThunk(CharType* newArray, size_t newSize)
        {setMemory(newArray, newSize);}

public:

    inline void setMemory(CharType* newArray, size_t newSize)
    {
        ok = false;
        memoryArray = nullptr;
        memorySize = 0;
        usedSize = 0;

        if (newSize >= 1)
        {
            memoryArray = newArray;
            memorySize = newSize - 1; // reserve space for NUL terminator
            ok = true;
        }
    }

public:

    inline bool valid() {return ok;}
    inline size_t size() {return usedSize;}

private:

    bool ok = false;

    // Not used if memorySize == 0.
    CharType* memoryArray = nullptr; 

    // Not including NUL terminator, memorySize >= 0
    size_t memorySize = 0; 

    // Not including NUL terminator, 0 <= usedSize <= memorySize
    size_t usedSize = 0; 

private:

    template <typename Type>
    inline void printIntFloat(Type value, const FormatNumberOptions& options);

};
