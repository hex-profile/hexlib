#pragma once

#include "charType/charType.h"
#include "formatting/formatOutputAtom.h"

//================================================================
//
// defaultSpecialChar
//
//================================================================

constexpr CharType defaultSpecialChar = '%';

//================================================================
//
// ParamMsg
//
// Core formatting function, takes format string in printf-style
// and array of values.
//
//================================================================

struct ParamMsg : public FormatOutputAtom
{

public:

    sysinline ParamMsg(CharType specialChar, const CharArray& format, const FormatOutputAtom* paramPtr, size_t paramSize)
        :
        specialChar(specialChar), format(format), paramPtr(paramPtr), paramSize(paramSize)
    {
        value = this;
        func = outputFunc;
    }

    const FormatOutputAtom& operator ~() const
    {
        return *this;
    }

private:

    static void outputFunc(const void* value, FormatOutputStream& outputStream);

private:

    CharType specialChar;
    CharArray format;
    const FormatOutputAtom* paramPtr;
    size_t paramSize;

};
