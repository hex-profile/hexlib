#pragma once

#include "charType/charType.h"
#include "formatting/formatOutputAtom.h"

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

    inline ParamMsg(const CharArray& format, const FormatOutputAtom* paramPtr, size_t paramSize)
        :
        format(format), paramPtr(paramPtr), paramSize(paramSize)
    {
        value = this;
        func = outputFunc;
    }

private:

    static void outputFunc(const void* value, FormatOutputStream& outputStream);

private:

    CharArray format;
    const FormatOutputAtom* paramPtr;
    size_t paramSize;

};
