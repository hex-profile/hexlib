#pragma once

#include "formatting/formatStream.h"

//================================================================
//
// PrettyNumber
//
//================================================================

template <typename Type>
class PrettyNumber
{

public:

    sysinline PrettyNumber(const FormatNumber<Type>& number)
        :
        number(number)
    {
    }

public:

    FormatNumber<Type> number;

};

//================================================================
//
// prettyNumber
//
//================================================================

template <typename Type>
sysinline PrettyNumber<Type> prettyNumber(const FormatNumber<Type>& number)
{
    return PrettyNumber<Type>(number);
}
