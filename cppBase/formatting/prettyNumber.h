#pragma once

#include "formatting/formatStream.h"

//================================================================
//
// PrettyNumber
//
//================================================================

template <typename Type>
struct PrettyNumber
{
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
    return {number};
}
