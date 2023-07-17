#pragma once

#include <string>

#include "charType/charArray.h"

//================================================================
//
// StlString
//
// STL string
//
//================================================================

using StlString = std::basic_string<CharType>;

//================================================================
//
// charArrayFromStl
//
//================================================================

template <typename Type>
sysinline CharArrayEx<Type> charArrayFromStl(const std::basic_string<Type>& str)
{
    return {str.data(), str.size()};
}
