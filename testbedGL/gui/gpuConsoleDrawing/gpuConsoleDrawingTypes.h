#pragma once

#include "vectorTypes/vectorBase.h"
#include "charType/charType.h"

//================================================================
//
// ConsoleElement
//
// Contains ASCII letter and RGB color.
//
//================================================================

using ConsoleElement = uint32;

//================================================================
//
// consoleElementLetter
//
//================================================================

sysinline CharType consoleElementLetter(const ConsoleElement& value)
{
    return value >> 24;
}

//================================================================
//
// consoleElementColor
//
//================================================================

sysinline uint32 consoleElementColor(const ConsoleElement& value)
{
    return value & 0xFFFFFF;
}

//================================================================
//
// consoleElementCompose
//
//================================================================

sysinline uint32 consoleElementCompose(CharType letter, uint32 color)
{
    return (uint32(letter) << 24) | (color & 0xFFFFFF);
}
