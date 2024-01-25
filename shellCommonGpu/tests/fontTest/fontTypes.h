#pragma once

#include "numbers/int/intBase.h"
#include "point/pointBase.h"

namespace fontTypes {

//================================================================
//
// fontTypesAligment
//
// Common alignment of font structures, so that
// they can be tightly placed one after another.
//
//================================================================

constexpr size_t fontTypesAligment = 4;

//================================================================
//
// FontInt
//
// Measured in pixels (not in fractions of a pixel).
// Letter sizes and offsets in pixels are limited to 0x7FFF.
//
//================================================================

using FontInt = int16;

//================================================================
//
// FontLetter
//
//================================================================

struct FontLetter
{
    // How many pixels to move the cursor after drawing this letter.
    Point<FontInt> advance;

    // How many pixels to add to the cursor to get the coordinates
    // where the upper left corner of the letter should be drawn. Always >= 0.
    Point<FontInt> org;

    // Letter dimensions in pixels.
    Point<FontInt> size;

    // The start of this letter's buffer in the overall font buffer.
    int32 offset;
};

COMPILE_ASSERT(sizeof(FontLetter) == 16 && alignof(FontLetter) == fontTypesAligment);

//================================================================
//
// FontHeader
//
//================================================================

struct FontHeader
{
    // Range of supported characters.
    int32 rangeOrg;
    int32 rangeEnd;

    // Index of the character used to display unsupported characters.
    int32 unsupportedCode;
};

COMPILE_ASSERT(sizeof(FontHeader) == 12 && alignof(FontHeader) == fontTypesAligment);

//----------------------------------------------------------------

}
