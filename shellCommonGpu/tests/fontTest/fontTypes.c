#include "fontTypes.h"

using namespace fontTypes;

//================================================================
//
// ExampleFont
//
//================================================================

struct ExampleFont
{
    FontHeader header;
    FontLetter letters[1];
    uint8 buffer[3];
};

COMPILE_ASSERT(alignof(ExampleFont) == fontTypesAligment);

////

static const ExampleFont exampleFontData =
{
    // header
    {
        32, // rangeOrg
        128, // rangeEnd
        127 // unsupportedCode
    },
    // letters
    {
        {
            {10, 12}, // advance
            {1, 2}, // org
            {16, 19}, // size
            1000 // offset
        }
    },
    // buffer
    {
        0x00, 0xFF, 0xFF
    }
};

////

const FontHeader& exampleFont()
{
    return exampleFontData.header;
}
