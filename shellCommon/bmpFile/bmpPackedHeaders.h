#pragma once

#include "numbers/int/intBase.h"

//================================================================
//
// BMP file structures that use #pragma pack
//
//================================================================

//================================================================
//
// BitmapFileHeader
//
//================================================================

#pragma pack(push, 1)

struct BitmapFileHeader
{
    uint16 bfType;
    uint32 bfSize;
    uint16 bfReserved1;
    uint16 bfReserved2;
    uint32 bfOffBits;
};

COMPILE_ASSERT(sizeof(BitmapFileHeader) == (2*32 + 3*16) / 8);
COMPILE_ASSERT(alignof(BitmapFileHeader) == 1);

//----------------------------------------------------------------

struct BitmapInfoHeader
{
    uint32 biSize;
    int32 biWidth;
    int32 biHeight;
    uint16 biPlanes;
    uint16 biBitCount;
    uint32 biCompression;
    uint32 biSizeImage;
    int32 biXPelsPerMeter;
    int32 biYPelsPerMeter;
    uint32 biClrUsed;
    uint32 biClrImportant;
};

COMPILE_ASSERT(sizeof(BitmapInfoHeader) == (2*16 + 9*32) / 8);
COMPILE_ASSERT(alignof(BitmapInfoHeader) == 1);

//================================================================
//
// BitmapQuad
//
//================================================================

struct BitmapQuad
{
    uint8 rgbBlue;
    uint8 rgbGreen;
    uint8 rgbRed;
    uint8 rgbReserved;
};

COMPILE_ASSERT(sizeof(BitmapQuad) == 4);

//================================================================
//
// BitmapPalette
//
//================================================================

struct BitmapPalette
{
    BitmapQuad palette[256];
};

COMPILE_ASSERT(sizeof(BitmapPalette) == 256 * 4);
COMPILE_ASSERT(alignof(BitmapPalette) == 1);

//================================================================
//
// BitmapFullHeader
//
//================================================================

struct BitmapFullHeader
    : 
    public BitmapFileHeader, 
    public BitmapInfoHeader,
    public BitmapPalette
{
};

COMPILE_ASSERT(sizeof(BitmapFullHeader) == sizeof(BitmapFileHeader) + sizeof(BitmapInfoHeader) + sizeof(BitmapPalette));
COMPILE_ASSERT(alignof(BitmapFullHeader) == 1);

//----------------------------------------------------------------

#pragma pack(pop)
