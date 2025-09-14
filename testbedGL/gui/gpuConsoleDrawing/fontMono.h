#pragma once

#include "kit/kit.h"
#include "data/matrix.h"
#include "data/gpuPtr.h"

//================================================================
//
// FontMono
//
//================================================================

template <typename Pointer>
struct FontMono
{
    // data
    ArrayEx<Pointer> data;

    // the width and height of a character
    Point<Space> charSize;

    // the value of font element which means non-empty pixel
    TYPE_CLEANSE(typename PtrElemType<Pointer>::T) dotValue;

    // supported character range
    Space rangeOrg;
    Space rangeSize;
};

//----------------------------------------------------------------

template <typename Pointer>
inline FontMono<Pointer> fontMono
(
    const ArrayEx<Pointer>& data,
    const Point<Space>& charSize,
    TYPE_CLEANSE(typename PtrElemType<Pointer>::T) dotValue,
    Space rangeOrg, Space rangeSize
)
{
    FontMono<Pointer> result;
    result.data = data;
    result.charSize = charSize;
    result.dotValue = dotValue;
    result.rangeOrg = rangeOrg;
    result.rangeSize = rangeSize;
    return result;
}

//================================================================
//
// fontMonoNull
//
//================================================================

template <typename Pointer>
inline FontMono<Pointer> fontMonoNull()
{
    return fontMono<Pointer>({}, point(1), 0, 0, 0);
}

//================================================================
//
// fontValid
//
//================================================================

template <typename Pointer>
inline bool fontValid(const FontMono<Pointer>& font)
{
    ensure(font.charSize >= 1);
    ensure(font.rangeOrg >= 0 && font.rangeSize >= 0);
    ensure(font.data.size() % areaOf(font.charSize) == 0);
    ensure(font.data.size() / areaOf(font.charSize) == font.rangeSize);
    return true;
}

//================================================================
//
// FontElement
//
//================================================================

using FontElement = char;

using CpuFontMono = FontMono<CpuPtr(const FontElement)>;
using GpuFontMono = FontMono<GpuPtr(const FontElement)>;
