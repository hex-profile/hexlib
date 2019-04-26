#pragma once

#include "vectorTypes/vectorBase.h"

//================================================================
//
// BSPLINE_CUBIC_PREFILTER_RANGE_EXTENSION_FACTOR
//
//================================================================

#define BSPLINE_CUBIC_PREFILTER_RANGE_EXTENSION_FACTOR 3.f

//================================================================
//
// BsplineExtendedType
//
//================================================================

template <typename Type>
struct BsplineExtendedType;

////

#define TMP_MACRO(Src, Dst) \
    template <> struct BsplineExtendedType<Src> {using T = Dst;};

TMP_MACRO(float16, float16)
TMP_MACRO(float16_x2, float16_x2)
TMP_MACRO(float16_x4, float16_x4)

TMP_MACRO(int8, float16)
TMP_MACRO(int8_x2, float16_x2)
TMP_MACRO(int8_x4, float16_x4)

TMP_MACRO(uint8, float16)
TMP_MACRO(uint8_x2, float16_x2)
TMP_MACRO(uint8_x4, float16_x4)

#undef TMP_MACRO
