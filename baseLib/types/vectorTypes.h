#pragma once

#include "types/intTypes.h"
#include "types/floatTypes.h"

#ifndef HEXLIB_HOST_VECTOR_BASE_TYPES
#define HEXLIB_HOST_VECTOR_BASE_TYPES

//================================================================
//
// Vector types.
//
// The header should define the following vector types: base_x2 and base_x4 
// where the base is {bool, int8, uint8, int16, uint16, int32, uint32, float16, float32}.
//
// For each type, the function make_[type] should be defined,
// where [type] is the name of the vector type.
//
//================================================================

//================================================================
//
// Vector types (2-component)
//
//================================================================

#define TMP_MACRO_X2(vector, scalar, byteAlignment) \
    \
    struct alignas(byteAlignment) vector \
    { \
        scalar x; \
        scalar y; \
    }; \
    \
    inline vector make_##vector(scalar x, scalar y) \
    { \
        vector tmp; \
        tmp.x = x; \
        tmp.y = y; \
        return tmp; \
    }

TMP_MACRO_X2(bool_x2, bool, 2)

TMP_MACRO_X2(int8_x2, int8_t, 2)
TMP_MACRO_X2(uint8_x2, uint8_t, 2)

TMP_MACRO_X2(int16_x2, int16_t, 4)
TMP_MACRO_X2(uint16_x2, uint16_t, 4)

TMP_MACRO_X2(int32_x2, int32_t, 8)
TMP_MACRO_X2(uint32_x2, uint32_t, 8)

TMP_MACRO_X2(float16_x2, float16, 4)
TMP_MACRO_X2(float32_x2, float32, 8)

#undef TMP_MACRO_X2

//================================================================
//
// Vector types (4-component)
//
//================================================================

#define TMP_MACRO_X4(vector, scalar, byteAlignment) \
    \
    struct alignas(byteAlignment) vector \
    { \
        scalar x; \
        scalar y; \
        scalar z; \
        scalar w; \
    }; \
    \
    inline vector make_##vector(scalar x, scalar y, scalar z, scalar w) \
    { \
        vector tmp; \
        tmp.x = x; \
        tmp.y = y; \
        tmp.z = z; \
        tmp.w = w; \
        return tmp; \
    }

TMP_MACRO_X4(bool_x4, bool, 4)

TMP_MACRO_X4(int8_x4, int8_t, 4)
TMP_MACRO_X4(uint8_x4, uint8_t, 4)

TMP_MACRO_X4(int16_x4, int16_t, 8)
TMP_MACRO_X4(uint16_x4, uint16_t, 8)

TMP_MACRO_X4(int32_x4, int32_t, 16)
TMP_MACRO_X4(uint32_x4, uint32_t, 16)

TMP_MACRO_X4(float16_x4, float16, 8)
TMP_MACRO_X4(float32_x4, float32, 16)

#undef TMP_MACRO_X4

//================================================================
//
// Check that vector types have identical size and alignment on all systems.
//
//================================================================

#define TMP_MACRO(Type, typeSize, typeAlignment) \
    static_assert(sizeof(Type) == typeSize && alignof(Type) == typeAlignment, "")

////

TMP_MACRO(bool_x2, 2, 2);

TMP_MACRO(int8_x2, 2, 2);
TMP_MACRO(uint8_x2, 2, 2);

TMP_MACRO(int16_x2, 4, 4);
TMP_MACRO(uint16_x2, 4, 4);

TMP_MACRO(int32_x2, 8, 8);
TMP_MACRO(uint32_x2, 8, 8);

TMP_MACRO(float16_x2, 4, 4);
TMP_MACRO(float32_x2, 8, 8);

////

TMP_MACRO(bool_x4, 4, 4);

TMP_MACRO(int8_x4, 4, 4);
TMP_MACRO(uint8_x4, 4, 4);

TMP_MACRO(int16_x4, 8, 8);
TMP_MACRO(uint16_x4, 8, 8);

TMP_MACRO(int32_x4, 16, 16);
TMP_MACRO(uint32_x4, 16, 16);

TMP_MACRO(float16_x4, 8, 8);
TMP_MACRO(float32_x4, 16, 16);

////

#undef TMP_MACRO

//----------------------------------------------------------------

#endif // HEXLIB_HOST_VECTOR_BASE_TYPES
