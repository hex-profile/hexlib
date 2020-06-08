#pragma once

#ifndef HEXLIB_VECTOR_BASE_TYPES
#define HEXLIB_VECTOR_BASE_TYPES

#include "types/intTypes.h"
#include "types/floatTypes.h"
#include "types/compileTools.h"

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Define vector types.
//
// The header should define the following vector types: base_x2 and base_x4 
// where the base is {bool, int8, uint8, int16, uint16, int32, uint32, float16, float32}.
//
// For each type, the function make_[type] should be defined,
// where [type] is the name of the vector type.
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Define vector types.
//
// The header should define the following vector types: base_x2 and base_x4 
// where the base is {bool, int8, uint8, int16, uint16, int32, uint32, float16, float32}.
//
// For each type, the function make_[type] should be defined,
// where [type] is the name of the vector type.
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// TMP_STRUCT_WITH_ALIGNMENT
//
//================================================================

#define TMP_STRUCT_WITH_ALIGNMENT(name, byteAlignment) \
    struct alignas(byteAlignment) name

//================================================================
//
// CUDA Types
//
//================================================================

#if defined(__CUDA_ARCH__)

#define TMP_MACRO(scalar, baseScalar, cudaScalar) \
    \
    using scalar##_x2 = cudaScalar##2; \
    \
    HEXLIB_INLINE scalar##_x2 make_##scalar##_x2(baseScalar x, baseScalar y) \
        {return make_##cudaScalar##2(x, y);} \
    \
    using scalar##_x4 = cudaScalar##4; \
    \
    HEXLIB_INLINE scalar##_x4 make_##scalar##_x4(baseScalar x, baseScalar y, baseScalar z, baseScalar w) \
        {return make_##cudaScalar##4(x, y, z, w);}

////

TMP_MACRO(int8, int8_t, char)
TMP_MACRO(uint8, uint8_t, uchar)

TMP_MACRO(int16, int16_t, short)
TMP_MACRO(uint16, uint16_t, ushort)

TMP_MACRO(int32, int32_t, int)
TMP_MACRO(uint32, uint32_t, uint)

TMP_MACRO(float32, float32, float)

////

#undef TMP_MACRO

#endif

//================================================================
//
// 2-component
//
//================================================================

#define TMP_MACRO(vector, scalar, byteAlignment) \
    \
    TMP_STRUCT_WITH_ALIGNMENT(vector, byteAlignment) \
    { \
        scalar x; \
        scalar y; \
    }; \
    \
    HEXLIB_INLINE vector make_##vector(scalar x, scalar y) \
    { \
        vector tmp; \
        tmp.x = x; \
        tmp.y = y; \
        return tmp; \
    }

TMP_MACRO(bool_x2, bool, 2)
TMP_MACRO(float16_x2, float16, 4)

#if !defined(__CUDA_ARCH__)

TMP_MACRO(int8_x2, int8_t, 2)
TMP_MACRO(uint8_x2, uint8_t, 2)

TMP_MACRO(int16_x2, int16_t, 4)
TMP_MACRO(uint16_x2, uint16_t, 4)

TMP_MACRO(int32_x2, int32_t, 8)
TMP_MACRO(uint32_x2, uint32_t, 8)
TMP_MACRO(float32_x2, float32, 8)

#endif

////

#undef TMP_MACRO

//================================================================
//
// 4-component
//
//================================================================

#define TMP_MACRO(vector, scalar, byteAlignment) \
    \
    TMP_STRUCT_WITH_ALIGNMENT(vector, byteAlignment) \
    { \
        scalar x; \
        scalar y; \
        scalar z; \
        scalar w; \
    }; \
    \
    HEXLIB_INLINE vector make_##vector(scalar x, scalar y, scalar z, scalar w) \
    { \
        vector tmp; \
        tmp.x = x; \
        tmp.y = y; \
        tmp.z = z; \
        tmp.w = w; \
        return tmp; \
    }

TMP_MACRO(bool_x4, bool, 4)
TMP_MACRO(float16_x4, float16, 8)

#if !defined(__CUDA_ARCH__)

TMP_MACRO(int8_x4, int8_t, 4)
TMP_MACRO(uint8_x4, uint8_t, 4)

TMP_MACRO(int16_x4, int16_t, 8)
TMP_MACRO(uint16_x4, uint16_t, 8)

TMP_MACRO(int32_x4, int32_t, 16)
TMP_MACRO(uint32_x4, uint32_t, 16)
TMP_MACRO(float32_x4, float32, 16)

#endif

////

#undef TMP_MACRO

//================================================================
//
// Undef
//
//================================================================

#undef TMP_STRUCT_WITH_ALIGNMENT

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Check that vector types have identical size and alignment on all systems.
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
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

#endif // HEXLIB_VECTOR_BASE_TYPES
