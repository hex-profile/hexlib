#pragma once

#ifndef HEXLIB_VECTOR_BASE_TYPES
#define HEXLIB_VECTOR_BASE_TYPES

#include "compileTools/compileTools.h"
#include "numbers/float/floatBase.h"
#include "numbers/int/intBase.h"
#include "vectorTypes/half/halfBase.h"

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

#if defined(__CUDA_ARCH__)

    #define TMP_STRUCT_WITH_ALIGNMENT(name, byteAlignment) \
        __declspec(align(byteAlignment)) struct name

#else

    #define TMP_STRUCT_WITH_ALIGNMENT(name, byteAlignment) \
        struct alignas(byteAlignment) name

#endif

//================================================================
//
// CUDA Types
//
//================================================================

#if defined(__CUDA_ARCH__)

#define TMP_MACRO(scalar, cudaScalar) \
    \
    using scalar##_x2 = cudaScalar##2; \
    \
    sysinline scalar##_x2 make_##scalar##_x2(scalar x, scalar y) \
        {return make_##cudaScalar##2(x, y);} \
    \
    using scalar##_x4 = cudaScalar##4; \
    \
    sysinline scalar##_x4 make_##scalar##_x4(scalar x, scalar y, scalar z, scalar w) \
        {return make_##cudaScalar##4(x, y, z, w);}

////

TMP_MACRO(int8, char)
TMP_MACRO(uint8, uchar)

TMP_MACRO(int16, short)
TMP_MACRO(uint16, ushort)

TMP_MACRO(int32, int)
TMP_MACRO(uint32, uint)

TMP_MACRO(float32, float)

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
    sysinline vector make_##vector(scalar x, scalar y) \
    { \
        vector tmp; \
        tmp.x = x; \
        tmp.y = y; \
        return tmp; \
    }

TMP_MACRO(bool_x2, bool, 2)
TMP_MACRO(float16_x2, float16, 4)

#if !defined(__CUDA_ARCH__)

TMP_MACRO(int8_x2, int8, 2)
TMP_MACRO(uint8_x2, uint8, 2)

TMP_MACRO(int16_x2, int16, 4)
TMP_MACRO(uint16_x2, uint16, 4)

TMP_MACRO(int32_x2, int32, 8)
TMP_MACRO(uint32_x2, uint32, 8)
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
    sysinline vector make_##vector(scalar x, scalar y, scalar z, scalar w) \
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

TMP_MACRO(int8_x4, int8, 4)
TMP_MACRO(uint8_x4, uint8, 4)

TMP_MACRO(int16_x4, int16, 8)
TMP_MACRO(uint16_x4, uint16, 8)

TMP_MACRO(int32_x4, int32, 16)
TMP_MACRO(uint32_x4, uint32, 16)
TMP_MACRO(float32_x4, float32, 16)

#endif

////

#undef TMP_MACRO

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Check that vector types have identical size and alignment on all systems.
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//
// Check macro
//

#define TMP_MACRO(Type, typeSize, typeAlignment) \
    COMPILE_ASSERT(sizeof(Type) == typeSize); \
    COMPILE_ASSERT(alignof(Type) == typeAlignment)

//
// 2-component vectors
//

TMP_MACRO(bool_x2, 2, 2);

TMP_MACRO(int8_x2, 2, 2);
TMP_MACRO(uint8_x2, 2, 2);

TMP_MACRO(int16_x2, 4, 4);
TMP_MACRO(uint16_x2, 4, 4);

TMP_MACRO(int32_x2, 8, 8);
TMP_MACRO(uint32_x2, 8, 8);

TMP_MACRO(float16_x2, 4, 4);
TMP_MACRO(float32_x2, 8, 8);

//
// 4-component vectors
//

TMP_MACRO(bool_x4, 4, 4);

TMP_MACRO(int8_x4, 4, 4);
TMP_MACRO(uint8_x4, 4, 4);

TMP_MACRO(int16_x4, 8, 8);
TMP_MACRO(uint16_x4, 8, 8);

TMP_MACRO(int32_x4, 16, 16);
TMP_MACRO(uint32_x4, 16, 16);

TMP_MACRO(float16_x4, 8, 8);
TMP_MACRO(float32_x4, 16, 16);

//
// Finish
//

#undef TMP_MACRO

//----------------------------------------------------------------

#endif // #ifndef HEXLIB_VECTOR_BASE_TYPES

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Define FOREACH lists for vector types.
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// VECTOR_INT_X1_FOREACH
// VECTOR_INT_X2_FOREACH
// VECTOR_INT_X4_FOREACH
//
//================================================================

#define VECTOR_INT_X1_FOREACH(action, extra) \
    action(int8, extra) \
    action(uint8, extra) \
    action(int16, extra) \
    action(uint16, extra) \
    action(int32, extra) \
    action(uint32, extra)

#define VECTOR_INT_X2_FOREACH(action, extra) \
    action(int8_x2, extra) \
    action(uint8_x2, extra) \
    action(int16_x2, extra) \
    action(uint16_x2, extra) \
    action(int32_x2, extra) \
    action(uint32_x2, extra)

#define VECTOR_INT_X4_FOREACH(action, extra) \
    action(int8_x4, extra) \
    action(uint8_x4, extra) \
    action(int16_x4, extra) \
    action(uint16_x4, extra) \
    action(int32_x4, extra) \
    action(uint32_x4, extra)

//================================================================
//
// VECTOR_INT_FOREACH
//
//================================================================

#define VECTOR_INT_FOREACH(action, extra) \
    VECTOR_INT_X1_FOREACH(action, extra) \
    VECTOR_INT_X2_FOREACH(action, extra) \
    VECTOR_INT_X4_FOREACH(action, extra)

//================================================================
//
// VECTOR_FLOAT_X1_FOREACH
// VECTOR_FLOAT_X2_FOREACH
// VECTOR_FLOAT_X4_FOREACH
//
//================================================================

#define VECTOR_FLOAT_X1_FOREACH(action, extra) \
    action(float16, extra) \
    action(float32, extra)

#define VECTOR_FLOAT_X2_FOREACH(action, extra) \
    action(float16_x2, extra) \
    action(float32_x2, extra)

#define VECTOR_FLOAT_X4_FOREACH(action, extra) \
    action(float16_x4, extra) \
    action(float32_x4, extra)

//================================================================
//
// VECTOR_FLOAT_FOREACH
//
//================================================================

#define VECTOR_FLOAT_FOREACH(action, extra) \
    VECTOR_FLOAT_X1_FOREACH(action, extra) \
    VECTOR_FLOAT_X2_FOREACH(action, extra) \
    VECTOR_FLOAT_X4_FOREACH(action, extra)
