#pragma once

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

#if defined(__CUDA_ARCH__)
    #include "vectorBaseCuda.inl"
#else
    #ifndef HEXLIB_HOST_VECTOR_BASE_TYPES
    #define HEXLIB_HOST_VECTOR_BASE_TYPES
        #include "vectorBaseHost.inl"
    #endif

#endif

//================================================================
//
// Check that vector types have identical size and alignment on all systems.
//
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
