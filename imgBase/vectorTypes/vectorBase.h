#pragma once

#include "extLib/types/vectorBase.h"
#include "numbers/int/intBase.h"

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
