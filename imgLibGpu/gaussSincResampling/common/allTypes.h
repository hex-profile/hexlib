#pragma once

//================================================================
//
// FOREACH_TYPE
//
//================================================================

#define FOREACH_TYPE(action) \
    \
    action(int8, int8, int8, 1) \
    action(uint8, uint8, uint8, 1) \
    action(float16, float16, float16, 1) \
    action(float32, float32, float32, 1) \
    \
    action(int8_x2, int8_x2, int8_x2, 2) \
    action(uint8_x2, uint8_x2, uint8_x2, 2) \
    action(float16_x2, float16_x2, float16_x2, 2) \
    action(float32_x2, float32_x2, float32_x2, 2) \
    \
    action(uint8_x4, uint8_x4, uint8_x4, 4)
