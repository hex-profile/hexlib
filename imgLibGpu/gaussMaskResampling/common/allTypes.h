#pragma once

#define FOREACH_TYPE(action) \
    \
    action(uint8, uint8, uint8, 1) \
    action(float16, float16, float16, 1) \
    action(float32, float32, float32, 1) \
    action(float16_x4, float16_x4, float16_x4, 4)
