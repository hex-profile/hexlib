#pragma once

#include <stdint.h>

//================================================================
//
// float16
//
//================================================================

struct float16
{
    uint16_t data;
};

static_assert(sizeof(float16) == 2, "");
static_assert(alignof(float16) == 2, "");
