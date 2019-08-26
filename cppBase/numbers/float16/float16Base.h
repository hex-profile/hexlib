#pragma once

#include <cstdint>

#include "compileTools/compileTools.h"

//================================================================
//
// 16-bit float support
//
//================================================================

#ifndef HEXLIB_FLOAT16
#define HEXLIB_FLOAT16

struct float16
{
    uint16_t data;
};

COMPILE_ASSERT(sizeof(float16) == 2 && alignof(float16) == 2);

#endif
