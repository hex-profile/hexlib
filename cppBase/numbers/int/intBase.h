#pragma once

#include <stdint.h>
#include <limits.h>

#include "compileTools/compileTools.h"

//================================================================
//
// Builtin integer types
//
//================================================================

using uint8 = uint8_t;
using int8 = int8_t;
COMPILE_ASSERT(sizeof(uint8) * CHAR_BIT == 8);
COMPILE_ASSERT(sizeof(int8) * CHAR_BIT == 8);

using uint16 = uint16_t;
using int16 = int16_t;
COMPILE_ASSERT(sizeof(uint16) * CHAR_BIT == 16);
COMPILE_ASSERT(sizeof(int16) * CHAR_BIT == 16);

using uint32 = uint32_t;
using int32 = int32_t;
COMPILE_ASSERT(sizeof(uint32) * CHAR_BIT == 32);
COMPILE_ASSERT(sizeof(int32) * CHAR_BIT == 32);

using uint64 = uint64_t;
using int64 = int64_t;
COMPILE_ASSERT(sizeof(uint64) * CHAR_BIT == 64);
COMPILE_ASSERT(sizeof(int64) * CHAR_BIT == 64);

//----------------------------------------------------------------

#define BUILTIN_INT_FOREACH(action, extra) \
    action(bool, extra) \
    action(signed char, extra) \
    action(unsigned char, extra) \
    action(signed short, extra) \
    action(unsigned short, extra) \
    action(signed int, extra) \
    action(unsigned int, extra) \
    action(signed long int, extra) \
    action(unsigned long int, extra) \
    action(signed long long, extra) \
    action(unsigned long long, extra)
