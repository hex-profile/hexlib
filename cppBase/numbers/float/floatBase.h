#pragma once

#include "extLib/types/floatBase.h"

//================================================================
//
// Builtin floating-point types
//
//================================================================

#define BUILTIN_FLOAT_FOREACH_NORMAL(action, extra) \
    action(float32, extra)

#define BUILTIN_FLOAT_FOREACH_LARGE(action, extra) \
    action(float64, extra)

#define BUILTIN_FLOAT_FOREACH(action, extra) \
    BUILTIN_FLOAT_FOREACH_NORMAL(action, extra) \
    BUILTIN_FLOAT_FOREACH_LARGE(action, extra)
