#pragma once

#include "vectorTypes/half/halfBase.h"
#include "numbers/float/floatBase.h"

//================================================================
//
// packFloat16
// unpackFloat16
//
//================================================================

float16 packFloat16(const float32& value);
float32 unpackFloat16(const float16& value);
