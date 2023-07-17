#pragma once

#include "numbers/float16/float16Base.h"
#include "numbers/float/floatBase.h"

#if defined(__CUDA_ARCH__)

#include <cuda_fp16.h>

//================================================================
//
// packFloat16
// unpackFloat16
//
//================================================================

COMPILE_ASSERT(sizeof(__half) == sizeof(float16));
COMPILE_ASSERT(alignof(__half) == alignof(float16));

//----------------------------------------------------------------

sysinline float16 packFloat16(float32 value)
{
    float16 tmp;
    * (__half*) &tmp.data = __float2half(value);
    return tmp;
}

//----------------------------------------------------------------

sysinline float32 unpackFloat16(const float16& value)
{
    return __half2float(* (const __half*) &value.data);
}

//----------------------------------------------------------------

#endif
