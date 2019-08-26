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

sysinline float16 packFloat16(float32 value)
{
    float16 tmp;
    tmp.data = __float2half_rn(value);
    return tmp;
}

//----------------------------------------------------------------

sysinline float32 unpackFloat16(const float16& value)
{
    return __half2float(value.data);
}

//----------------------------------------------------------------

#endif
