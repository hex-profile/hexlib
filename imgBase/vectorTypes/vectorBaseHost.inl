#ifndef _F9C2F7343198A455
#define _F9C2F7343198A455

#include "compileTools/compileTools.h"
#include "numbers/float/floatBase.h"
#include "numbers/int/intBase.h"
#include "vectorTypes/half/halfBase.h"

//================================================================
//
// Vector types (2-component)
//
//================================================================

#define TMP_MACRO_X2(vector, scalar, byteAlignment) \
    \
    struct alignas(byteAlignment) vector \
    { \
        scalar x; \
        scalar y; \
    }; \
    \
    sysinline vector make_##vector(scalar x, scalar y) \
    { \
        vector tmp; \
        tmp.x = x; \
        tmp.y = y; \
        return tmp; \
    }

TMP_MACRO_X2(bool_x2, bool, 2)

TMP_MACRO_X2(int8_x2, int8, 2)
TMP_MACRO_X2(uint8_x2, uint8, 2)

TMP_MACRO_X2(int16_x2, int16, 4)
TMP_MACRO_X2(uint16_x2, uint16, 4)

TMP_MACRO_X2(int32_x2, int32, 8)
TMP_MACRO_X2(uint32_x2, uint32, 8)

TMP_MACRO_X2(float16_x2, float16, 4)
TMP_MACRO_X2(float32_x2, float32, 8)

#undef TMP_MACRO_X2

//================================================================
//
// Vector types (4-component)
//
//================================================================

#define TMP_MACRO_X4(vector, scalar, byteAlignment) \
    \
    struct alignas(byteAlignment) vector \
    { \
        scalar x; \
        scalar y; \
        scalar z; \
        scalar w; \
    }; \
    \
    sysinline vector make_##vector(scalar x, scalar y, scalar z, scalar w) \
    { \
        vector tmp; \
        tmp.x = x; \
        tmp.y = y; \
        tmp.z = z; \
        tmp.w = w; \
        return tmp; \
    }

TMP_MACRO_X4(bool_x4, bool, 4)

TMP_MACRO_X4(int8_x4, int8, 4)
TMP_MACRO_X4(uint8_x4, uint8, 4)

TMP_MACRO_X4(int16_x4, int16, 8)
TMP_MACRO_X4(uint16_x4, uint16, 8)

TMP_MACRO_X4(int32_x4, int32, 16)
TMP_MACRO_X4(uint32_x4, uint32, 16)

TMP_MACRO_X4(float16_x4, float16, 8)
TMP_MACRO_X4(float32_x4, float32, 16)

#undef TMP_MACRO_X4

//----------------------------------------------------------------

#endif // _F9C2F7343198A455
