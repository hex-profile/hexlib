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

#define TMP_MACRO(oldname, makeOldname, newname, element) \
    \
    typedef oldname newname; \
    \
    sysinline newname make_##newname(element x, element y) \
        {return makeOldname(x, y);}

TMP_MACRO(uchar2, make_uchar2, uint8_x2, uint8)
TMP_MACRO(char2, make_char2, int8_x2, int8)

TMP_MACRO(ushort2, make_ushort2, uint16_x2, uint16)
TMP_MACRO(short2, make_short2, int16_x2, int16)

TMP_MACRO(uint2, make_uint2, uint32_x2, uint32)
TMP_MACRO(int2, make_int2, int32_x2, int32)

TMP_MACRO(float2, make_float2, float32_x2, float32)

#undef TMP_MACRO

//================================================================
//
// Vector types (4-component)
//
//================================================================

#define TMP_MACRO(oldname, makeOldname, newname, element) \
    \
    typedef oldname newname; \
    \
    sysinline newname make_##newname(element x, element y, element z, element w) \
        {return makeOldname(x, y, z, w);}

TMP_MACRO(uchar4, make_uchar4, uint8_x4, uint8)
TMP_MACRO(char4, make_char4, int8_x4, int8)

TMP_MACRO(ushort4, make_ushort4, uint16_x4, uint16)
TMP_MACRO(short4, make_short4, int16_x4, int16)

TMP_MACRO(uint4, make_uint4, uint32_x4, uint32)
TMP_MACRO(int4, make_int4, int32_x4, int32)

TMP_MACRO(float4, make_float4, float32_x4, float32)

#undef TMP_MACRO

//================================================================
//
// 16-bit float (2-component)
//
//================================================================

__declspec(align(4))
struct float16_x2
{
    float16 x;
    float16 y;
};

sysinline float16_x2 make_float16_x2(float16 x, float16 y)
{
    float16_x2 tmp;
    tmp.x = x;
    tmp.y = y;
    return tmp;
}

//================================================================
//
// 16-bit float (4-component)
//
//================================================================

__declspec(align(8))
struct float16_x4
{
    float16 x;
    float16 y;
    float16 z;
    float16 w;
};

sysinline float16_x4 make_float16_x4(float16 x, float16 y, float16 z, float16 w)
{
    float16_x4 tmp;
    tmp.x = x;
    tmp.y = y;
    tmp.z = z;
    tmp.w = w;
    return tmp;
}

//================================================================
//
// bool (2-component)
//
//================================================================

#define TMP_MACRO_X2(vector, scalar, byteAlignment) \
    \
    __declspec(align(byteAlignment)) \
    struct vector \
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

#undef TMP_MACRO_X2

//================================================================
//
// bool (4-component)
//
//================================================================

#define TMP_MACRO_X4(vector, scalar, byteAlignment) \
    \
    __declspec(align(byteAlignment)) \
    struct vector \
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

#undef TMP_MACRO_X4

//----------------------------------------------------------------

#endif // _F9C2F7343198A455
