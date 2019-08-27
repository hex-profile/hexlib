//================================================================
//
// int8
//
//================================================================

#define PIXELS (int8, int8)
#include "warpImageFunc.inl"
#undef PIXELS

#define PIXELS (int8_x2, int8_x2)
#include "warpImageFunc.inl"
#undef PIXELS

#define PIXELS (int8_x4, int8_x4)
#include "warpImageFunc.inl"
#undef PIXELS

//----------------------------------------------------------------

#define PIXELS (float16, int8)
#include "warpImageFunc.inl"
#undef PIXELS

#define PIXELS (float16_x2, int8_x2)
#include "warpImageFunc.inl"
#undef PIXELS

#define PIXELS (float16_x4, int8_x4)
#include "warpImageFunc.inl"
#undef PIXELS

//================================================================
//
// uint8
//
//================================================================

#define PIXELS (uint8, uint8)
#include "warpImageFunc.inl"
#undef PIXELS

#define PIXELS (uint8_x2, uint8_x2)
#include "warpImageFunc.inl"
#undef PIXELS

#define PIXELS (uint8_x4, uint8_x4)
#include "warpImageFunc.inl"
#undef PIXELS

//----------------------------------------------------------------

#define PIXELS (float16, uint8)
#include "warpImageFunc.inl"
#undef PIXELS

#define PIXELS (float16_x2, uint8_x2)
#include "warpImageFunc.inl"
#undef PIXELS

#define PIXELS (float16_x4, uint8_x4)
#include "warpImageFunc.inl"
#undef PIXELS

//================================================================
//
// float16
//
//================================================================

#define PIXELS (float16, float16)
#include "warpImageFunc.inl"
#undef PIXELS

#define PIXELS (float16_x2, float16_x2)
#include "warpImageFunc.inl"
#undef PIXELS

#define PIXELS (float16_x4, float16_x4)
#include "warpImageFunc.inl"
#undef PIXELS
