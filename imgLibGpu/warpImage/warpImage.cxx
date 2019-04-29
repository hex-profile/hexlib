#define PIXEL int8
#include "warpImageFunc.inl"
#undef PIXEL

#define PIXEL int8_x2
#include "warpImageFunc.inl"
#undef PIXEL

#define PIXEL int8_x4
#include "warpImageFunc.inl"
#undef PIXEL

//----------------------------------------------------------------

#define PIXEL uint8
#include "warpImageFunc.inl"
#undef PIXEL

#define PIXEL uint8_x2
#include "warpImageFunc.inl"
#undef PIXEL

#define PIXEL uint8_x4
#include "warpImageFunc.inl"
#undef PIXEL

//----------------------------------------------------------------

#define PIXEL float16
#include "warpImageFunc.inl"
#undef PIXEL

#define PIXEL float16_x2
#include "warpImageFunc.inl"
#undef PIXEL

#define PIXEL float16_x4
#include "warpImageFunc.inl"
#undef PIXEL
