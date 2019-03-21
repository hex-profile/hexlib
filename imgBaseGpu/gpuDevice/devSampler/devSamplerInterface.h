#pragma once

#include "vectorTypes/vectorBase.h"

//================================================================
//
// DevSamplerType
//
//================================================================

enum DevSamplerType {DevSampler1D, DevSampler2D};

//================================================================
//
// DevSamplerReadMode
//
// The type of the result?
//
//================================================================

enum DevSamplerReadMode {DevSamplerFloat, DevSamplerInt, DevSamplerUint};

//================================================================
//
// DevSamplerReturnType
//
// The type returned by sampler.
//
//================================================================

template <DevSamplerReadMode readMode, int rank>
struct DevSamplerReturnType;

////

#define TMP_MACRO(readMode, rank, Type) \
    template <> struct DevSamplerReturnType<readMode, rank> {using T = Type;};

////

TMP_MACRO(DevSamplerFloat, 1, float32)
TMP_MACRO(DevSamplerFloat, 2, float32_x2)
TMP_MACRO(DevSamplerFloat, 4, float32_x4)

TMP_MACRO(DevSamplerInt, 1, int32)
TMP_MACRO(DevSamplerInt, 2, int32_x2)
TMP_MACRO(DevSamplerInt, 4, int32_x4)

TMP_MACRO(DevSamplerUint, 1, uint32)
TMP_MACRO(DevSamplerUint, 2, uint32_x2)
TMP_MACRO(DevSamplerUint, 4, uint32_x4)

////

#undef TMP_MACRO

//================================================================
//
// DevSamplerResult
//
//================================================================

template <typename Sampler>
struct DevSamplerResult
{
    using T = void;
};

//================================================================
//
// devDefineSampler
// devSamplerParamType
//
//================================================================

#define devDefineSampler(sampler, samplerType, readMode, rank)

#define devSamplerParamType(samplerType, readMode, rank)

//================================================================
//
// devTex2D
//
// Takes float X, Y and returns the value of type DevSamplerReturnType<>.
// If the return type is float, interpolation is possible.
//
//================================================================

#define devTex2D(sampler, X, Y)

//================================================================
//
// devTex1Dfetch
//
// Takes integer X and returns the value of type DevSamplerReturnType<>.
// Interpolation is NOT possible. Conversion to normalized float is possible.
//
//================================================================

#define devTex1Dfetch(sampler, X)
