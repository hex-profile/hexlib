#pragma once

#include "devSamplerInterface.h"
#include "data/matrix.h"
#include "storage/opaqueStruct.h"

//================================================================
//
// EmuSamplerData
//
//================================================================

using EmuSamplerData = OpaqueStruct<32>;

//================================================================
//
// EmuSamplerTex2D
//
//================================================================

#ifdef _MSC_VER
    #define EMU_SAMPLER_TEX_DECL __cdecl // MS bug workaround
#else
    #define EMU_SAMPLER_TEX_DECL
#endif

//----------------------------------------------------------------

typedef void EMU_SAMPLER_TEX_DECL EmuSamplerTex2D(const EmuSamplerData& data, float32 X, float32 Y, void* result);

//================================================================
//
// EmuSamplerTex1Dfetch
//
//================================================================

typedef void EmuSamplerTex1Dfetch(const EmuSamplerData& data, Space offset, void* result);

//================================================================
//
// EmuSamplerState
//
//================================================================

struct EmuSamplerState
{
    EmuSamplerTex2D* tex2D;
    EmuSamplerTex1Dfetch* tex1Dfetch;
    EmuSamplerData data;

    inline void reset()
    {
        tex1Dfetch = 0;
        tex2D = 0;
    }
};

//================================================================
//
// GpuSamplerLink
//
//================================================================

struct GpuSamplerLink
{
    EmuSamplerState* state;
};

//================================================================
//
// EmuSampler
//
//================================================================

template <DevSamplerType samplerType, DevSamplerReadMode readMode, int rank>
struct EmuSampler
{
    GpuSamplerLink link;

    using ResultType = typename DevSamplerReturnType<readMode, rank>::T;

    inline operator const GpuSamplerLink& () const
        {return link;}

    inline const GpuSamplerLink* operator &() const
        {return &link;}
};

//================================================================
//
// devDefineSampler
//
//================================================================

#undef devDefineSampler

#define devDefineSampler(sampler, samplerType, readMode, rank) \
    \
    static EmuSamplerState PREP_PASTE2(sampler, State) = {0, 0}; \
    static const EmuSampler<samplerType, readMode, rank> sampler = {&PREP_PASTE2(sampler, State)};

//----------------------------------------------------------------

#undef devSamplerParamType

#define devSamplerParamType(samplerType, readMode, rank) \
    const EmuSampler<samplerType, readMode, rank>&

//================================================================
//
// DevSamplerResult<EmuSampler>
//
//================================================================

template <DevSamplerType samplerType, DevSamplerReadMode readMode, int rank>
struct DevSamplerResult<EmuSampler<samplerType, readMode, rank>>
{
    using T = typename DevSamplerReturnType<readMode, rank>::T;
};

//================================================================
//
// emuTex2D
//
//================================================================

template <DevSamplerType samplerType, DevSamplerReadMode readMode, int rank>
inline typename DevSamplerReturnType<readMode, rank>::T emuTex2D(const EmuSampler<samplerType, readMode, rank>& sampler, float32 X, float32 Y)
{
    typename DevSamplerReturnType<readMode, rank>::T result;
    sampler.link.state->tex2D(sampler.link.state->data, X, Y, &result);
    return result;
}

//================================================================
//
// devTex2D
//
//================================================================

#undef devTex2D

#define devTex2D(sampler, X, Y) \
    emuTex2D(sampler, X, Y)

//================================================================
//
// emuTex1Dfetch
//
//================================================================

template <DevSamplerType samplerType, DevSamplerReadMode readMode, int rank>
inline typename DevSamplerReturnType<readMode, rank>::T emuTex1Dfetch(const EmuSampler<samplerType, readMode, rank>& sampler, Space offset)
{
    typename DevSamplerReturnType<readMode, rank>::T result;
    sampler.link.state->tex1Dfetch(sampler.link.state->data, offset, &result);
    return result;
}

//================================================================
//
// devTex1Dfetch
//
//================================================================

#undef devTex1Dfetch

#define devTex1Dfetch(sampler, offset) \
    emuTex1Dfetch(sampler, offset)
