#pragma once

#include "gpuDevice/devSampler/devSampler.h"
#include "numbers/float/floatBase.h"
#include "point/pointBase.h"

//================================================================
//
// tex2D
//
//================================================================

template <typename SamplerType>
sysinline auto tex2D(SamplerType srcSampler, const Point<float32>& pos)
{
    return devTex2D(srcSampler, pos.X, pos.Y);
}

//================================================================
//
// computeTexstep
//
//================================================================

template <typename Source>
sysinline Point<float32> computeTexstep(const Source& source)
{
    return 1.f / convertFloat32(clampMin(GetSize<Source>::func(source), 1));
}
