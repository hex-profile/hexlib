#pragma once

#include "gpuDevice/devSampler/devSampler.h"
#include "vectorTypes/vectorType.h"
#include "imageRead/positionTools.h"

#if DEVCODE

//================================================================
//
// tex2D
//
//================================================================

template <typename SamplerType>
sysinline typename DevSamplerResult<SamplerType>::T tex2D(SamplerType srcSampler, const Point<float32>& pos)
{
    return devTex2D(srcSampler, pos.X, pos.Y);
}

//----------------------------------------------------------------

#endif

//================================================================
//
// computeTexstep
//
//================================================================

template <typename Source>
sysinline Point<float32> computeTexstep(const Source& source)
    {return 1.f / convertFloat32(clampMin(GetSize<Source>::func(source), 1));}
