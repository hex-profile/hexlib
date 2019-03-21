#pragma once

#include "gpuProcessHeader.h"

namespace downsampleTwiceModelSpace {

//================================================================
//
// Type
//
//================================================================

using Type = float16;

//================================================================
//
// downsampleTwiceModel
//
//================================================================

bool downsampleTwiceModel(const GpuMatrix<const Type>& src, const GpuMatrix<Type>& dst, stdPars(GpuProcessKit));

//----------------------------------------------------------------

}

using downsampleTwiceModelSpace::downsampleTwiceModel;
