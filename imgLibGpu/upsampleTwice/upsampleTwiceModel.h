#pragma once

#include "gpuProcessHeader.h"

namespace upsampleTwiceModelSpace {

//================================================================
//
// upsampleTwiceModel
//
//================================================================

using Type = float16;

bool upsampleTwiceModel(const GpuMatrix<const Type>& src, const GpuMatrix<Type>& dst, stdPars(GpuProcessKit));

//----------------------------------------------------------------

}

using upsampleTwiceModelSpace::upsampleTwiceModel;
