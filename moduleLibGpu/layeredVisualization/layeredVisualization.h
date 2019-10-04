#pragma once

#include "data/gpuLayeredMatrix.h"
#include "gpuModuleHeader.h"
#include "gpuModuleKit.h"
#include "gpuSupport/gpuTool.h"
#include "imageConsole/gpuImageConsole.h"

namespace layeredVisualization {

//================================================================
//
// visualizeLayeredVector
//
//================================================================

template <typename VectorType, typename PresenceType>
stdbool visualizeLayeredVector
(
    const GpuLayeredMatrix<const VectorType>& vectorValue,
    const GpuLayeredMatrix<const PresenceType>& vectorPresence,
    bool independentPresenceMode,
    float32 maxVector,
    const Point<float32>& upsampleFactor,
    const Point<Space>& upsampleSize,
    bool upsampleInterpolation,
    const ImgOutputHint& hint,
    stdPars(GpuModuleProcessKit)
);

//================================================================
//
// visualizeLayeredVector
//
//================================================================

template <typename VectorType>
stdbool visualizeLayeredVector
(
    const GpuLayeredMatrix<const VectorType>& vectorValue,
    float32 maxVector,
    const Point<float32>& upsampleFactor,
    const Point<Space>& upsampleSize,
    bool upsampleInterpolation,
    const ImgOutputHint& hint,
    stdPars(GpuModuleProcessKit)
);

//----------------------------------------------------------------

}

using layeredVisualization::visualizeLayeredVector;
