#pragma once

#include "data/gpuLayeredMatrix.h"
#include "gpuProcessHeader.h"
#include "gpuSupport/gpuTool.h"
#include "visualizeComplexFilter/types.h"

namespace visualizeComplex {

//================================================================
//
// visualizeComplexFilterFunc
//
//================================================================

stdbool visualizeComplexFilterFunc
(
    const GpuMatrix<const ComplexFloat>& src, 
    const GpuMatrix<ComplexFloat>& dst,
    const Point<float32>& upsampleFactor,
    bool interpolation,
    bool dstModulation,
    const Point<float32>& dstModulationFreq,
    stdPars(GpuProcessKit)
);

//----------------------------------------------------------------

}
