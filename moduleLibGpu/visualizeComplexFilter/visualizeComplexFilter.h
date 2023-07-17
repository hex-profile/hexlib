#pragma once

#include "formatting/formatOutputAtom.h"
#include "gpuModuleHeader.h"
#include "pyramid/pyramidScale.h"
#include "visualizeComplexFilter/types.h"

namespace visualizeComplex {

//================================================================
//
// visualizeComplexFilter
//
//================================================================

stdbool visualizeComplexFilter
(
    const GpuMatrix<const ComplexFloat>& image,
    const Point<float32>& filterFreq, // In original resolution.
    float32 filterSubsamplingFactor,
    int displayedOrientation,
    int displayedScale,
    const Point<Space>& displayedSize,
    float32 displayMagnitude,
    float32 arrowFactor,
    const PyramidScale& pyramidScale,
    const FormatOutputAtom& name,
    stdPars(GpuModuleProcessKit)
);

//----------------------------------------------------------------

}

using visualizeComplex::visualizeComplexFilter;
