#pragma once

#include "gpuModuleHeader.h"
#include "kits/displayParamsKit.h"
#include "pyramid/gpuPyramid.h"
#include "numbers/float16/float16Base.h"

namespace phaseCorrTest {

//================================================================
//
// ProcessKit
//
//================================================================

using ProcessKit = GpuModuleProcessKit;

//================================================================
//
// Process
//
//================================================================

using MemFloat = float16;

using InputPixel = MemFloat;

//----------------------------------------------------------------

struct Process
{
    const GpuMatrix<const InputPixel>& oldImage;
    const GpuMatrix<const InputPixel>& newImage;
    const Point<float32>& baseVector;
    const Point<float32>& userPoint;
};

//================================================================
//
// PhaseCorrTest
//
//================================================================

class PhaseCorrTest
{

public:

    PhaseCorrTest();
    ~PhaseCorrTest();

public:

    void serialize(const ModuleSerializeKit& kit);
    bool isActive() const;
    stdbool process(const Process& o, stdPars(ProcessKit));

private:

    DynamicClass<class PhaseCorrTestImpl> instance;

};

//----------------------------------------------------------------

}
