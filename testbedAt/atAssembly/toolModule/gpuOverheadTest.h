#pragma once

#include "gpuModuleHeader.h"
#include "rndgen/rndgenBase.h"
#include "cfgTools/numericVar.h"

//================================================================
//
// GpuOverheadTest
//
//================================================================

class GpuOverheadTest
{

public:

    KIT_COMBINE2(ReallocKit, ModuleReallocKit, GpuAppExecKit);
    KIT_COMBINE2(ProcessKit, ModuleProcessKit, GpuAppExecKit);

    void serialize(const ModuleSerializeKit& kit);

public:

    stdbool process(stdPars(ProcessKit));

private:

    BoolVarStatic<false> active;

    uint32 runCount = 0;
    uint32 writeCount = 0;

    RndgenState rndgenState = 0xB632009B;

    BoolVarStatic<false> fixedGroupSize;
    BoolVarStatic<false> fixedImageRatio;
    NumericVarStatic<Space, 1, 65536, 1000> reliabilityFactor;
    NumericVarStatic<Space, 1, 32, 16> fixedGroupWarps;

};
