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

    using ReallocKit = KitCombine<ModuleReallocKit, GpuAppExecKit>;
    using ProcessKit = KitCombine<ModuleProcessKit, GpuAppExecKit>;

    void serialize(const ModuleSerializeKit& kit);

public:

    void process(stdPars(ProcessKit));

private:

    BoolVar active{false};

    uint32 runCount = 0;
    uint32 writeCount = 0;

    RndgenState rndgenState = 0xB632009B;

    BoolVar fixedGroupSize{false};
    BoolVar fixedImageRatio{false};
    NumericVarStatic<Space, 1, 65536, 1000> reliabilityFactor;
    NumericVarStatic<Space, 1, 32, 16> fixedGroupWarps;

};
