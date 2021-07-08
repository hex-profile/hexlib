#pragma once

#include "cfgTools/boolSwitch.h"
#include "cfgTools/numericVar.h"
#include "kits/moduleKit.h"
#include "gpuShell/gpuShellKits.h"
#include "interfaces/threadManagerKit.h"
#include "gpuLayer/gpuLayer.h"
#include "allocation/mallocKit.h"
#include "gpuLayer/gpuLayerImpl.h"

namespace gpuShell {

//================================================================
//
// GpuContextHelper
//
// A convenience class.
//
//================================================================

class GpuContextHelper
{

public:

    bool serialize(const CfgSerializeKit& kit)
        {return gpuDeviceIndex.serialize(kit, STR("GPU Device Index"));}

public:

    KIT_COMBINE3(InitKit, GpuInitKit, ErrorLogKit, MsgLogKit);
    stdbool createContext(GpuProperties& gpuProperties, GpuContextOwner& gpuContext, stdPars(InitKit));

private:

    NumericVar<int32> gpuDeviceIndex{typeMin<int32>(), typeMax<int32>(), 0};

};

//================================================================
//
// GpuShellTarget
//
//================================================================

struct GpuShellTarget
{
    virtual stdbool exec(stdPars(GpuShellKit)) =0;
};

//================================================================
//
// GpuShellTargetThunk
//
//================================================================

template <typename Lambda>
class GpuShellTargetThunk : public GpuShellTarget
{

public:

    virtual stdbool exec(stdPars(GpuShellKit))
        {return lambda(stdPassThru);}

    GpuShellTargetThunk(const Lambda& lambda)
        : lambda(lambda) {}

private:

    const Lambda& lambda;

};

//================================================================
//
// ExecGlobalToolkit
// ExecCyclicToolkit
//
//================================================================

KIT_CREATE2(GpuApiImplKit, GpuInitApiImpl&, gpuInitApi, GpuExecApiImpl&, gpuExecApi);

KIT_COMBINE5(ExecCyclicToolkit, GpuApiImplKit, GpuPropertiesKit, GpuCurrentContextKit, GpuCurrentStreamKit, LocalLogKit);

//================================================================
//
// GpuShellImpl
//
//================================================================

class GpuShellImpl
{

public:

    void serialize(const CfgSerializeKit& kit);

public:

    stdbool execCyclicShell(GpuShellTarget& app, stdPars(ExecCyclicToolkit));

public:

    template <typename Lambda>
    stdbool execCyclicShellLambda(const Lambda& lambda, stdPars(ExecCyclicToolkit))
    {
        GpuShellTargetThunk<Lambda> target{lambda};
        return execCyclicShell(target, stdPassThru);
    }

private:

    StandardSignal gpuEnqueueModeCycle;
    NumericVarStatic<int, 0, 2, 0> gpuEnqueueModeVar;

    BoolSwitch<false> gpuCoverageModeVar;
    NumericVarStatic<int32, 4, 16384, 1024> coverageQueueCapacity;

};

//----------------------------------------------------------------

}

using gpuShell::GpuContextHelper;
