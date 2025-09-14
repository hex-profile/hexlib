#pragma once

#include "cfgTools/boolSwitch.h"
#include "cfgTools/numericVar.h"
#include "kits/moduleKit.h"
#include "gpuShell/gpuShellKits.h"
#include "gpuLayer/gpuLayer.h"
#include "allocation/mallocKit.h"
#include "gpuLayer/gpuLayerImpl.h"
#include "cfgTools/multiSwitch.h"
#include "storage/adapters/callable.h"

namespace gpuShell {

//================================================================
//
// GpuContextHelper
//
// Stateless, only config variables.
//
//================================================================

class GpuContextHelper
{

public:

    bool serialize(const CfgSerializeKit& kit);

public:

    using InitKit = KitCombine<GpuInitKit, ErrorLogKit, MsgLogKit>;
    void createContext(GpuProperties& gpuProperties, GpuContextOwner& gpuContext, stdPars(InitKit));

private:

    NumericVar<int32> gpuDeviceIndex{typeMin<int32>(), typeMax<int32>(), 0};

    MultiSwitch<GpuScheduling, GpuScheduling::COUNT, GpuScheduling::Spin> gpuScheduling;

};

//================================================================
//
// GpuShellTarget
//
//================================================================

using GpuShellTarget = Callable<void (stdPars(GpuShellKit))>;

//================================================================
//
// ExecCyclicToolkit
//
//================================================================

KIT_CREATE2(GpuApiImplKit, GpuInitApiImpl&, gpuInitApi, GpuExecApiImpl&, gpuExecApi);

using ExecCyclicToolkit = KitCombine<GpuApiImplKit, GpuPropertiesKit, GpuCurrentContextKit, GpuCurrentStreamKit, LocalLogKit>;

//================================================================
//
// GpuShellImpl
//
// Stateless, only config vars.
//
//================================================================

class GpuShellImpl
{

public:

    void serialize(const CfgSerializeKit& kit, bool hotkeys);

public:

    void execCyclicShell(GpuShellTarget& app, stdPars(ExecCyclicToolkit));

private:

    StandardSignal gpuEnqueueModeCycle;
    NumericVarStatic<int, 0, 2, 0> gpuEnqueueModeVar;

    BoolSwitch gpuCoverageModeVar{false};
    NumericVarStatic<int32, 4, 16384, 1024> coverageQueueCapacity;

};

//----------------------------------------------------------------

}

using gpuShell::GpuContextHelper;
