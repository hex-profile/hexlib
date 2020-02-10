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
// A convenience class
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

    NumericVarStatic<int32, 0, 256, 0> gpuDeviceIndex;

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

private:

    StandardSignal gpuEnqueueModeCycle;
    NumericVarStatic<int, 0, 2, 0> gpuEnqueueModeVar;

    BoolSwitch<false> gpuCoverageModeVar;
    NumericVarStatic<int32, 4, 16384, 1024> coverageQueueCapacity;

};

//----------------------------------------------------------------

}

using gpuShell::GpuContextHelper;
