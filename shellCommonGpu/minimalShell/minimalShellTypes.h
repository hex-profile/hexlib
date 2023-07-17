#pragma once

#include "gpuModuleKit.h"
#include "gpuAppliedApi/gpuAppliedApi.h"
#include "storage/adapters/lambdaThunk.h"

namespace minimalShell {

//================================================================
//
// GpuExternalContext
//
//================================================================

struct GpuExternalContext
{
    GpuProperties gpuProperties;
    GpuContext gpuContext;
    GpuStream gpuStream;
};

//
// Optional external context kit.
//

KIT_CREATE(GpuExternalContextKit, const GpuExternalContext*, gpuExternalContext);

//================================================================
//
// EngineReallocKit
// EngineProcessKit
//
//================================================================

using EngineReallocKit = GpuModuleReallocKit;
using EngineProcessKit = GpuModuleProcessKit;

//================================================================
//
// EngineModuleRealloc
//
//================================================================

struct EngineModuleRealloc
{
    virtual bool reallocValid() const =0;
    virtual stdbool realloc(stdPars(EngineReallocKit)) =0;
};

//================================================================
//
// EngineModule
//
//================================================================

struct EngineModule : public EngineModuleRealloc
{
    virtual stdbool process(stdPars(EngineProcessKit)) =0;
};

//----------------------------------------------------------------

LAMBDA_THUNK3
(
    engineModuleThunk,
    EngineModule,
    bool reallocValid() const,
    lambda0(),
    stdbool realloc(stdPars(EngineReallocKit)),
    lambda1(stdPassThru),
    stdbool process(stdPars(EngineProcessKit)),
    lambda2(stdPassThru)
)

//================================================================
//
// Settings
//
//================================================================

struct Settings
{
    //
    // By default, the minimal shell creates and destroys gpu context and stream,
    // but it can also use the external ones.
    //

    virtual void setGpuContextMaintainer(bool value) =0;

    //
    // Serialize display params, etc.
    //

    virtual void setProfilerShellHotkeys(bool value) =0;
    virtual void setGpuShellHotkeys(bool value) =0;
    virtual void setDisplayParamsHotkeys(bool value) =0;
    virtual void setBmpConsoleHotkeys(bool value) =0;

    //
    // Image console saving helper.
    //

    virtual void setImageSavingActive(bool active) =0;
    virtual void setImageSavingDir(const CharType* dir) =0; // can be NULL
    virtual void setImageSavingLockstepCounter(uint32 counter) =0;
    virtual const CharType* getImageSavingDir() const =0;
};

//================================================================
//
// DesiredOutputSizeKit
//
//================================================================

using DesiredOutputSize = OptionalObject<Point<Space>>;

KIT_CREATE(DesiredOutputSizeKit, const DesiredOutputSize&, desiredOutputSize);

//----------------------------------------------------------------

}
