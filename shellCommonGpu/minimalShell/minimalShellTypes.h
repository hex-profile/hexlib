#pragma once

#include "gpuModuleKit.h"

namespace minimalShell {

//================================================================
//
// EngineReallocKit
// EngineProcessKit
//
//================================================================

using EngineReallocKit = KitCombine<GpuModuleReallocKit, GpuBlockAllocatorKit>;
using EngineProcessKit = KitCombine<GpuModuleProcessKit, GpuBlockAllocatorKit>;

//================================================================
//
// EngineModule
//
//================================================================

struct EngineModule
{
    virtual bool reallocValid() const =0;
    virtual stdbool realloc(stdPars(EngineReallocKit)) =0;
    virtual stdbool process(stdPars(EngineProcessKit)) =0;
};

//================================================================
//
// Settings
//
//================================================================

struct Settings
{
    virtual void setImageSavingActive(bool active) =0;
    virtual void setImageSavingDir(const CharType* dir) =0; // can be NULL
    virtual void setImageSavingLockstepCounter(uint32 counter) =0;
    virtual const CharType* getImageSavingDir() const =0;
};

//----------------------------------------------------------------

}
