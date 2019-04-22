#pragma once

#include "data/gpuMatrix.h"
#include "gpuModuleKit.h"
#include "kits/alternativeVersionKit.h"
#include "kits/displayParamsKit.h"
#include "kits/gpuRgbFrameKit.h"
#include "kits/processInspector.h"
#include "storage/smartPtr.h"
#include "vectorTypes/vectorBase.h"

//================================================================
//
// AtEngineReallocKit
//
//================================================================

KIT_COMBINE2(AtEngineReallocKit, GpuModuleReallocKit, GpuBlockAllocatorKit);
KIT_COMBINE4(AtEngineProcessKit, GpuModuleProcessKit, GpuRgbFrameKit, AlternativeVersionKit, DisplayParamsKit);

//================================================================
//
// AtEngine
//
//================================================================

struct AtEngine
{
    virtual CharType* getName() const =0;

    virtual void serialize(const ModuleSerializeKit& kit) =0;
    virtual void setInputResolution(const Point<Space>& frameSize) =0;
    virtual void setInputMetadata(const CfgSerializeKit& kit) =0;

    virtual bool reallocValid() const =0;
    virtual stdbool realloc(stdPars(AtEngineReallocKit)) =0;

    virtual void inspectProcess(ProcessInspector& inspector) =0;
    virtual stdbool process(stdPars(AtEngineProcessKit)) =0;

    virtual ~AtEngine() {}
};

//================================================================
//
// AtEngineFactory
//
//================================================================

struct AtEngineFactory
{
    virtual UniquePtr<AtEngine> create() const =0;
};
