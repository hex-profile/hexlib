#pragma once

#include "gpuModuleKit.h"
#include "kits/alternativeVersionKit.h"
#include "kits/displayParamsKit.h"
#include "kits/processInspector.h"
#include "storage/smartPtr.h"
#include "data/gpuMatrix.h"
#include "vectorTypes/vectorBase.h"

//================================================================
//
// GpuRgbFrameKit
//
//================================================================

KIT_CREATE1(GpuRgbFrameKit, GpuMatrix<const uint8_x4>, gpuRgbFrame);

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
    virtual void serialize(const ModuleSerializeKit& kit) =0;

    virtual void setFrameSize(const Point<Space>& frameSize) =0;

    virtual bool reallocValid() const =0;
    virtual bool realloc(stdPars(AtEngineReallocKit)) =0;

    virtual void inspectProcess(ProcessInspector& inspector) =0;
    virtual bool process(stdPars(AtEngineProcessKit)) =0;

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
