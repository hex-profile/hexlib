#pragma once

#include "data/gpuMatrix.h"
#include "gpuModuleKit.h"
#include "kits/alternativeVersionKit.h"
#include "kits/displayParamsKit.h"
#include "kits/gpuRgbFrameKit.h"
#include "kits/inputVideoNameKit.h"
#include "kits/processInspector.h"
#include "storage/smartPtr.h"
#include "vectorTypes/vectorBase.h"

//================================================================
//
// AtEngineReallocKit
// AtEngineProcessKit
//
//================================================================

using AtEngineReallocKit = GpuModuleReallocKit;
using AtEngineProcessKit = KitCombine<GpuModuleProcessKit, GpuRgbFrameKit>;

//================================================================
//
// InputMetadataSerializeKit
//
//================================================================

using InputMetadataSerializeKit = KitCombine<CfgSerializeKit, InputVideoNameKit>;

//================================================================
//
// AtEngine
//
//================================================================

struct AtEngine
{
    virtual CharType* getName() const =0;

    virtual void setInputResolution(const Point<Space>& frameSize) =0;
    virtual void serialize(const ModuleSerializeKit& kit) =0;
    virtual void inputMetadataSerialize(const InputMetadataSerializeKit& kit) =0;

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
