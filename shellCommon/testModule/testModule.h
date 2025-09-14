#pragma once

#include "data/gpuMatrix.h"
#include "gpuModuleKit.h"
#include "kits/gpuRgbFrameKit.h"
#include "kits/inputVideoNameKit.h"
#include "kits/processInspector.h"
#include "storage/smartPtr.h"
#include "gpuLayer/gpuLayerKits.h"

//================================================================
//
// TestModuleExternals
//
//================================================================

struct TestModuleExternals
{
    virtual ~TestModuleExternals() {}
};

//================================================================
//
// TestModule
//
//================================================================

struct TestModule
{
    virtual ~TestModule() {}

    ////

    virtual void setInputResolution(const Point<Space>& frameSize) =0;
    virtual void serialize(const ModuleSerializeKit& kit) =0;

    ////

    using InputMetadataSerializeKit = KitCombine<CfgSerializeKit, InputVideoNameKit>;
    virtual void inputMetadataSerialize(const InputMetadataSerializeKit& kit) =0;

    ////

    virtual void setExternals(UniquePtr<TestModuleExternals>&& value)
        {}

    ////

    using ReallocKit = GpuModuleReallocKit;
    virtual bool reallocValid() const =0;
    virtual void realloc(stdPars(ReallocKit)) =0;
    virtual void dealloc() =0;

    ////

    using ProcessKit = KitCombine<GpuModuleProcessKit, GpuRgbFrameKit, GpuMemoryAllocationKit>;
    virtual void inspectProcess(ProcessInspector& inspector) =0;
    virtual void process(stdPars(ProcessKit)) =0;
};

//================================================================
//
// TestModuleFactory
//
//================================================================

struct TestModuleFactory
{
    virtual const CharType* configName() const =0;
    virtual const CharType* displayName() const =0;

    virtual UniquePtr<TestModule> create() const =0;

    virtual UniquePtr<TestModuleExternals> createExternals() const
        {return nullptr;}
};
