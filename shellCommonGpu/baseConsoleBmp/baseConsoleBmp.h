#pragma once

#include "baseInterfaces/baseImageConsole.h"
#include "storage/dynamicClass.h"
#include "userOutput/diagnosticKit.h"
#include "cpuFuncKit.h"
#include "cfg/cfgInterface.h"
#include "storage/smartPtr.h"

namespace baseConsoleBmp {

//================================================================
//
// Pixel
//
//================================================================

using Pixel = uint8_x4;

//================================================================
//
// Kit
//
//================================================================

using Kit = KitCombine<DiagnosticKit, ProfilerKit, CpuFastAllocKit, DataProcessingKit>;

//================================================================
//
// Counter
//
//================================================================

using Counter = uint32;

//================================================================
//
// BaseConsoleBmp
//
//================================================================

struct BaseConsoleBmp
{
    static UniquePtr<BaseConsoleBmp> create();
    virtual ~BaseConsoleBmp() {}

    ////

    virtual void setActive(bool active) =0;
    virtual void setDir(const CharType* dir) =0; // can be NULL

    ////

    virtual void serialize(const CfgSerializeKit& kit, bool hotkeys) =0;

    ////

    virtual void clearState() =0;

    ////

    virtual bool active() const =0;
    virtual const CharType* getOutputDir() const =0;
    virtual void setLockstepCounter(Counter counter) =0;

    ////

    virtual void saveImage(const MatrixAP<const Pixel>& img, const FormatOutputAtom& desc, uint32 id, stdPars(Kit)) =0;
    virtual void saveImage(const Point<Space>& imageSize, BaseImageProvider& imageProvider, const FormatOutputAtom& desc, uint32 id, stdPars(Kit)) =0;

};

//================================================================
//
// BaseConsoleBmpThunk
//
// Temporal thunk: adds necessary kit to BaseConsoleBmp
//
//================================================================

class BaseConsoleBmpThunk : public BaseImageConsole, public BaseVideoOverlay
{

public:

    BaseConsoleBmpThunk(BaseConsoleBmp& saver, BaseImageConsole& baseConsole, BaseVideoOverlay& baseOverlay, const Kit& kit)
        : saver(saver), baseConsole(baseConsole), baseOverlay(baseOverlay), kit(kit) {}

public:

    void addImage(const MatrixAP<const Pixel>& img, const ImgOutputHint& hint, bool dataProcessing, stdParsNull)
    {
        baseConsole.addImage(img, hint, dataProcessing, stdPass);
        saver.saveImage(img, hint.desc, hint.id, stdPass);
    }

    void clear(stdParsNull)
    {
        baseConsole.clear(stdPassThru);
    }

    void update(stdParsNull)
    {
        baseConsole.update(stdPassThru);
    }

public:

    void overlayClear(stdParsNull)
    {
        baseOverlay.overlayClear(stdPassThru);
    }

    void overlaySet(const Point<Space>& size, bool dataProcessing, BaseImageProvider& imageProvider, const FormatOutputAtom& desc, uint32 id, bool textEnabled, stdParsNull)
    {
        baseOverlay.overlaySet(size, dataProcessing, imageProvider, desc, id, textEnabled, stdPass);
        saver.saveImage(size, imageProvider, desc, id, stdPass);
    }

    void overlaySetFake(stdParsNull)
    {
        baseOverlay.overlaySetFake(stdPassThru);
    }

    void overlayUpdate(stdParsNull)
    {
        baseOverlay.overlayUpdate(stdPassThru);
    }

private:

    BaseConsoleBmp& saver;
    BaseImageConsole& baseConsole;
    BaseVideoOverlay& baseOverlay;
    Kit kit;

};

//----------------------------------------------------------------

}

using baseConsoleBmp::BaseConsoleBmp;
using baseConsoleBmp::BaseConsoleBmpThunk;
