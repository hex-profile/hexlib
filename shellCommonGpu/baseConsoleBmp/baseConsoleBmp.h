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

    virtual stdbool saveImage(const MatrixAP<const Pixel>& img, const FormatOutputAtom& desc, uint32 id, stdPars(Kit)) =0;
    virtual stdbool saveImage(const Point<Space>& imageSize, BaseImageProvider& imageProvider, const FormatOutputAtom& desc, uint32 id, stdPars(Kit)) =0;

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

    stdbool addImage(const MatrixAP<const Pixel>& img, const ImgOutputHint& hint, bool dataProcessing, stdParsNull)
    {
        require(baseConsole.addImage(img, hint, dataProcessing, stdPass));
        require(saver.saveImage(img, hint.desc, hint.id, stdPass));
        returnTrue;
    }

    stdbool clear(stdParsNull)
    {
        require(baseConsole.clear(stdPassThru));
        returnTrue;
    }

    stdbool update(stdParsNull)
    {
        require(baseConsole.update(stdPassThru));
        returnTrue;
    }

public:

    stdbool overlayClear(stdParsNull)
    {
        return baseOverlay.overlayClear(stdPassThru);
    }

    stdbool overlaySet(const Point<Space>& size, bool dataProcessing, BaseImageProvider& imageProvider, const FormatOutputAtom& desc, uint32 id, bool textEnabled, stdParsNull)
    {
        require(baseOverlay.overlaySet(size, dataProcessing, imageProvider, desc, id, textEnabled, stdPass));
        require(saver.saveImage(size, imageProvider, desc, id, stdPass));
        returnTrue;
    }

    stdbool overlaySetFake(stdParsNull)
    {
        return baseOverlay.overlaySetFake(stdPassThru);
    }

    stdbool overlayUpdate(stdParsNull)
    {
        return baseOverlay.overlayUpdate(stdPassThru);
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
