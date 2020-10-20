#pragma once

#include "baseImageConsole/baseImageConsole.h"
#include "interfaces/fileToolsKit.h"
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

using Kit = KitCombine<DiagnosticKit, ProfilerKit, CpuFastAllocKit, DataProcessingKit, FileToolsKit>;

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

public:

    static UniquePtr<BaseConsoleBmp> create();
    virtual ~BaseConsoleBmp() {}

public:

    virtual void setDefaultActive(bool active) =0;
    virtual void setDefaultDir(const CharType* dir) =0; // can be NULL

public:

    virtual void serialize(const CfgSerializeKit& kit) =0;

public:

    virtual bool active() const =0;
    virtual const CharType* getOutputDir() const =0;
    virtual void setLockstepCounter(Counter counter) =0;

public:

    virtual stdbool saveImage(const Matrix<const Pixel>& img, const FormatOutputAtom& desc, uint32 id, stdPars(Kit)) =0;
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

    stdbool addImage(const Matrix<const Pixel>& img, const ImgOutputHint& hint, stdNullPars)
    {
        require(baseConsole.addImage(img, hint, stdPass));
        require(saver.saveImage(img, hint.desc, hint.id, stdPass));
        returnTrue;
    }

    stdbool clear(stdNullPars)
    {
        require(baseConsole.clear(stdPassThru));
        returnTrue;
    }

    stdbool update(stdNullPars)
    {
        require(baseConsole.update(stdPassThru));
        returnTrue;
    }

public:

    stdbool setImage(const Point<Space>& size, bool dataProcessing, BaseImageProvider& imageProvider, const FormatOutputAtom& desc, uint32 id, bool textEnabled, stdNullPars)
    {
        require(baseOverlay.setImage(size, dataProcessing, imageProvider, desc, id, textEnabled, stdPass));
        require(saver.saveImage(size, imageProvider, desc, id, stdPass));
        returnTrue;
    }

    stdbool setImageFake(stdNullPars)
    {
        return baseOverlay.setImageFake(stdPassThru);
    }

    stdbool updateImage(stdNullPars) 
    {
        return baseOverlay.updateImage(stdPassThru);
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
