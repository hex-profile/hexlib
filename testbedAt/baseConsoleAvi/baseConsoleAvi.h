#pragma once

#include "baseInterfaces/baseImageConsole.h"
#include "storage/dynamicClass.h"
#include "userOutput/diagnosticKit.h"
#include "cpuFuncKit.h"

namespace baseConsoleAvi {

//================================================================
//
// Pixel
//
//================================================================

using Pixel = uint8_x4;

//================================================================
//
// Codec
//
//================================================================

using Codec = uint32;

Codec codecFromStr(const CharType* s);

//================================================================
//
// FPS
//
//================================================================

using FPS = int32;

//================================================================
//
// Kit
//
//================================================================

using Kit = KitCombine<DiagnosticKit, ProfilerKit, CpuFastAllocKit, DataProcessingKit>;

//================================================================
//
// BaseConsoleAvi
//
//================================================================

class BaseConsoleAvi
{

public:

    BaseConsoleAvi();
    ~BaseConsoleAvi();

    void saveImage(const Point<Space>& imageSize, BaseImageProvider& imageProvider, const FormatOutputAtom& desc, uint32 id, stdPars(Kit));
    void saveImage(const MatrixAP<const Pixel>& img, const FormatOutputAtom& desc, uint32 id, stdPars(Kit));

    void setOutputDir(const CharType* outputDir, stdPars(Kit));
    void setFps(const FPS& fps, stdPars(Kit));
    void setCodec(const Codec& codec, stdPars(Kit));
    void setMaxSegmentFrames(int32 maxSegmentFrames, stdPars(Kit));

private:

    DynamicClass<class BaseConsoleAviImpl> instance;

};

//================================================================
//
// BaseConsoleAviThunk
//
// Temporal thunk: adds necessary kit to BaseConsoleAvi
//
//================================================================

class BaseConsoleAviThunk : public BaseImageConsole, public BaseVideoOverlay
{

public:

    BaseConsoleAviThunk(BaseConsoleAvi& saver, BaseImageConsole& baseConsole, BaseVideoOverlay& baseOverlay, const Kit& kit)
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

    BaseConsoleAvi& saver;
    BaseImageConsole& baseConsole;
    BaseVideoOverlay& baseOverlay;
    Kit kit;

};

//----------------------------------------------------------------

}

using baseConsoleAvi::BaseConsoleAvi;
using baseConsoleAvi::BaseConsoleAviThunk;
