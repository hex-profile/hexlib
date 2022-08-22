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

    stdbool saveImage(const Point<Space>& imageSize, BaseImageProvider& imageProvider, const FormatOutputAtom& desc, uint32 id, stdPars(Kit));
    stdbool saveImage(const Matrix<const Pixel>& img, const FormatOutputAtom& desc, uint32 id, stdPars(Kit));

    stdbool setOutputDir(const CharType* outputDir, stdPars(Kit));
    stdbool setFps(const FPS& fps, stdPars(Kit));
    stdbool setCodec(const Codec& codec, stdPars(Kit));
    stdbool setMaxSegmentFrames(int32 maxSegmentFrames, stdPars(Kit));

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

    stdbool addImage(const Matrix<const Pixel>& img, const ImgOutputHint& hint, bool dataProcessing, stdNullPars)
    {
        require(baseConsole.addImage(img, hint, dataProcessing, stdPass));
        require(saver.saveImage(img, hint.desc, hint.id, stdPass));
        returnTrue;
    }

    stdbool clear(stdNullPars)
    {
        return baseConsole.clear(stdPassThru);
    }

    stdbool update(stdNullPars)
    {
        return baseConsole.update(stdPassThru);
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

    BaseConsoleAvi& saver;
    BaseImageConsole& baseConsole;
    BaseVideoOverlay& baseOverlay;
    Kit kit;

};

//----------------------------------------------------------------

}

using baseConsoleAvi::BaseConsoleAvi;
using baseConsoleAvi::BaseConsoleAviThunk;
