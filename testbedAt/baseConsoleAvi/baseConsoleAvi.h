#pragma once

#include "allocation/mallocKit.h"
#include "errorLog/errorLogKit.h"
#include "interfaces/fileTools.h"
#include "kits/msgLogsKit.h"
#include "storage/dynamicClass.h"
#include "baseImageConsole/baseImageConsole.h"

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
//
//
//================================================================

using Kit = KitCombine<ErrorLogKit, MsgLogsKit, FileToolsKit, MallocKit, CpuFastAllocKit, DataProcessingKit>;

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

    stdbool addImage(const Matrix<const Pixel>& img, const ImgOutputHint& hint, stdNullPars)
    {
        bool ok1 = errorBlock(baseConsole.addImage(img, hint, stdPass));
        bool ok2 = errorBlock(saver.saveImage(img, hint.desc, hint.id, stdPass));
        require(ok1 && ok2);
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

    stdbool setImage(const Point<Space>& size, BaseImageProvider& imageProvider, const FormatOutputAtom& desc, uint32 id, bool textEnabled, stdNullPars)
    {
        bool ok1 = errorBlock(baseOverlay.setImage(size, imageProvider, desc, id, textEnabled, stdPass));
        bool ok2 = errorBlock(saver.saveImage(size, imageProvider, desc, id, stdPass));
        require(ok1 && ok2);
        returnTrue;
    }

    stdbool setImageFake(stdNullPars)
    {
        return baseOverlay.setImageFake(stdPassThru);
    }

    stdbool updateImage(stdNullPars) 
    {
        returnTrue;
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
