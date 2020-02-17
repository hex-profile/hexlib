#pragma once

#include "atInterface/atInterface.h"
#include "errorLog/errorLogKit.h"
#include "kits/msgLogsKit.h"
#include "interfaces/fileTools.h"
#include "allocation/mallocKit.h"
#include "storage/dynamicClass.h"

namespace outImgAvi {

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
// OutImgAvi
//
//================================================================

class OutImgAvi
{

public:

    KIT_COMBINE4(Kit, ErrorLogKit, MsgLogsKit, FileToolsKit, MallocKit);

public:

    OutImgAvi();
    ~OutImgAvi();

    stdbool saveImage(const Matrix<const uint8>& img, const FormatOutputAtom& desc, uint32 id, stdPars(Kit));
    stdbool saveImage(const Matrix<const uint8_x4>& img, const FormatOutputAtom& desc, uint32 id, stdPars(Kit));
    stdbool saveImage(const Point<Space>& imageSize, AtImageProvider<uint8_x4>& imageProvider, const FormatOutputAtom& desc, uint32 id, stdPars(Kit));

    stdbool setOutputDir(const CharType* outputDir, stdPars(Kit));
    stdbool setFps(const FPS& fps, stdPars(Kit));
    stdbool setCodec(const Codec& codec, stdPars(Kit));
    stdbool setMaxSegmentFrames(int32 maxSegmentFrames, stdPars(Kit));

private:
                
    DynamicClass<class OutImgAviImpl> instance;

};

//================================================================
//
// OutImgAviThunk
//
// Temporal thunk: adds necessary kit to OutImgAvi
//
//================================================================

class OutImgAviThunk
    :
    public AtImgConsole,
    public AtVideoOverlay

{

public:

    using Kit = OutImgAvi::Kit;

public:

    OutImgAviThunk(OutImgAvi& outAvi, AtImgConsole& baseConsole, AtVideoOverlay& baseOverlay, const Kit& kit)
        : outAvi(outAvi), baseConsole(baseConsole), baseOverlay(baseOverlay), kit(kit) {}

public:

    stdbool addImageFunc(const Matrix<const uint8_x4>& img, const ImgOutputHint& hint, stdNullPars)
    {
        bool ok1 = errorBlock(outAvi.saveImage(img, hint.desc, hint.id, stdPass));
        bool ok2 = errorBlock(baseConsole.addImage(img, hint, stdPass));
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

    stdbool setImage(const Point<Space>& size, AtImageProvider<uint8_x4>& imageProvider, const FormatOutputAtom& desc, uint32 id, bool textEnabled, stdNullPars)
    {
        bool ok1 = errorBlock(outAvi.saveImage(size, imageProvider, desc, id, stdPass));
        bool ok2 = errorBlock(baseOverlay.setImage(size, imageProvider, desc, id, textEnabled, stdPass));
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

    OutImgAvi& outAvi;
    AtImgConsole& baseConsole;
    AtVideoOverlay& baseOverlay;
    Kit kit;

};

//----------------------------------------------------------------

}

using outImgAvi::OutImgAvi;
using outImgAvi::OutImgAviThunk;
