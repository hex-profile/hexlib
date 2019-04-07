#pragma once

#include "atInterface/atInterface.h"
#include "outImgAvi/objectHolder.h"
#include "errorLog/errorLogKit.h"
#include "kits/msgLogsKit.h"
#include "interfaces/fileTools.h"
#include "allocation/mallocKit.h"

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

    bool saveImage(const Matrix<const uint8>& img, const FormatOutputAtom& desc, uint32 id, stdPars(Kit));
    bool saveImage(const Matrix<const uint8_x4>& img, const FormatOutputAtom& desc, uint32 id, stdPars(Kit));
    bool saveImage(const Point<Space>& imageSize, AtImageProvider<uint8_x4>& imageProvider, const FormatOutputAtom& desc, uint32 id, stdPars(Kit));

    bool setOutputDir(const CharType* outputDir, stdPars(Kit));
    bool setFps(const FPS& fps, stdPars(Kit));
    bool setCodec(const Codec& codec, stdPars(Kit));
    bool setMaxSegmentFrames(int32 maxSegmentFrames, stdPars(Kit));

private:

    ObjectHolder<class OutImgAviImpl> instance;

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

    UseType(OutImgAvi, Kit);

public:

    OutImgAviThunk(OutImgAvi& outAvi, AtImgConsole& baseConsole, AtVideoOverlay& baseOverlay, const Kit& kit)
        : outAvi(outAvi), baseConsole(baseConsole), baseOverlay(baseOverlay), kit(kit) {}

public:

    bool addImageFunc(const Matrix<const uint8>& img, const ImgOutputHint& hint, stdNullPars)
    {
        stdBegin;
        bool ok1 = outAvi.saveImage(img, hint.desc, hint.id, stdPass);
        bool ok2 = baseConsole.addImage(img, hint, stdPass);
        require(ok1 && ok2);
        stdEnd;
    }

    bool addImageFunc(const Matrix<const uint8_x4>& img, const ImgOutputHint& hint, stdNullPars)
    {
        stdBegin;
        bool ok1 = outAvi.saveImage(img, hint.desc, hint.id, stdPass);
        bool ok2 = baseConsole.addImage(img, hint, stdPass);
        require(ok1 && ok2);
        stdEnd;
    }

    bool clear(stdNullPars)
        {return baseConsole.clear(stdPassThru);}

    bool update(stdNullPars)
        {return baseConsole.update(stdPassThru);}

public:

    bool setImage(const Point<Space>& size, AtImageProvider<uint8_x4>& imageProvider, const FormatOutputAtom& desc, uint32 id, bool textEnabled, stdNullPars)
    {
        stdBegin;
        bool ok1 = outAvi.saveImage(size, imageProvider, desc, id, stdPass);
        bool ok2 = baseOverlay.setImage(size, imageProvider, desc, id, textEnabled, stdPass);
        require(ok1 && ok2);
        stdEnd;
    }

    bool setFakeImage(stdNullPars)
    {
        return baseOverlay.setFakeImage(stdPassThru);
    }

    bool updateImage(stdNullPars) {return true;}

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