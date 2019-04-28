#pragma once

#include "charType/charType.h"
#include "stdFunc/stdFunc.h"
#include "atInterface/atInterfaceKit.h"
#include "data/matrix.h"
#include "imageConsole/imageConsole.h"
#include "vectorTypes/vectorBase.h"

//================================================================
//
// AtVideoFrame
//
//================================================================

struct AtVideoFrame : public Matrix<const uint8_x4>
{
    inline AtVideoFrame(const Matrix<const uint8_x4>& that)
        : Matrix<const uint8_x4>(that) {}
};

//================================================================
//
// AtVideoOverlay
//
//================================================================

template <typename Element>
struct AtImageProvider
{
    virtual Space getPitch() const =0;
    virtual Space baseByteAlignment() const =0;

    virtual stdbool saveImage(const Matrix<Element>& dest, stdNullPars) =0;
};

//----------------------------------------------------------------

struct AtVideoOverlay
{
    virtual stdbool setImage(const Point<Space>& size, AtImageProvider<uint8_x4>& imageProvider, const FormatOutputAtom& desc, uint32 id, bool textEnabled, stdNullPars) =0;
    virtual stdbool setFakeImage(stdNullPars) =0;
    virtual stdbool updateImage(stdNullPars) =0;
};

//----------------------------------------------------------------

struct AtAsyncOverlay
{
    virtual stdbool setImage(const Point<Space>& size, AtImageProvider<uint8_x4>& imageProvider, stdNullPars) =0;
};

//================================================================
//
// AtImgConsole
//
//================================================================

struct AtImgConsole
{
    virtual stdbool addImageFunc(const Matrix<const uint8_x4>& img, const ImgOutputHint& hint, stdNullPars) =0;
    virtual stdbool clear(stdNullPars) =0;
    virtual stdbool update(stdNullPars) =0;

    template <typename Type>
    inline stdbool addImage(const Matrix<Type>& img, const ImgOutputHint& hint, stdNullPars)
        {return addImageFunc(makeConst(img), hint, stdNullPassThru);}
};

//================================================================
//
// AtSignalTest
//
//================================================================

using AtActionId = uint32;

//================================================================
//
// AtSignalSet
//
//================================================================

struct AtSignalSet
{
    virtual bool actsetClear() =0;
    virtual bool actsetUpdate() =0;
    virtual bool actsetAdd(AtActionId id, const CharType* name, const CharType* key, const CharType* comment) =0;
};

//================================================================
//
// AtSignalTest
//
//================================================================

struct AtSignalTest
{
    virtual int32 actionCount() =0;
    virtual bool actionItem(int32 index, AtActionId& id) =0;
};

//================================================================
//
// AtVideoInfo
//
//================================================================

KIT_CREATE6(
    AtVideoInfo,
    CharArray, videofileName,
    Space, frameIndex,
    Space, frameCount,
    bool, interlacedMode,
    bool, interlacedLower,
    Point<Space>, frameSize
);
