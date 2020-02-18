#pragma once

#include "charType/charType.h"
#include "stdFunc/stdFunc.h"
#include "atInterface/atInterfaceKit.h"
#include "data/matrix.h"
#include "imageConsole/imageConsole.h"
#include "vectorTypes/vectorBase.h"
#include "baseImageConsole/baseImageConsole.h"

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
// AtImgConsole
//
//================================================================

using AtImgConsole = BaseImageConsole;

//================================================================
//
// AtImageProvider
// AtVideoOverlay
// AtAsyncOverlay
//
//================================================================

using AtImageProvider = BaseImageProvider;

//----------------------------------------------------------------

using AtVideoOverlay = BaseVideoOverlay;

//----------------------------------------------------------------

struct AtAsyncOverlay
{
    virtual stdbool setImage(const Point<Space>& size, BaseImageProvider& imageProvider, stdNullPars) =0;
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
