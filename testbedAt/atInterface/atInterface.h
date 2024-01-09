#pragma once

#include "charType/charType.h"
#include "stdFunc/stdFunc.h"
#include "atInterface/atInterfaceKit.h"
#include "data/matrix.h"
#include "imageConsole/imageConsoleTypes.h"
#include "vectorTypes/vectorBase.h"
#include "baseInterfaces/baseImageConsole.h"
#include "baseInterfaces/baseSignals.h"

//================================================================
//
// AtVideoFrame
//
//================================================================

struct AtVideoFrame : public MatrixAP<const uint8_x4>
{
    inline AtVideoFrame(const MatrixAP<const uint8_x4>& that)
        : MatrixAP<const uint8_x4>(that) {}
};

//================================================================
//
// BaseImageConsole
// BaseImageProvider
// BaseVideoOverlay
// AtAsyncOverlay
//
//================================================================

struct AtAsyncOverlay
{
    virtual stdbool setImage(const Point<Space>& size, BaseImageProvider& imageProvider, stdNullPars) =0;
};

//================================================================
//
// AtVideoInfo
//
//================================================================

struct AtVideoInfo
{
    CharArray videofileName;
    Space frameIndex;
    Space frameCount;
    bool interlacedMode;
    bool interlacedLower;
    Point<Space> frameSize;
};
