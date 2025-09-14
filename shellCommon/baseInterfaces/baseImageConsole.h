#pragma once

#include "stdFunc/stdFunc.h"
#include "data/matrix.h"
#include "imageConsole/imageConsoleTypes.h"
#include "vectorTypes/vectorBase.h"
#include "storage/adapters/lambdaThunk.h"

//================================================================
//
// BaseImageConsole
//
//================================================================

struct BaseImageConsole
{
    virtual void addImage(const MatrixAP<const uint8_x4>& img, const ImgOutputHint& hint, bool dataProcessing, stdParsNull) =0;
    virtual void clear(stdParsNull) =0;
    virtual void update(stdParsNull) =0;
};

//----------------------------------------------------------------

struct BaseImageConsoleNull : public BaseImageConsole
{
    virtual void addImage(const MatrixAP<const uint8_x4>& img, const ImgOutputHint& hint, bool dataProcessing, stdParsNull)
        {}

    virtual void clear(stdParsNull)
        {}

    virtual void update(stdParsNull)
        {}
};

//================================================================
//
// BaseImageProvider
//
//================================================================

struct BaseImageProvider
{
    // The desired optimal pitch in elements.
    virtual Space desiredPitch() const =0;

    // The desired base address alignment in bytes.
    virtual Space desiredBaseByteAlignment() const =0;

    // Saves to BGR32.
    virtual void saveBgr32(const MatrixAP<uint8_x4>& dest, stdParsNull) =0;

    // Saves to BGR24. The destination is an uint8 image
    // with width 3 times more than the color image width.
    virtual void saveBgr24(const MatrixAP<uint8>& dest, stdParsNull) =0;
};

//================================================================
//
// BaseVideoOverlay
//
//================================================================

struct BaseVideoOverlay
{
    virtual void overlayClear(stdParsNull) =0;

    virtual void overlaySet
    (
        const Point<Space>& size,
        bool dataProcessing,
        BaseImageProvider& imageProvider,
        const FormatOutputAtom& desc,
        uint32 id,
        bool textEnabled,
        stdParsNull
    )
    =0;

    virtual void overlaySetFake(stdParsNull) =0;

    virtual void overlayUpdate(stdParsNull) =0;
};

//================================================================
//
// BaseVideoOverlayNull
//
//================================================================

class BaseVideoOverlayNull : public BaseVideoOverlay
{
    virtual void overlayClear(stdParsNull)
        {}

    virtual void overlaySet(const Point<Space>& size, bool dataProcessing, BaseImageProvider& imageProvider, const FormatOutputAtom& desc, uint32 id, bool textEnabled, stdParsNull)
        {}

    virtual void overlaySetFake(stdParsNull)
        {}

    virtual void overlayUpdate(stdParsNull)
        {}
};
