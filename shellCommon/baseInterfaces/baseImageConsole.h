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
    virtual stdbool addImage(const MatrixAP<const uint8_x4>& img, const ImgOutputHint& hint, bool dataProcessing, stdParsNull) =0;
    virtual stdbool clear(stdParsNull) =0;
    virtual stdbool update(stdParsNull) =0;
};

//----------------------------------------------------------------

struct BaseImageConsoleNull : public BaseImageConsole
{
    virtual stdbool addImage(const MatrixAP<const uint8_x4>& img, const ImgOutputHint& hint, bool dataProcessing, stdParsNull)
        {returnTrue;}

    virtual stdbool clear(stdParsNull)
        {returnTrue;}

    virtual stdbool update(stdParsNull)
        {returnTrue;}
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
    virtual stdbool saveBgr32(const MatrixAP<uint8_x4>& dest, stdParsNull) =0;

    // Saves to BGR24. The destination is an uint8 image
    // with width 3 times more than the color image width.
    virtual stdbool saveBgr24(const MatrixAP<uint8>& dest, stdParsNull) =0;
};

//================================================================
//
// BaseVideoOverlay
//
//================================================================

struct BaseVideoOverlay
{
    virtual stdbool overlayClear(stdParsNull) =0;

    virtual stdbool overlaySet
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

    virtual stdbool overlaySetFake(stdParsNull) =0;

    virtual stdbool overlayUpdate(stdParsNull) =0;
};

//================================================================
//
// BaseVideoOverlayNull
//
//================================================================

class BaseVideoOverlayNull : public BaseVideoOverlay
{
    virtual stdbool overlayClear(stdParsNull)
        {returnTrue;}

    virtual stdbool overlaySet(const Point<Space>& size, bool dataProcessing, BaseImageProvider& imageProvider, const FormatOutputAtom& desc, uint32 id, bool textEnabled, stdParsNull)
        {returnTrue;}

    virtual stdbool overlaySetFake(stdParsNull)
        {returnTrue;}

    virtual stdbool overlayUpdate(stdParsNull)
        {returnTrue;}
};
