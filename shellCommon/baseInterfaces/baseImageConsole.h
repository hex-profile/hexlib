#pragma once

#include "stdFunc/stdFunc.h"
#include "data/matrix.h"
#include "imageConsole/imageConsole.h"
#include "vectorTypes/vectorBase.h"

//================================================================
//
// BaseImageConsole
//
//================================================================

struct BaseImageConsole
{
    virtual stdbool addImage(const Matrix<const uint8_x4>& img, const ImgOutputHint& hint, stdNullPars) =0;
    virtual stdbool clear(stdNullPars) =0;
    virtual stdbool update(stdNullPars) =0;
};

//----------------------------------------------------------------

struct BaseImageConsoleNull : public BaseImageConsole
{
    virtual stdbool addImage(const Matrix<const uint8_x4>& img, const ImgOutputHint& hint, stdNullPars)
        {returnTrue;}

    virtual stdbool clear(stdNullPars)
        {returnTrue;}

    virtual stdbool update(stdNullPars)
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
    virtual stdbool saveBgr32(const Matrix<uint8_x4>& dest, stdNullPars) =0;

    // Saves to BGR24. The destination is an uint8 image
    // with width 3 times more than the color image width.
    virtual stdbool saveBgr24(const Matrix<uint8>& dest, stdNullPars) =0;
};

//================================================================
//
// BaseVideoOverlay
//
//================================================================

struct BaseVideoOverlay
{
    virtual stdbool setImage(const Point<Space>& size, bool dataProcessing, BaseImageProvider& imageProvider, const FormatOutputAtom& desc, uint32 id, bool textEnabled, stdNullPars) =0;
    virtual stdbool setImageFake(stdNullPars) =0;
    virtual stdbool updateImage(stdNullPars) =0;
};

//----------------------------------------------------------------

class BaseVideoOverlayNull : public BaseVideoOverlay
{
    virtual stdbool setImage(const Point<Space>& size, bool dataProcessing, BaseImageProvider& imageProvider, const FormatOutputAtom& desc, uint32 id, bool textEnabled, stdNullPars)
        {returnTrue;}

    virtual stdbool setImageFake(stdNullPars)
        {returnTrue;}

    virtual stdbool updateImage(stdNullPars)
        {returnTrue;}
};
