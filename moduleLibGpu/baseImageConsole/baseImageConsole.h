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

//================================================================
//
// BaseImageProvider
//
//================================================================

struct BaseImageProvider
{
    virtual Space getPitch() const =0;
    virtual Space baseByteAlignment() const =0;
    virtual stdbool saveImage(const Matrix<uint8_x4>& dest, stdNullPars) =0;
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
