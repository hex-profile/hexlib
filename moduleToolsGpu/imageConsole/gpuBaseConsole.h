#pragma once

#include "data/gpuMatrix.h"
#include "imageConsole/imageConsoleTypes.h"
#include "stdFunc/stdFunc.h"
#include "vectorTypes/vectorBase.h"
#include "storage/adapters/lambdaThunk.h"
#include "gpuProcessKit.h"

//================================================================
//
// GpuBaseConsoleKit
//
//================================================================

using GpuBaseConsoleKit = GpuProcessKit;

//================================================================
//
// GpuBaseConsoleSettings
//
//================================================================

struct GpuBaseConsoleSettings
{
    virtual bool getTextEnabled() =0;
    virtual void setTextEnabled(bool textEnabled) =0;
};

//================================================================
//
// GpuBaseConsoleImages
//
//================================================================

struct GpuBaseConsoleImages
{
    using Kit = GpuBaseConsoleKit;

    virtual stdbool clear(stdPars(Kit)) =0;
    virtual stdbool update(stdPars(Kit)) =0;
    virtual stdbool addImageBgr(const GpuMatrix<const uint8_x4>& img, const ImgOutputHint& hint, stdPars(Kit)) =0;
};

//================================================================
//
// GpuImageProviderBgr32
//
//================================================================

struct GpuImageProviderBgr32
{
    virtual stdbool saveImage(const GpuMatrix<uint8_x4>& dest, stdNullPars) const =0;
};

LAMBDA_THUNK
(
    gpuImageProviderBgr32,
    GpuImageProviderBgr32,
    stdbool saveImage(const GpuMatrix<uint8_x4>& dest, stdNullPars) const,
    lambda(dest, stdNullPass)
)

//================================================================
//
// GpuBaseConsoleOverlay
//
//================================================================

struct GpuBaseConsoleOverlay
{
    using Kit = GpuBaseConsoleKit;

    virtual stdbool overlayClear(stdPars(Kit)) =0;
    virtual stdbool overlaySetImageBgr(const Point<Space>& size, const GpuImageProviderBgr32& img, const ImgOutputHint& hint, stdPars(Kit)) =0;
    virtual stdbool overlaySetImageFake(stdPars(Kit)) =0;
    virtual stdbool overlayUpdate(stdPars(Kit)) =0;
};

//================================================================
//
// GpuBaseConsole
//
// Abstract interface of image output console taking GPU images.
//
//================================================================

struct GpuBaseConsole : GpuBaseConsoleSettings, GpuBaseConsoleImages, GpuBaseConsoleOverlay
{
    using Kit = GpuBaseConsoleKit;
};

//================================================================
//
// GpuBaseConsoleNull
//
//================================================================

class GpuBaseConsoleNull : public GpuBaseConsole
{

public:

    stdbool clear(stdPars(Kit))
        {returnTrue;}

    stdbool update(stdPars(Kit))
        {returnTrue;}

    stdbool addImageBgr(const GpuMatrix<const uint8_x4>& img, const ImgOutputHint& hint, stdPars(Kit))
        {returnTrue;}

    stdbool overlayClear(stdPars(Kit))
        {returnTrue;}

    stdbool overlaySetImageBgr(const Point<Space>& size, const GpuImageProviderBgr32& img, const ImgOutputHint& hint, stdPars(Kit))
        {returnTrue;}

    stdbool overlaySetImageFake(stdPars(Kit))
        {returnTrue;}

    stdbool overlayUpdate(stdPars(Kit))
        {returnTrue;}

    bool getTextEnabled()
        {return false;}

    void setTextEnabled(bool textEnabled)
        {}

};

//================================================================
//
// GpuBaseConsoleSplitter
//
//================================================================

class GpuBaseConsoleSplitter : public GpuBaseConsole
{

public:

    GpuBaseConsoleSplitter(GpuBaseConsole& a, GpuBaseConsole& b)
        : a(a), b(b) {}

public:

    stdbool clear(stdPars(Kit));
    stdbool update(stdPars(Kit));
    stdbool addImageBgr(const GpuMatrix<const uint8_x4>& img, const ImgOutputHint& hint, stdPars(Kit));
    stdbool overlayClear(stdPars(Kit));
    stdbool overlaySetImageBgr(const Point<Space>& size, const GpuImageProviderBgr32& img, const ImgOutputHint& hint, stdPars(Kit));
    stdbool overlaySetImageFake(stdPars(Kit));
    stdbool overlayUpdate(stdPars(Kit));
    bool getTextEnabled();
    void setTextEnabled(bool textEnabled);

private:

    GpuBaseConsole& a;
    GpuBaseConsole& b;

};
