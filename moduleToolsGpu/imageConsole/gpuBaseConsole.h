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

    virtual void clear(stdPars(Kit)) =0;
    virtual void update(stdPars(Kit)) =0;
    virtual void addImageBgr(const GpuMatrixAP<const uint8_x4>& img, const ImgOutputHint& hint, stdPars(Kit)) =0;
};

//================================================================
//
// GpuImageProviderBgr32
//
//================================================================

struct GpuImageProviderBgr32
{
    virtual void saveImage(const GpuMatrixAP<uint8_x4>& dest, stdParsNull) const =0;
};

LAMBDA_THUNK
(
    gpuImageProviderBgr32,
    GpuImageProviderBgr32,
    void saveImage(const GpuMatrixAP<uint8_x4>& dest, stdParsNull) const,
    lambda(dest, stdPassNull)
)

//================================================================
//
// GpuBaseConsoleOverlay
//
//================================================================

struct GpuBaseConsoleOverlay
{
    using Kit = GpuBaseConsoleKit;

    virtual void overlayClear(stdPars(Kit)) =0;
    virtual void overlaySetImageBgr(const Point<Space>& size, const GpuImageProviderBgr32& img, const ImgOutputHint& hint, stdPars(Kit)) =0;
    virtual void overlaySetImageFake(stdPars(Kit)) =0;
    virtual void overlayUpdate(stdPars(Kit)) =0;
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

    void clear(stdPars(Kit))
        {}

    void update(stdPars(Kit))
        {}

    void addImageBgr(const GpuMatrixAP<const uint8_x4>& img, const ImgOutputHint& hint, stdPars(Kit))
        {}

    void overlayClear(stdPars(Kit))
        {}

    void overlaySetImageBgr(const Point<Space>& size, const GpuImageProviderBgr32& img, const ImgOutputHint& hint, stdPars(Kit))
        {}

    void overlaySetImageFake(stdPars(Kit))
        {}

    void overlayUpdate(stdPars(Kit))
        {}

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

    void clear(stdPars(Kit));
    void update(stdPars(Kit));
    void addImageBgr(const GpuMatrixAP<const uint8_x4>& img, const ImgOutputHint& hint, stdPars(Kit));
    void overlayClear(stdPars(Kit));
    void overlaySetImageBgr(const Point<Space>& size, const GpuImageProviderBgr32& img, const ImgOutputHint& hint, stdPars(Kit));
    void overlaySetImageFake(stdPars(Kit));
    void overlayUpdate(stdPars(Kit));
    bool getTextEnabled();
    void setTextEnabled(bool textEnabled);

private:

    GpuBaseConsole& a;
    GpuBaseConsole& b;

};
