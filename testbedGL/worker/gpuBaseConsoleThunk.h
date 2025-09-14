#pragma once

#include "gpuMatrixCopy/gpuMatrixCopy.h"
#include "channels/buffers/overlayBuffer/overlayBuffer.h"

namespace gpuBaseConsoleThunk {

//================================================================
//
// OverlayUpdater
//
//================================================================

using OverlayUpdater = Callable<void (stdParsNull)>;

//================================================================
//
// ThunkContext
//
//================================================================

struct ThunkContext
{
    bool textEnabled;

    OverlayBuffer& overlayBuffer;

    OverlayUpdater& overlayUpdater;

    using AuxKit = KitCombine<GpuEventAllocKit, GpuMemoryAllocationKit, LocalLogKit>;
    AuxKit auxKit;
};

//================================================================
//
// GpuBaseConsoleThunk
//
//================================================================

struct GpuBaseConsoleThunk : public GpuBaseConsole, private ThunkContext
{
    GpuBaseConsoleThunk(const ThunkContext& context)
        : ThunkContext{context} {}

    ////

    virtual bool getTextEnabled()
        {return textEnabled;}

    virtual void setTextEnabled(bool textEnabled)
        {this->textEnabled = textEnabled;}

    //
    // Small image console is not supported.
    //

    virtual void clear(stdPars(Kit))
        {}

    virtual void update(stdPars(Kit))
        {}

    ////

    virtual void addImageBgr(const GpuMatrixAP<const uint8_x4>& img, const ImgOutputHint& hint, stdPars(Kit))
    {
        if (hint.target == ImgOutputOverlay)
        {
            auto provider = gpuImageProviderBgr32 | [&] (const GpuMatrixAP<uint8_x4>& dest, stdParsNull)
                {gpuMatrixCopy(img, dest, stdPass);};

            overlaySetImageBgr(img.size(), provider, hint, stdPass);
        }
    }

    ////

    virtual void overlayClear(stdPars(Kit))
    {
        overlayBuffer.clearImage();
    }

    ////

    virtual void overlaySetImageBgr(const Point<Space>& size, const GpuImageProviderBgr32& img, const ImgOutputHint& hint, stdPars(Kit))
    {
        stdScopedBegin;

        auto& oldKit = kit;
        auto kit = kitCombine(oldKit, auxKit);

        overlayBuffer.setImage(size, img, stdPass);

        if (textEnabled && kit.dataProcessing)
            printMsg(kit.localLog, STR("OVERLAY: %"), hint.desc);

        stdScopedEnd;
    }

    virtual void overlaySetImageFake(stdPars(Kit))
    {
    }

    virtual void overlayUpdate(stdPars(Kit))
    {
        overlayUpdater(stdPassNull);
    }
};

//----------------------------------------------------------------

}

using gpuBaseConsoleThunk::GpuBaseConsoleThunk;

