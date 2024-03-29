#pragma once

#include "gpuMatrixCopy/gpuMatrixCopy.h"
#include "channels/buffers/overlayBuffer/overlayBuffer.h"

namespace gpuBaseConsoleThunk {

//================================================================
//
// OverlayUpdater
//
//================================================================

using OverlayUpdater = Callable<stdbool (stdParsNull)>;

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

    virtual stdbool clear(stdPars(Kit))
        {returnTrue;}

    virtual stdbool update(stdPars(Kit))
        {returnTrue;}

    ////

    virtual stdbool addImageBgr(const GpuMatrixAP<const uint8_x4>& img, const ImgOutputHint& hint, stdPars(Kit))
    {
        if (hint.target == ImgOutputOverlay)
        {
            auto provider = gpuImageProviderBgr32 | [&] (const GpuMatrixAP<uint8_x4>& dest, stdParsNull)
                {return gpuMatrixCopy(img, dest, stdPass);};

            require(overlaySetImageBgr(img.size(), provider, hint, stdPass));
        }

        returnTrue;
    }

    ////

    virtual stdbool overlayClear(stdPars(Kit))
    {
        overlayBuffer.clearImage();
        returnTrue;
    }

    ////

    virtual stdbool overlaySetImageBgr(const Point<Space>& size, const GpuImageProviderBgr32& img, const ImgOutputHint& hint, stdPars(Kit))
    {
        stdScopedBegin;

        auto& oldKit = kit;
        auto kit = kitCombine(oldKit, auxKit);

        require(overlayBuffer.setImage(size, img, stdPass));

        if (textEnabled && kit.dataProcessing)
            printMsg(kit.localLog, STR("OVERLAY: %"), hint.desc);

        stdScopedEnd;
    }

    virtual stdbool overlaySetImageFake(stdPars(Kit))
    {
        returnTrue;
    }

    virtual stdbool overlayUpdate(stdPars(Kit))
    {
        return overlayUpdater(stdPassNull);
    }
};

//----------------------------------------------------------------

}

using gpuBaseConsoleThunk::GpuBaseConsoleThunk;

