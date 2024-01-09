#pragma once

#include "baseInterfaces/baseImageConsole.h"
#include "dataAlloc/gpuArrayMemory.h"
#include "gpuAppliedApi/gpuAppliedApi.h"
#include "gpuProcessKit.h"
#include "imageConsole/gpuImageConsole.h"
#include "kits/msgLogsKit.h"

//================================================================
//
// GpuBaseImageProvider
//
// Provider that copies GPU image to CPU destination.
//
// On set image, preallocates intermediate GPU buffer assuming
// max expected CPU row byte alignment.
//
//================================================================

class GpuBaseImageProvider : public BaseImageProvider
{

public:

    using ColorPixel = uint8_x4;
    using MonoPixel = uint8;

public:

    GpuBaseImageProvider(const GpuProcessKit& kit)
        : kit(kit) {}

    stdbool setImage(const GpuMatrixAP<const ColorPixel>& image, stdParsNull);

public:

    Space desiredPitch() const
        {return gpuImage.memPitch();}

    Space desiredBaseByteAlignment() const
        {return kit.gpuProperties.samplerAndFastTransferBaseAlignment;}

    stdbool saveBgr32(const MatrixAP<ColorPixel>& dest, stdParsNull);

    stdbool saveBgr24(const MatrixAP<MonoPixel>& dest, stdParsNull);

private:

    GpuMatrixAP<const ColorPixel> gpuImage;
    GpuArrayMemory<ColorPixel> buffer;
    GpuProcessKit kit;

};

//================================================================
//
// GpuBaseConsoleByCpuThunk
//
// Implements GpuBaseConsole using output to BaseImageConsole.
// Performs copy GPU memory => CPU memory.
//
// The GPU images passed to these functions, should use max (sampler) alignment.
//
//================================================================

class GpuBaseConsoleByCpuThunk : public GpuBaseConsole
{

public:

    stdbool clear(stdPars(Kit))
    {
        if (kit.dataProcessing)
            require(baseImageConsole.clear(stdPassThru));

        returnTrue;
    }

    stdbool update(stdPars(Kit))
    {
        if (kit.dataProcessing)
            require(baseImageConsole.update(stdPassThru));

        returnTrue;
    }

public:

    stdbool addImageBgr(const GpuMatrixAP<const uint8_x4>& img, const ImgOutputHint& hint, stdPars(Kit))
        {return addImageCopyImpl(img, hint, stdPassThru);}

public:

    stdbool overlayClear(stdPars(Kit))
    {
        if (kit.dataProcessing)
            require(baseVideoOverlay.overlayClear(stdPassThru));

        returnTrue;
    }

    stdbool overlaySetImageBgr(const Point<Space>& size, const GpuImageProviderBgr32& img, const ImgOutputHint& hint, stdPars(Kit));

    stdbool overlaySetImageFake(stdPars(Kit))
    {
        if (kit.dataProcessing)
            require(baseVideoOverlay.overlaySetFake(stdPassThru));

        returnTrue;
    }

    stdbool overlayUpdate(stdPars(Kit))
    {
        if (kit.dataProcessing)
            require(baseVideoOverlay.overlayUpdate(stdPassThru));

        returnTrue;
    }

public:

    bool getTextEnabled()
        {return textEnabled;}

    void setTextEnabled(bool textEnabled)
        {this->textEnabled = textEnabled;}

public:

    template <typename Type>
    stdbool addImageCopyImpl(const GpuMatrixAP<const Type>& gpuMatrix, const ImgOutputHint& hint, stdPars(Kit));

public:

    inline GpuBaseConsoleByCpuThunk(BaseImageConsole& baseImageConsole, BaseVideoOverlay& baseVideoOverlay)
        : baseImageConsole(baseImageConsole), baseVideoOverlay(baseVideoOverlay) {}

private:

    BaseImageConsole& baseImageConsole;
    BaseVideoOverlay& baseVideoOverlay;
    bool textEnabled = true;

};
