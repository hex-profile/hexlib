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

    void setImage(const GpuMatrixAP<const ColorPixel>& image, stdParsNull);

public:

    Space desiredPitch() const
        {return gpuImage.memPitch();}

    Space desiredBaseByteAlignment() const
        {return kit.gpuProperties.samplerAndFastTransferBaseAlignment;}

    void saveBgr32(const MatrixAP<ColorPixel>& dest, stdParsNull);

    void saveBgr24(const MatrixAP<MonoPixel>& dest, stdParsNull);

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

    void clear(stdPars(Kit))
    {
        if (kit.dataProcessing)
            baseImageConsole.clear(stdPassThru);
    }

    void update(stdPars(Kit))
    {
        if (kit.dataProcessing)
            baseImageConsole.update(stdPassThru);
    }

public:

    void addImageBgr(const GpuMatrixAP<const uint8_x4>& img, const ImgOutputHint& hint, stdPars(Kit))
        {addImageCopyImpl(img, hint, stdPassThru);}

public:

    void overlayClear(stdPars(Kit))
    {
        if (kit.dataProcessing)
            baseVideoOverlay.overlayClear(stdPassThru);
    }

    void overlaySetImageBgr(const Point<Space>& size, const GpuImageProviderBgr32& img, const ImgOutputHint& hint, stdPars(Kit));

    void overlaySetImageFake(stdPars(Kit))
    {
        if (kit.dataProcessing)
            baseVideoOverlay.overlaySetFake(stdPassThru);
    }

    void overlayUpdate(stdPars(Kit))
    {
        if (kit.dataProcessing)
            baseVideoOverlay.overlayUpdate(stdPassThru);
    }

public:

    bool getTextEnabled()
        {return textEnabled;}

    void setTextEnabled(bool textEnabled)
        {this->textEnabled = textEnabled;}

public:

    template <typename Type>
    void addImageCopyImpl(const GpuMatrixAP<const Type>& gpuMatrix, const ImgOutputHint& hint, stdPars(Kit));

public:

    inline GpuBaseConsoleByCpuThunk(BaseImageConsole& baseImageConsole, BaseVideoOverlay& baseVideoOverlay)
        : baseImageConsole(baseImageConsole), baseVideoOverlay(baseVideoOverlay) {}

private:

    BaseImageConsole& baseImageConsole;
    BaseVideoOverlay& baseVideoOverlay;
    bool textEnabled = true;

};
