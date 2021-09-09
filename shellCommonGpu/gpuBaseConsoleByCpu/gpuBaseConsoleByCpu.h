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

    stdbool setImage(const GpuMatrix<const ColorPixel>& image, stdNullPars);

public:

    Space desiredPitch() const
        {return gpuImage.memPitch();}

    Space desiredBaseByteAlignment() const
        {return kit.gpuProperties.samplerAndFastTransferBaseAlignment;}

    stdbool saveBgr32(const Matrix<ColorPixel>& dest, stdNullPars);

    stdbool saveBgr24(const Matrix<MonoPixel>& dest, stdNullPars);

private:

    GpuMatrix<const ColorPixel> gpuImage;
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

    stdbool clear(stdNullPars)
    {
        if (kit.dataProcessing) 
            require(baseImageConsole.clear(stdPassThru)); 

        returnTrue;
    }

    stdbool update(stdNullPars)
    {
        if (kit.dataProcessing)
            require(baseImageConsole.update(stdPassThru));

        returnTrue;
    }

public:

    stdbool addImageBgr(const GpuMatrix<const uint8_x4>& img, const ImgOutputHint& hint, stdNullPars)
        {return addImageCopyImpl(img, hint, stdPassThru);}

public:

    stdbool overlaySetImageBgr(const Point<Space>& size, const GpuImageProviderBgr32& img, const ImgOutputHint& hint, stdNullPars);

    stdbool overlaySetFakeImage(stdNullPars)
    {
        if (kit.dataProcessing)
            require(baseVideoOverlay.setImageFake(stdPassThru));

        returnTrue;
    }

    stdbool overlayUpdate(stdNullPars)
    {
        if (kit.dataProcessing) 
            require(baseVideoOverlay.updateImage(stdPassThru));

        returnTrue;
    }

public:

    bool getTextEnabled()
        {return textEnabled;}

    void setTextEnabled(bool textEnabled)
        {this->textEnabled = textEnabled;}

public:

    template <typename Type>
    stdbool addImageCopyImpl(const GpuMatrix<const Type>& gpuMatrix, const ImgOutputHint& hint, stdNullPars);

public:

    using Kit = KitCombine<GpuProcessKit, MsgLogsKit>;

    inline GpuBaseConsoleByCpuThunk(BaseImageConsole& baseImageConsole, BaseVideoOverlay& baseVideoOverlay, const Kit& kit)
        : baseImageConsole(baseImageConsole), baseVideoOverlay(baseVideoOverlay), kit(kit) {}

private:

    BaseImageConsole& baseImageConsole;
    BaseVideoOverlay& baseVideoOverlay;
    bool overlaySet = false;
    bool textEnabled = true;
    Kit kit;

};
