#pragma once

#include "baseImageConsole/baseImageConsole.h"
#include "dataAlloc/gpuArrayMemory.h"
#include "gpuAppliedApi/gpuAppliedApi.h"
#include "gpuProcessKit.h"
#include "imageConsole/gpuImageConsole.h"
#include "kits/msgLogsKit.h"

//================================================================
//
// GpuBaseImageProvider
//
//================================================================

class GpuBaseImageProvider : public BaseImageProvider
{

public:

    GpuBaseImageProvider(const GpuProcessKit& kit)
        : kit(kit) {}

    stdbool setImage(const GpuMatrix<const uint8_x4>& image, stdNullPars);

public:

    bool dataProcessing() const
        {return kit.dataProcessing;}

    Space getPitch() const
        {return gpuImage.memPitch();}

    Space baseByteAlignment() const
        {return kit.gpuProperties.samplerBaseAlignment;}

    stdbool saveImage(const Matrix<uint8_x4>& dest, stdNullPars);

private:

    GpuMatrix<const uint8_x4> gpuImage;
    GpuArrayMemory<uint8_x4> buffer;

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

    KIT_COMBINE2(Kit, GpuProcessKit, MsgLogsKit);

    inline GpuBaseConsoleByCpuThunk(BaseImageConsole& baseImageConsole, BaseVideoOverlay& baseVideoOverlay, const Kit& kit)
        : baseImageConsole(baseImageConsole), baseVideoOverlay(baseVideoOverlay), kit(kit) {}

private:

    BaseImageConsole& baseImageConsole;
    BaseVideoOverlay& baseVideoOverlay;
    bool overlaySet = false;
    bool textEnabled = true;
    Kit kit;

};
