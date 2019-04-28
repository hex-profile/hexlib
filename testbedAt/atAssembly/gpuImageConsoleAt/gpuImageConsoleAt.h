#pragma once

#include "atInterface/atInterface.h"
#include "dataAlloc/gpuArrayMemory.h"
#include "gpuAppliedApi/gpuAppliedApi.h"
#include "gpuProcessKit.h"
#include "imageConsole/gpuImageConsole.h"
#include "kits/msgLogsKit.h"

//================================================================
//
// AtProviderFromGpuImage
//
//================================================================

class AtProviderFromGpuImage : public AtImageProvider<uint8_x4>
{

public:

    AtProviderFromGpuImage(const GpuProcessKit& kit)
        : kit(kit) {}

    stdbool setImage(const GpuMatrix<const uint8_x4>& image, stdNullPars);

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
// GpuBaseAtConsoleThunk
//
// Implements GpuBaseConsole using output to AtImgConsole.
// Performs copy GPU memory => CPU memory.
//
// The GPU images passed to these functions, should use max (sampler) alignment.
//
//================================================================

class GpuBaseAtConsoleThunk : public GpuBaseConsole
{

public:

    stdbool clear(stdNullPars)
        {return !kit.dataProcessing ? true : atImgConsole.clear(stdPassThru);}

    stdbool update(stdNullPars)
        {return !kit.dataProcessing ? true : atImgConsole.update(stdPassThru);}

public:

    stdbool addImageBgr(const GpuMatrix<const uint8_x4>& img, const ImgOutputHint& hint, stdNullPars)
        {return addImageCopyImpl(img, hint, stdPassThru);}

public:

    stdbool overlaySetImageBgr(const Point<Space>& size, const GpuImageProviderBgr32& img, const ImgOutputHint& hint, stdNullPars);

    stdbool overlaySetFakeImage(stdNullPars)
        {return !kit.dataProcessing ? true : atVideoOverlay.setFakeImage(stdPassThru);}

    stdbool overlayUpdate(stdNullPars)
        {return !kit.dataProcessing ? true : atVideoOverlay.updateImage(stdPassThru);}

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

    inline GpuBaseAtConsoleThunk(AtImgConsole& atImgConsole, AtVideoOverlay& atVideoOverlay, const Kit& kit)
        : atImgConsole(atImgConsole), atVideoOverlay(atVideoOverlay), kit(kit) {}

private:

    AtImgConsole& atImgConsole;
    AtVideoOverlay& atVideoOverlay;
    bool overlaySet = false;
    bool textEnabled = true;
    Kit kit;

};
