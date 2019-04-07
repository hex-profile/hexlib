#pragma once

#include "imageConsole/gpuImageConsole.h"
#include "atInterface/atInterface.h"
#include "gpuProcessKit.h"
#include "gpuAppliedApi/gpuAppliedApi.h"
#include "kits/msgLogsKit.h"

//================================================================
//
// AtProviderFromGpuImage
//
//================================================================

class AtProviderFromGpuImage : public AtImageProvider<uint8_x4>
{

public:

    AtProviderFromGpuImage(const GpuMatrix<const uint8_x4>& gpuImage, const GpuProcessKit& kit)
        : gpuImage(gpuImage), kit(kit) {}

    Space getPitch() const
        {return gpuImage.memPitch();}

    Space baseByteAlignment() const
        {return kit.gpuProperties.samplerBaseAlignment;}

    bool saveImage(const Matrix<uint8_x4>& dest, stdNullPars);

private:

    GpuMatrix<const uint8_x4> gpuImage;
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

    bool clear(stdNullPars)
        {return !kit.dataProcessing ? true : atImgConsole.clear(stdPassThru);}

    bool update(stdNullPars)
        {return !kit.dataProcessing ? true : atImgConsole.update(stdPassThru);}

public:

    bool addImage(const GpuMatrix<const uint8>& img, const ImgOutputHint& hint, stdNullPars)
        {return addImageCopyImpl(img, hint, stdPassThru);}

public:

    bool addImageBgr(const GpuMatrix<const uint8_x4>& img, const ImgOutputHint& hint, stdNullPars)
        {return addImageCopyImpl(img, hint, stdPassThru);}

public:

    bool overlaySetImageBgr(const Point<Space>& size, const GpuImageProviderBgr32& img, const ImgOutputHint& hint, stdNullPars);

    bool overlaySetFakeImage(stdNullPars)
        {return !kit.dataProcessing ? true : atVideoOverlay.setFakeImage(stdPassThru);}

    bool overlayUpdate(stdNullPars)
        {return !kit.dataProcessing ? true : atVideoOverlay.updateImage(stdPassThru);}

public:

    bool getTextEnabled()
        {return textEnabled;}

    void setTextEnabled(bool textEnabled)
        {this->textEnabled = textEnabled;}

public:

    template <typename Type>
    bool addImageCopyImpl(const GpuMatrix<const Type>& gpuMatrix, const ImgOutputHint& hint, stdNullPars);

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
