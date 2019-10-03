#pragma once

#include "imageConsole/gpuImageConsole.h"
#include "gpuProcessKit.h"
#include "kits/msgLogsKit.h"
#include "errorLog/errorLog.h"
#include "kits/userPoint.h"
#include "kits/moduleKit.h"

namespace gpuImageConsoleImpl {

//================================================================
//
// GpuProhibitedConsoleThunk
//
//================================================================

class GpuProhibitedConsoleThunk : public GpuBaseConsole
{

public:

    stdbool clear(stdNullPars)
        {REQUIRE(false); returnTrue;}

    stdbool update(stdNullPars)
        {REQUIRE(false); returnTrue;}

    stdbool addImage(const GpuMatrix<const uint8>& img, const ImgOutputHint& hint, stdNullPars)
        {REQUIRE(false); returnTrue;}

    stdbool addImageBgr(const GpuMatrix<const uint8_x4>& img, const ImgOutputHint& hint, stdNullPars)
        {REQUIRE(false); returnTrue;}

    stdbool overlaySetImageBgr(const Point<Space>& size, const GpuImageProviderBgr32& img, const ImgOutputHint& hint, stdNullPars)
        {REQUIRE(false); returnTrue;}

    stdbool overlaySetFakeImage(stdNullPars)
        {REQUIRE(false); returnTrue;}

    stdbool overlayUpdate(stdNullPars)
        {REQUIRE(false); returnTrue;}

    bool getTextEnabled()
        {return false;}

    void setTextEnabled(bool textEnabled)
        {}

public:

    GpuProhibitedConsoleThunk(const ErrorLogKit& kit) : kit(kit) {}

private:

    ErrorLogKit kit;

};

//================================================================
//
// GpuBaseConsoleIgnoreThunk
//
//================================================================

class GpuBaseConsoleIgnoreThunk : public GpuBaseConsole
{

public:

    stdbool clear(stdNullPars)
        {returnTrue;}

    stdbool update(stdNullPars)
        {returnTrue;}

    stdbool addImage(const GpuMatrix<const uint8>& img, const ImgOutputHint& hint, stdNullPars)
        {returnTrue;}

    stdbool addImageBgr(const GpuMatrix<const uint8_x4>& img, const ImgOutputHint& hint, stdNullPars)
        {returnTrue;}

    stdbool overlaySetImageBgr(const Point<Space>& size, const GpuImageProviderBgr32& img, const ImgOutputHint& hint, stdNullPars)
        {returnTrue;}

    stdbool overlaySetFakeImage(stdNullPars)
        {returnTrue;}

    stdbool overlayUpdate(stdNullPars)
        {returnTrue;}

    bool getTextEnabled()
        {return false;}

    void setTextEnabled(bool textEnabled)
        {}

};

//================================================================
//
// VectorDisplayMode
//
//================================================================

enum VectorDisplayMode
{
    VectorDisplayColor,
    VectorDisplayMagnitude,
    VectorDisplayX,
    VectorDisplayY,

    VectorDisplayModeCount
};

//================================================================
//
// ColorMode
//
//================================================================

enum ColorMode {ColorYuv, ColorRgb};

//================================================================
//
// GpuImageConsoleThunk
//
//================================================================

class GpuImageConsoleThunk : public GpuImageConsole
{

public:

    inline GpuImageConsoleKit getKit()
        {return GpuImageConsoleKit(*this, displayFactor);}

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

public:

    stdbool clear(stdNullPars)
        {return baseConsole.clear(stdPassThru);}

    stdbool update(stdNullPars)
        {return baseConsole.update(stdPassThru);}

    stdbool addImageBgr(const GpuMatrix<const uint8_x4>& img, const ImgOutputHint& hint, stdNullPars)
        {return baseConsole.addImageBgr(img, hint, stdPassThru);}

    stdbool overlaySetImageBgr(const Point<Space>& size, const GpuImageProviderBgr32& img, const ImgOutputHint& hint, stdNullPars)
        {return baseConsole.overlaySetImageBgr(size, img, hint, stdPassThru);}

    stdbool overlaySetFakeImage(stdNullPars)
        {return baseConsole.overlaySetFakeImage(stdPassThru);}

    stdbool overlayUpdate(stdNullPars)
        {return baseConsole.overlayUpdate(stdPassThru);}

    bool getTextEnabled()
        {return baseConsole.getTextEnabled();}

    void setTextEnabled(bool textEnabled)
        {baseConsole.setTextEnabled(textEnabled);}

    //----------------------------------------------------------------
    //
    // Add scalar image with upsampling and range scaling
    //
    //----------------------------------------------------------------

public:

    #define TMP_MACRO(Type, _) \
        \
        stdbool addMatrixFunc \
        ( \
            const GpuMatrix<const Type>& img, \
            float32 minVal, float32 maxVal, \
            const Point<float32>& upsampleFactor, \
            InterpType upsampleType, \
            const Point<Space>& upsampleSize, \
            BorderMode borderMode, \
            const ImgOutputHint& hint, \
            stdNullPars \
        ) \
        { \
            return addMatrixExImpl(img, 0, minVal, maxVal, upsampleFactor, upsampleType, upsampleSize, borderMode, hint, stdPassThru); \
        }

    IMAGE_CONSOLE_FOREACH_SCALAR_TYPE(TMP_MACRO, _)

    #undef TMP_MACRO

    //----------------------------------------------------------------
    //
    // Add the selected scalar plane of 2X-vector image
    //
    //----------------------------------------------------------------

public:

    #define TMP_MACRO(Type, _) \
        \
        stdbool addMatrixChanFunc \
        ( \
            const GpuMatrix<const Type>& img, \
            int channel, \
            float32 minVal, float32 maxVal, \
            const Point<float32>& upsampleFactor, \
            InterpType upsampleType, \
            const Point<Space>& upsampleSize, \
            BorderMode borderMode, \
            const ImgOutputHint& hint, \
            stdNullPars \
        ) \
        { \
            return addMatrixExImpl(img, channel, minVal, maxVal, upsampleFactor, upsampleType, upsampleSize, borderMode, hint, stdPassThru); \
        }

    IMAGE_CONSOLE_FOREACH_VECTOR_TYPE(TMP_MACRO, _)

    #undef TMP_MACRO

    //----------------------------------------------------------------
    //
    //
    //
    //----------------------------------------------------------------

private:

    template <typename Type>
    stdbool addMatrixExImpl
    (
        const GpuMatrix<const Type>& img,
        int channel,
        float32 minVal, float32 maxVal,
        const Point<float32>& upsampleFactor,
        InterpType upsampleType,
        const Point<Space>& upsampleSize,
        BorderMode borderMode,
        const ImgOutputHint& hint,
        stdNullPars
    );

    //----------------------------------------------------------------
    //
    // addVectorImage
    //
    //----------------------------------------------------------------

public:

    template <typename Vector>
    stdbool addVectorImageGeneric
    (
        const GpuMatrix<const Vector>& image,
        float32 maxVector,
        const Point<float32>& upsampleFactor,
        InterpType upsampleType,
        const Point<Space>& upsampleSize,
        BorderMode borderMode,
        const ImgOutputHint& hint,
        stdNullPars
    );

    #define TMP_MACRO(Vector, o) \
        \
        stdbool addVectorImageFunc \
        ( \
            const GpuMatrix<const Vector>& image, \
            float32 maxVector, \
            const Point<float32>& upsampleFactor, \
            InterpType upsampleType, \
            const Point<Space>& upsampleSize, \
            BorderMode borderMode, \
            const ImgOutputHint& hint, \
            stdNullPars \
        ) \
        { \
            return addVectorImageGeneric<Vector>(image, maxVector, upsampleFactor, upsampleType, upsampleSize, borderMode, hint, stdPassThru); \
        }

    IMAGE_CONSOLE_FOREACH_VECTOR_IMAGE_TYPE(TMP_MACRO, o)

    #undef TMP_MACRO

    //----------------------------------------------------------------
    //
    // addYuvImage420
    //
    //----------------------------------------------------------------

public:

    template <typename Type>
    stdbool addYuvImage420Func
    (
        const GpuPackedYuv<const Type>& image,
        const ImgOutputHint& hint,
        stdNullPars
    );

    ////

    #define TMP_MACRO(Type, _) \
        \
        stdbool addYuvImage420 \
        ( \
            const GpuPackedYuv<const Type>& image, \
            const ImgOutputHint& hint, \
            stdNullPars \
        ) \
        { \
            return addYuvImage420Func(image, hint, stdNullPassThru); \
        }

    IMAGE_CONSOLE_FOREACH_YUV420_TYPE(TMP_MACRO, _)

    #undef TMP_MACRO

    //----------------------------------------------------------------
    //
    // Add YUV color image in 4:4:4 sampling
    //
    //----------------------------------------------------------------

public:

    template <typename Type>
    stdbool addColorImageFunc
    (
        const GpuMatrix<const Type>& img,
        float32 minVal, float32 maxVal,
        const Point<float32>& upsampleFactor,
        InterpType upsampleType,
        const Point<Space>& upsampleSize,
        BorderMode borderMode,
        const ImgOutputHint& hint,
        ColorMode colorMode,
        stdNullPars
    );

    ////

    #define TMP_MACRO(Type, args) \
        TMP_MACRO2(Type, PREP_ARG2_0 args, PREP_ARG2_1 args)

    #define TMP_MACRO2(Type, funcName, colorSpace) \
        \
        stdbool funcName \
        ( \
            const GpuMatrix<const Type>& img, \
            float32 minVal, float32 maxVal, \
            const Point<float32>& upsampleFactor, \
            InterpType upsampleType, \
            const Point<Space>& upsampleSize, \
            BorderMode borderMode, \
            const ImgOutputHint& hint, \
            stdNullPars \
        ) \
        { \
            return addColorImageFunc(img, minVal, maxVal, upsampleFactor, upsampleType, upsampleSize, borderMode, hint, colorSpace, stdPassThru); \
        }

    IMAGE_CONSOLE_FOREACH_X4_TYPE(TMP_MACRO, (addYuvColorImage, ColorYuv))
    IMAGE_CONSOLE_FOREACH_X4_TYPE(TMP_MACRO, (addRgbColorImage, ColorRgb))

    #undef TMP_MACRO

    //----------------------------------------------------------------
    //
    // Creation
    //
    //----------------------------------------------------------------

public:

    KIT_COMBINE4(Kit, GpuProcessKit, MsgLogsKit, UserPointKit, OutputLevelKit);

public:

    inline GpuImageConsoleThunk(GpuBaseConsole& baseConsole, float32 displayFactor, VectorDisplayMode vectorDisplayMode, const Kit& kit)
        : baseConsole(baseConsole), displayFactor(displayFactor), vectorDisplayMode(vectorDisplayMode), kit(kit) {}

private:

    GpuBaseConsole& baseConsole;

    float32 displayFactor;

    VectorDisplayMode const vectorDisplayMode;

    Kit const kit;

};

//----------------------------------------------------------------

}
