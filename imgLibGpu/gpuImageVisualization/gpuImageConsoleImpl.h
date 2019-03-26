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

    bool clear(stdNullPars)
        {stdBegin; REQUIRE(false); stdEnd;}

    bool update(stdNullPars)
        {stdBegin; REQUIRE(false); stdEnd;}

    bool addImage(const GpuMatrix<const uint8>& img, const ImgOutputHint& hint, stdNullPars)
        {stdBegin; REQUIRE(false); stdEnd;}

    bool addImageBgr(const GpuMatrix<const uint8_x4>& img, const ImgOutputHint& hint, stdNullPars)
        {stdBegin; REQUIRE(false); stdEnd;}

    bool overlaySetImageBgr(const Point<Space>& size, const GpuImageProviderBgr32& img, const ImgOutputHint& hint, stdNullPars)
        {stdBegin; REQUIRE(false); stdEnd;}

    bool overlaySetFakeImage(stdNullPars)
        {stdBegin; REQUIRE(false); stdEnd;}

    bool overlayUpdate(stdNullPars)
        {stdBegin; REQUIRE(false); stdEnd;}

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

    bool clear(stdNullPars)
        {return true;}

    bool update(stdNullPars)
        {return true;}

    bool addImage(const GpuMatrix<const uint8>& img, const ImgOutputHint& hint, stdNullPars)
        {return true;}

    bool addImageBgr(const GpuMatrix<const uint8_x4>& img, const ImgOutputHint& hint, stdNullPars)
        {return true;}

    bool overlaySetImageBgr(const Point<Space>& size, const GpuImageProviderBgr32& img, const ImgOutputHint& hint, stdNullPars)
        {return true;}

    bool overlaySetFakeImage(stdNullPars)
        {return true;}

    bool overlayUpdate(stdNullPars)
        {return true;}

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

    bool clear(stdNullPars)
        {return baseConsole.clear(stdPassThru);}

    bool update(stdNullPars)
        {return baseConsole.update(stdPassThru);}

    bool addImage(const GpuMatrix<const uint8>& img, const ImgOutputHint& hint, stdNullPars)
        {return baseConsole.addImage(img, hint, stdPassThru);}

    bool addImageBgr(const GpuMatrix<const uint8_x4>& img, const ImgOutputHint& hint, stdNullPars)
        {return baseConsole.addImageBgr(img, hint, stdPassThru);}

    bool overlaySetImageBgr(const Point<Space>& size, const GpuImageProviderBgr32& img, const ImgOutputHint& hint, stdNullPars)
        {return baseConsole.overlaySetImageBgr(size, img, hint, stdPassThru);}

    bool overlaySetFakeImage(stdNullPars)
        {return baseConsole.overlaySetFakeImage(stdPassThru);}

    bool overlayUpdate(stdNullPars)
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
        bool addMatrixFunc \
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
        bool addMatrixChanFunc \
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
    bool addMatrixExImpl
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
    bool addVectorImageGeneric
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
        bool addVectorImageFunc \
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
    bool addYuvImage420Func
    (
        const GpuImageYuv<const Type>& image,
        const ImgOutputHint& hint,
        stdNullPars
    );

    ////

    #define TMP_MACRO(Type, _) \
        \
        bool addYuvImage420 \
        ( \
            const GpuImageYuv<const Type>& image, \
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
    bool addColorImageFunc
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
        bool funcName \
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