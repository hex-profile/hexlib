#pragma once

#include "errorLog/errorLog.h"
#include "gpuProcessKit.h"
#include "imageConsole/gpuImageConsole.h"
#include "imageConsole/imageConsoleModes.h"
#include "kits/moduleKit.h"
#include "kits/msgLogsKit.h"
#include "kits/userPointKit.h"

namespace gpuImageConsoleImpl {

//================================================================
//
// GpuBaseConsoleProhibitThunk
//
//================================================================

class GpuBaseConsoleProhibitThunk : public GpuBaseConsole
{

public:

    bool getTextEnabled()
        {return false;}

    void setTextEnabled(bool textEnabled)
        {}

    stdbool clear(stdPars(Kit))
        {CHECK(false); returnTrue;}

    stdbool update(stdPars(Kit))
        {CHECK(false); returnTrue;}

    stdbool addImageBgr(const GpuMatrix<const uint8_x4>& img, const ImgOutputHint& hint, stdPars(Kit))
        {CHECK(false); returnTrue;}

    stdbool overlayClear(stdPars(Kit))
        {CHECK(false); returnTrue;}

    stdbool overlaySetImageBgr(const Point<Space>& size, const GpuImageProviderBgr32& img, const ImgOutputHint& hint, stdPars(Kit))
        {CHECK(false); returnTrue;}

    stdbool overlaySetImageFake(stdPars(Kit))
        {CHECK(false); returnTrue;}

    stdbool overlayUpdate(stdPars(Kit))
        {CHECK(false); returnTrue;}

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

    //----------------------------------------------------------------
    //
    // Basic thunks.
    //
    //----------------------------------------------------------------

public:

    stdbool clear(stdPars(Kit))
        {return baseConsole.clear(stdPassThru);}

    stdbool update(stdPars(Kit))
        {return baseConsole.update(stdPassThru);}

    stdbool addImageBgr(const GpuMatrix<const uint8_x4>& img, const ImgOutputHint& hint, stdPars(Kit))
        {return baseConsole.addImageBgr(img, hint, stdPassThru);}

    stdbool overlayClear(stdPars(Kit))
        {return baseConsole.overlayClear(stdPassThru);}

    stdbool overlaySetImageBgr(const Point<Space>& size, const GpuImageProviderBgr32& img, const ImgOutputHint& hint, stdPars(Kit))
        {return baseConsole.overlaySetImageBgr(size, img, hint, stdPassThru);}

    stdbool overlaySetImageFake(stdPars(Kit))
        {return baseConsole.overlaySetImageFake(stdPassThru);}

    stdbool overlayUpdate(stdPars(Kit))
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

    using Kit = KitCombine<GpuProcessKit, MsgLogsKit, UserPointKit, VerbosityKit>;

public:

    inline GpuImageConsoleThunk
    (
        GpuBaseConsole& baseConsole,
        DisplayMode displayMode,
        VectorMode vectorMode,
        const Kit& kit
    )
        :
        baseConsole(baseConsole),
        displayMode(displayMode),
        vectorMode(vectorMode),
        kit(kit)
    {
    }

private:

    GpuBaseConsole& baseConsole;

    DisplayMode const displayMode;
    VectorMode const vectorMode;
    Kit const kit;

};

//----------------------------------------------------------------

}
