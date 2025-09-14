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

    void clear(stdPars(Kit))
        {CHECK(false);}

    void update(stdPars(Kit))
        {CHECK(false);}

    void addImageBgr(const GpuMatrixAP<const uint8_x4>& img, const ImgOutputHint& hint, stdPars(Kit))
        {CHECK(false);}

    void overlayClear(stdPars(Kit))
        {CHECK(false);}

    void overlaySetImageBgr(const Point<Space>& size, const GpuImageProviderBgr32& img, const ImgOutputHint& hint, stdPars(Kit))
        {CHECK(false);}

    void overlaySetImageFake(stdPars(Kit))
        {CHECK(false);}

    void overlayUpdate(stdPars(Kit))
        {CHECK(false);}

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

    void clear(stdPars(Kit))
        {baseConsole.clear(stdPassThru);}

    void update(stdPars(Kit))
        {baseConsole.update(stdPassThru);}

    void addImageBgr(const GpuMatrixAP<const uint8_x4>& img, const ImgOutputHint& hint, stdPars(Kit))
        {baseConsole.addImageBgr(img, hint, stdPassThru);}

    void overlayClear(stdPars(Kit))
        {baseConsole.overlayClear(stdPassThru);}

    void overlaySetImageBgr(const Point<Space>& size, const GpuImageProviderBgr32& img, const ImgOutputHint& hint, stdPars(Kit))
        {baseConsole.overlaySetImageBgr(size, img, hint, stdPassThru);}

    void overlaySetImageFake(stdPars(Kit))
        {baseConsole.overlaySetImageFake(stdPassThru);}

    void overlayUpdate(stdPars(Kit))
        {baseConsole.overlayUpdate(stdPassThru);}

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
        void addMatrixFunc \
        ( \
            const GpuMatrixAP<const Type>& img, \
            float32 minVal, float32 maxVal, \
            const Point<float32>& upsampleFactor, \
            InterpType upsampleType, \
            const Point<Space>& upsampleSize, \
            BorderMode borderMode, \
            const ImgOutputHint& hint, \
            stdParsNull \
        ) \
        { \
            addMatrixExImpl(img, 0, minVal, maxVal, upsampleFactor, upsampleType, upsampleSize, borderMode, hint, stdPassThru); \
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
        void addMatrixChanFunc \
        ( \
            const GpuMatrixAP<const Type>& img, \
            int channel, \
            float32 minVal, float32 maxVal, \
            const Point<float32>& upsampleFactor, \
            InterpType upsampleType, \
            const Point<Space>& upsampleSize, \
            BorderMode borderMode, \
            const ImgOutputHint& hint, \
            stdParsNull \
        ) \
        { \
            addMatrixExImpl(img, channel, minVal, maxVal, upsampleFactor, upsampleType, upsampleSize, borderMode, hint, stdPassThru); \
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
    void addMatrixExImpl
    (
        const GpuMatrixAP<const Type>& img,
        int channel,
        float32 minVal, float32 maxVal,
        const Point<float32>& upsampleFactor,
        InterpType upsampleType,
        const Point<Space>& upsampleSize,
        BorderMode borderMode,
        const ImgOutputHint& hint,
        stdParsNull
    );

    //----------------------------------------------------------------
    //
    // addVectorImage
    //
    //----------------------------------------------------------------

public:

    template <typename Vector>
    void addVectorImageGeneric
    (
        const GpuMatrix<const Vector>& image,
        float32 maxVector,
        const Point<float32>& upsampleFactor,
        InterpType upsampleType,
        const Point<Space>& upsampleSize,
        BorderMode borderMode,
        const ImgOutputHint& hint,
        stdParsNull
    );

    #define TMP_MACRO(Vector, o) \
        \
        void addVectorImageFunc \
        ( \
            const GpuMatrix<const Vector>& image, \
            float32 maxVector, \
            const Point<float32>& upsampleFactor, \
            InterpType upsampleType, \
            const Point<Space>& upsampleSize, \
            BorderMode borderMode, \
            const ImgOutputHint& hint, \
            stdParsNull \
        ) \
        { \
            addVectorImageGeneric<Vector>(image, maxVector, upsampleFactor, upsampleType, upsampleSize, borderMode, hint, stdPassThru); \
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
    void addYuvImage420Func
    (
        const GpuPackedYuv<const Type>& image,
        const ImgOutputHint& hint,
        stdParsNull
    );

    ////

    #define TMP_MACRO(Type, _) \
        \
        void addYuvImage420 \
        ( \
            const GpuPackedYuv<const Type>& image, \
            const ImgOutputHint& hint, \
            stdParsNull \
        ) \
        { \
            addYuvImage420Func(image, hint, stdPassNullThru); \
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
    void addColorImageFunc
    (
        const GpuMatrixAP<const Type>& img,
        float32 minVal, float32 maxVal,
        const Point<float32>& upsampleFactor,
        InterpType upsampleType,
        const Point<Space>& upsampleSize,
        BorderMode borderMode,
        const ImgOutputHint& hint,
        ColorMode colorMode,
        stdParsNull
    );

    ////

    #define TMP_MACRO(Type, args) \
        TMP_MACRO2(Type, PREP_ARG2_0 args, PREP_ARG2_1 args)

    #define TMP_MACRO2(Type, funcName, colorSpace) \
        \
        void funcName \
        ( \
            const GpuMatrixAP<const Type>& img, \
            float32 minVal, float32 maxVal, \
            const Point<float32>& upsampleFactor, \
            InterpType upsampleType, \
            const Point<Space>& upsampleSize, \
            BorderMode borderMode, \
            const ImgOutputHint& hint, \
            stdParsNull \
        ) \
        { \
            addColorImageFunc(img, minVal, maxVal, upsampleFactor, upsampleType, upsampleSize, borderMode, hint, colorSpace, stdPassThru); \
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
