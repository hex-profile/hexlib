#pragma once

#include "imageConsole/gpuBaseConsole.h"
#include "numbers/float/floatBase.h"
#include "imageRead/borderMode.h"
#include "imageRead/interpType.h"
#include "data/gpuImageYuv.h"

//================================================================
//
// Supported types.
//
//================================================================

#define IMAGE_CONSOLE_FOREACH_SCALAR_TYPE(action, extra) \
    action(uint8, extra) \
    action(int8, extra) \
    action(uint16, extra) \
    action(int16, extra) \
    action(uint32, extra) \
    action(int32, extra) \
    action(float16, extra) \
    action(float32, extra) \

#define IMAGE_CONSOLE_FOREACH_X2_TYPE(action, extra) \
    action(uint8_x2, extra) \
    action(int8_x2, extra) \
    action(uint16_x2, extra) \
    action(int16_x2, extra) \
    action(uint32_x2, extra) \
    action(int32_x2, extra) \
    action(float16_x2, extra) \
    action(float32_x2, extra)

#define IMAGE_CONSOLE_FOREACH_X4_TYPE(action, extra) \
    action(uint8_x4, extra) \
    action(int8_x4, extra) \
    action(uint16_x4, extra) \
    action(int16_x4, extra) \
    action(uint32_x4, extra) \
    action(int32_x4, extra) \
    action(float16_x4, extra) \
    action(float32_x4, extra)

#define IMAGE_CONSOLE_FOREACH_VECTOR_TYPE(action, extra) \
    IMAGE_CONSOLE_FOREACH_X2_TYPE(action, extra) \
    IMAGE_CONSOLE_FOREACH_X4_TYPE(action, extra)

//================================================================
//
// GpuImageConsole
//
// Advanced image console interface: more types and ranged output.
//
//================================================================

struct GpuImageConsole : public GpuBaseConsole
{

    //----------------------------------------------------------------
    //
    // Add scalar image with upsampling and range scaling
    //
    //----------------------------------------------------------------

    #define TMP_MACRO(Type, _) \
        \
        virtual stdbool addMatrixFunc \
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
        =0;

    IMAGE_CONSOLE_FOREACH_SCALAR_TYPE(TMP_MACRO, _)

    #undef TMP_MACRO

    ////

    template <typename Type, typename Pitch>
    sysinline stdbool addMatrixEx
    (
        const GpuMatrix<Type, Pitch>& img,
        float32 minVal, float32 maxVal,
        const Point<float32>& upsampleFactor,
        InterpType upsampleType,
        const Point<Space>& upsampleSize,
        BorderMode borderMode,
        const ImgOutputHint& hint,
        stdParsNull
    )
    {
        return addMatrixFunc(makeConst(img), minVal, maxVal, upsampleFactor, upsampleType, upsampleSize, borderMode, hint, stdPassNullThru);
    }

    //----------------------------------------------------------------
    //
    // Add the selected scalar plane of 2X-vector or 4X-vector image
    //
    //----------------------------------------------------------------

    #define TMP_MACRO(Type, _) \
        \
        virtual stdbool addMatrixChanFunc \
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
        =0;

    IMAGE_CONSOLE_FOREACH_VECTOR_TYPE(TMP_MACRO, _)

    #undef TMP_MACRO

    ////

    template <typename Type, typename Pitch>
    sysinline stdbool addMatrixChan
    (
        const GpuMatrix<Type, Pitch>& img,
        int channel,
        float32 minVal, float32 maxVal,
        const Point<float32>& upsampleFactor,
        InterpType upsampleType,
        const Point<Space>& upsampleSize,
        BorderMode borderMode,
        const ImgOutputHint& hint,
        stdParsNull
    )
    {
        return addMatrixChanFunc(makeConst(img), channel, minVal, maxVal, upsampleFactor, upsampleType, upsampleSize, borderMode, hint, stdPassNullThru);
    }

    ////

    template <typename Type, typename Pitch>
    sysinline stdbool addMatrixChan
    (
        const GpuMatrix<Type, Pitch>& img,
        int channel,
        float32 minVal, float32 maxVal,
        const ImgOutputHint& hint,
        stdParsNull
    )
    {
        return addMatrixChanFunc(makeConst(img), channel, minVal, maxVal, point(1.f), INTERP_NONE, img.size(), BORDER_ZERO, hint, stdPassNullThru);
    }

    //----------------------------------------------------------------
    //
    // Output vector image
    //
    //----------------------------------------------------------------

    #define IMAGE_CONSOLE_FOREACH_VECTOR_IMAGE_TYPE(action, extra) \
        \
        action(float16_x2, extra) \
        action(float16_x4, extra) \
        \
        action(float32_x2, extra) \
        action(float32_x4, extra)

    ////

    #define TMP_MACRO(VectorType, o) \
        \
        virtual stdbool addVectorImageFunc \
        ( \
            const GpuMatrix<const VectorType>& image, \
            float32 maxVector, \
            const Point<float32>& upsampleFactor, \
            InterpType upsampleType, \
            const Point<Space>& upsampleSize, \
            BorderMode borderMode, \
            const ImgOutputHint& hint, \
            stdParsNull \
        ) \
        =0;

    IMAGE_CONSOLE_FOREACH_VECTOR_IMAGE_TYPE(TMP_MACRO, o)

    ////

    template <typename VectorType, typename Pitch>
    sysinline stdbool addVectorImage
    (
        const GpuMatrix<VectorType, Pitch>& image,
        float32 maxVector,
        const Point<float32>& upsampleFactor,
        InterpType upsampleType,
        const Point<Space>& upsampleSize,
        BorderMode borderMode,
        const ImgOutputHint& hint,
        stdParsNull
    )
    {
        return addVectorImageFunc(makeConst(image), maxVector, upsampleFactor, upsampleType, upsampleSize, borderMode, hint, stdPassNullThru);
    }

    template <typename VectorType, typename Pitch>
    sysinline stdbool addVectorImageSimple
    (
        const GpuMatrix<VectorType, Pitch>& image,
        float32 maxVector,
        const ImgOutputHint& hint,
        stdParsNull
    )
    {
        return addVectorImageFunc(makeConst(image), maxVector, point(1.f), INTERP_NONE, point(0), BORDER_ZERO, hint, stdPassNullThru);
    }

    #undef TMP_MACRO

    //----------------------------------------------------------------
    //
    // Output YUV color image
    //
    //----------------------------------------------------------------

    #define IMAGE_CONSOLE_FOREACH_YUV420_TYPE(action, params) \
        action(float16, params) \
        action(int8, params) \

    ////

    #define TMP_MACRO(Type, _) \
        \
        virtual stdbool addYuvImage420 \
        ( \
            const GpuPackedYuv<const Type>& image, \
            const ImgOutputHint& hint, \
            stdParsNull \
        ) \
        =0;

    IMAGE_CONSOLE_FOREACH_YUV420_TYPE(TMP_MACRO, _)

    #undef TMP_MACRO

    //----------------------------------------------------------------
    //
    // Add YUV color image
    //
    //----------------------------------------------------------------

    #define TMP_MACRO(Type, funcName) \
        \
        virtual stdbool funcName \
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
        =0;

    IMAGE_CONSOLE_FOREACH_X4_TYPE(TMP_MACRO, addYuvColorImage)
    IMAGE_CONSOLE_FOREACH_X4_TYPE(TMP_MACRO, addRgbColorImage)

    #undef TMP_MACRO

};

//================================================================
//
// GpuImageConsoleNull
//
//================================================================

struct GpuImageConsoleNull : public GpuImageConsole
{

    //----------------------------------------------------------------
    //
    // Basic functions
    //
    //----------------------------------------------------------------

    //
    // Basic control functions.
    //

    stdbool clear(stdPars(Kit)) {returnTrue;}
    stdbool update(stdPars(Kit)) {returnTrue;}

    //
    // Basic output interaces.
    //

    stdbool addImageBgr(const GpuMatrixAP<const uint8_x4>& img, const ImgOutputHint& hint, stdPars(Kit)) {returnTrue;}


    //
    // Video overlay output.
    //

    stdbool overlayClear(stdPars(Kit)) {returnTrue;}
    stdbool overlaySetImageBgr(const Point<Space>& size, const GpuImageProviderBgr32& img, const ImgOutputHint& hint, stdPars(Kit)) {returnTrue;}
    stdbool overlaySetImageFake(stdPars(Kit)) {returnTrue;}
    stdbool overlayUpdate(stdPars(Kit)) {returnTrue;}

    //
    // Text enabled.
    //

    bool getTextEnabled() {return false;}
    void setTextEnabled(bool textEnabled) {}

    //----------------------------------------------------------------
    //
    // Add scalar image with upsampling and range scaling
    //
    //----------------------------------------------------------------

    #define TMP_MACRO(Type, _) \
        \
        stdbool addMatrixFunc \
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
            returnTrue; \
        }

    IMAGE_CONSOLE_FOREACH_SCALAR_TYPE(TMP_MACRO, _)

    #undef TMP_MACRO

    //----------------------------------------------------------------
    //
    // Add the selected scalar plane of 2X-vector or 4X-vector image
    //
    //----------------------------------------------------------------

    #define TMP_MACRO(Type, _) \
        \
        stdbool addMatrixChanFunc \
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
            returnTrue; \
        }

    IMAGE_CONSOLE_FOREACH_VECTOR_TYPE(TMP_MACRO, _)

    #undef TMP_MACRO

    //----------------------------------------------------------------
    //
    // Output vector image
    //
    //----------------------------------------------------------------

    #define TMP_MACRO(VectorType, o) \
        \
        stdbool addVectorImageFunc \
        ( \
            const GpuMatrix<const VectorType>& image, \
            float32 maxVector, \
            const Point<float32>& upsampleFactor, \
            InterpType upsampleType, \
            const Point<Space>& upsampleSize, \
            BorderMode borderMode, \
            const ImgOutputHint& hint, \
            stdParsNull \
        ) \
        { \
            returnTrue; \
        }

    IMAGE_CONSOLE_FOREACH_VECTOR_IMAGE_TYPE(TMP_MACRO, o)

    #undef TMP_MACRO

    //----------------------------------------------------------------
    //
    // Output YUV color image
    //
    //----------------------------------------------------------------

    #define TMP_MACRO(Type, _) \
        \
        stdbool addYuvImage420(const GpuPackedYuv<const Type>& image, const ImgOutputHint& hint, stdParsNull) \
            {returnTrue;}

    IMAGE_CONSOLE_FOREACH_YUV420_TYPE(TMP_MACRO, _)

    #undef TMP_MACRO

    //----------------------------------------------------------------
    //
    // addYuvColorImage
    //
    //----------------------------------------------------------------

    #define TMP_MACRO(Type, funcName) \
        \
        stdbool funcName \
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
            returnTrue; \
        }

    IMAGE_CONSOLE_FOREACH_X4_TYPE(TMP_MACRO, addYuvColorImage)
    IMAGE_CONSOLE_FOREACH_X4_TYPE(TMP_MACRO, addRgbColorImage)

    #undef TMP_MACRO

};
