#pragma once

#include "numbers/float/floatBase.h"
#include "stdFunc/stdFunc.h"
#include "imageRead/borderMode.h"
#include "imageRead/interpType.h"
#include "data/gpuImageYuv.h"
#include "data/gpuMatrix.h"
#include "vectorTypes/vectorBase.h"
#include "imageConsole/imageConsole.h"
#include "imageConsole/gpuImageConsoleKit.h"

//================================================================
//
// GpuImageProviderBgr32
//
//================================================================

struct GpuImageProviderBgr32
{
    virtual bool saveImage(const GpuMatrix<uint8_x4>& dest, stdNullPars) const =0;
};

//================================================================
//
// GpuBaseConsole
//
// Abstract interface of image output console taking GPU images.
//
//================================================================

struct GpuBaseConsole
{

    //
    // Basic control functions.
    //

    virtual bool clear(stdNullPars) =0;
    virtual bool update(stdNullPars) =0;

    //
    // Basic output interaces.
    //

    virtual bool addImageBgr(const GpuMatrix<const uint8_x4>& img, const ImgOutputHint& hint, stdNullPars) =0;

    //
    // Video overlay output.
    //

    virtual bool overlaySetImageBgr(const Point<Space>& size, const GpuImageProviderBgr32& img, const ImgOutputHint& hint, stdNullPars) =0;
    virtual bool overlaySetFakeImage(stdNullPars) =0;
    virtual bool overlayUpdate(stdNullPars) =0;

    //
    //
    //

    virtual bool getTextEnabled() =0;
    virtual void setTextEnabled(bool textEnabled) =0;

};

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

    #define IMAGE_CONSOLE_FOREACH_SCALAR_TYPE(action, extra) \
        \
        action(uint8, extra) \
        action(int8, extra) \
        action(uint16, extra) \
        action(int16, extra) \
        action(uint32, extra) \
        action(int32, extra) \
        action(float16, extra) \
        action(float32, extra) \

    ////

    #define TMP_MACRO(Type, _) \
        \
        virtual bool addMatrixFunc \
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
        =0;

    IMAGE_CONSOLE_FOREACH_SCALAR_TYPE(TMP_MACRO, _)

    #undef TMP_MACRO

    ////

    template <typename Type>
    inline bool addMatrixEx
    (
        const GpuMatrix<Type>& img,
        float32 minVal, float32 maxVal,
        const Point<float32>& upsampleFactor,
        InterpType upsampleType,
        const Point<Space>& upsampleSize,
        BorderMode borderMode,
        const ImgOutputHint& hint,
        stdNullPars
    )
    {
        return addMatrixFunc(makeConst(img), minVal, maxVal, upsampleFactor, upsampleType, upsampleSize, borderMode, hint, stdNullPassThru);
    }

    ////

    template <typename Type>
    inline bool addMatrix(const GpuMatrix<Type>& img, float32 minVal, float32 maxVal, const ImgOutputHint& hint, stdNullPars)
    {
        return addMatrixFunc(makeConst(img), minVal, maxVal, point(1.f), INTERP_NONE, img.size(), BORDER_ZERO, hint, stdNullPassThru);
    }

    //----------------------------------------------------------------
    //
    // Vector types
    //
    //----------------------------------------------------------------

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

    //----------------------------------------------------------------
    //
    // Add the selected scalar plane of 2X-vector or 4X-vector image
    //
    //----------------------------------------------------------------

    #define TMP_MACRO(Type, _) \
        \
        virtual bool addMatrixChanFunc \
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
        =0;

    IMAGE_CONSOLE_FOREACH_VECTOR_TYPE(TMP_MACRO, _)

    #undef TMP_MACRO

    ////

    template <typename Type>
    inline bool addMatrixChan
    (
        const GpuMatrix<Type>& img,
        int channel,
        float32 minVal, float32 maxVal,
        const Point<float32>& upsampleFactor,
        InterpType upsampleType,
        const Point<Space>& upsampleSize,
        BorderMode borderMode,
        const ImgOutputHint& hint,
        stdNullPars
    )
    {
        return addMatrixChanFunc(makeConst(img), channel, minVal, maxVal, upsampleFactor, upsampleType, upsampleSize, borderMode, hint, stdNullPassThru);
    }

    ////

    template <typename Type>
    inline bool addMatrixChan
    (
        const GpuMatrix<Type>& img,
        int channel,
        float32 minVal, float32 maxVal,
        const ImgOutputHint& hint,
        stdNullPars
    )
    {
        return addMatrixChanFunc(makeConst(img), channel, minVal, maxVal, point(1.f), INTERP_NONE, img.size(), BORDER_ZERO, hint, stdNullPassThru);
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
        virtual bool addVectorImageFunc \
        ( \
            const GpuMatrix<const VectorType>& image, \
            float32 maxVector, \
            const Point<float32>& upsampleFactor, \
            InterpType upsampleType, \
            const Point<Space>& upsampleSize, \
            BorderMode borderMode, \
            const ImgOutputHint& hint, \
            stdNullPars \
        ) \
        =0; \

    IMAGE_CONSOLE_FOREACH_VECTOR_IMAGE_TYPE(TMP_MACRO, o)

    ////

    template <typename VectorType>
    inline bool addVectorImage
    (
        const GpuMatrix<VectorType>& image,
        float32 maxVector,
        const Point<float32>& upsampleFactor,
        InterpType upsampleType,
        const Point<Space>& upsampleSize,
        BorderMode borderMode,
        const ImgOutputHint& hint,
        stdNullPars
    )
    {
        return addVectorImageFunc(makeConst(image), maxVector, upsampleFactor, upsampleType, upsampleSize, borderMode, hint, stdNullPassThru);
    }

    template <typename VectorType>
    inline bool addVectorImageSimple(const GpuMatrix<VectorType>& image, float32 maxVector, const ImgOutputHint& hint, stdNullPars)
    {
        return addVectorImageFunc(makeConst(image), maxVector, point(1.f), INTERP_NONE, point(0), BORDER_ZERO, hint, stdNullPassThru);
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
        virtual bool addYuvImage420 \
        ( \
            const GpuImageYuv<const Type>& image, \
            const ImgOutputHint& hint, \
            stdNullPars \
        ) \
        =0; \

    IMAGE_CONSOLE_FOREACH_YUV420_TYPE(TMP_MACRO, _)

    #undef TMP_MACRO

    //----------------------------------------------------------------
    //
    // Add YUV color image
    //
    //----------------------------------------------------------------

    #define TMP_MACRO(Type, funcName) \
        \
        virtual bool funcName \
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

    bool clear(stdNullPars) {return true;}
    bool update(stdNullPars) {return true;}

    //
    // Basic output interaces.
    //

    bool addImage(const GpuMatrix<const uint8>& img, const ImgOutputHint& hint, stdNullPars) {return true;}
    bool addImageBgr(const GpuMatrix<const uint8_x4>& img, const ImgOutputHint& hint, stdNullPars) {return true;}

    //
    // Video overlay output.
    //

    bool overlaySetImageBgr(const Point<Space>& size, const GpuImageProviderBgr32& img, const ImgOutputHint& hint, stdNullPars) {return true;}
    bool overlaySetFakeImage(stdNullPars) {return true;}
    bool overlayUpdate(stdNullPars) {return true;}

    //
    //
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
            return true; \
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
            return true; \
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
        bool addVectorImageFunc \
        ( \
            const GpuMatrix<const VectorType>& image, \
            float32 maxVector, \
            const Point<float32>& upsampleFactor, \
            InterpType upsampleType, \
            const Point<Space>& upsampleSize, \
            BorderMode borderMode, \
            const ImgOutputHint& hint, \
            stdNullPars \
        ) \
        { \
            return true; \
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
        bool addYuvImage420(const GpuImageYuv<const Type>& image, const ImgOutputHint& hint, stdNullPars) \
            {return true;}

    IMAGE_CONSOLE_FOREACH_YUV420_TYPE(TMP_MACRO, _)

    #undef TMP_MACRO

    //----------------------------------------------------------------
    //
    // addYuvColorImage
    //
    //----------------------------------------------------------------

    #define TMP_MACRO(Type, funcName) \
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
            return true; \
        }

    IMAGE_CONSOLE_FOREACH_X4_TYPE(TMP_MACRO, addYuvColorImage)
    IMAGE_CONSOLE_FOREACH_X4_TYPE(TMP_MACRO, addRgbColorImage)

    #undef TMP_MACRO

};
