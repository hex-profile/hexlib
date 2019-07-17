#if HOSTCODE
#include "gpuImageConsoleImpl.h"
#endif

#include "kit/kit.h"
#include "data/gpuMatrix.h"
#include "gpuDevice/gpuDevice.h"
#include "gpuDevice/loadstore/loadNorm.h"
#include "gpuDevice/loadstore/storeNorm.h"
#include "gpuSupport/gpuMixedCode.h"
#include "gpuSupport/gpuTool.h"
#include "vectorTypes/vectorOperations.h"
#include "vectorTypes/vectorType.h"
#include "convertYuv420/convertYuvRgbFunc.h"
#include "flipMatrix.h"
#include "readInterpolate/gpuTexCubic.h"
#include "types/lt/ltType.h"

#if HOSTCODE
#include "errorLog/errorLog.h"
#include "numbers/divRound.h"
#include "numbers/float/floatType.h"
#include "userOutput/paramMsg.h"
#include "dataAlloc/gpuMatrixMemory.h"
#include "gpuAppliedApi/gpuAppliedApi.h"
#include "gpuSupport/gpuTexTools.h"
#include "vectorTypes/half/halfType.h"
#include "convertYuv420/convertYuv420ToBgr.h"
#include "copyMatrixAsArray.h"
#include "numbers/mathIntrinsics.h"
#include "dataAlloc/matrixMemory.h"
#include "dataAlloc/matrixMemory.inl"
#include "gpuImageVisualization/visualizeVectorImage/visualizeVectorImage.h"
#include "userOutput/printMsgEx.h"
#include "imageRead/positionTools.h"
#include "diagTools/readGpuElement.h"
#endif

namespace gpuImageConsoleImpl {

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Scalar visualization
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// visualizeScalarThreadCountX
//
//================================================================

static const Space visualizeScalarThreadCountX = 64;

//================================================================
//
// VisualizeScalar
//
//================================================================

template <typename Dst>
struct VisualizeScalar
{
    LinearTransform<Point<float32>> coordBackTransform;
    LinearTransform<float32> valueTransform;
    Point<float32> srcTexstep;
    GpuMatrix<Dst> dst;
};

//================================================================
//
// srcSampler
//
//================================================================

devDefineSampler(visualizeScalarSampler1, DevSampler2D, DevSamplerFloat, 1)
devDefineSampler(visualizeScalarSampler2, DevSampler2D, DevSamplerFloat, 2)
devDefineSampler(visualizeScalarSampler4, DevSampler2D, DevSamplerFloat, 4)

//================================================================
//
// visualizeScalar
//
//================================================================

#define VISUALIZE_SCALAR_KERNEL_BODY(name, readTerm, Dst) \
    \
    devDefineKernel(name, VisualizeScalar<Dst>, o) \
    { \
        MATRIX_EXPOSE_EX(o.dst, dst); \
        \
        Space X = devGroupX * visualizeScalarThreadCountX + devThreadX; \
        Space Y = devGroupY; \
        \
        if_not (X < dstSizeX) return; \
        \
        float32 Xs = X + 0.5f; \
        float32 Ys = Y + 0.5f; \
        \
        float32 srcXs = Xs * o.coordBackTransform.C1.X + o.coordBackTransform.C0.X; \
        float32 srcYs = Ys * o.coordBackTransform.C1.Y + o.coordBackTransform.C0.Y; \
        \
        float32 srcValue = readTerm; \
        \
        float32 dstValue = o.valueTransform.C1 * srcValue + o.valueTransform.C0; \
        \
        using DstFloat = VECTOR_REBASE_(Dst, float32); \
        DstFloat storeValue = vectorExtend<DstFloat>(dstValue); \
        \
        storeNorm(MATRIX_POINTER(dst, X, Y), storeValue); \
    }

//----------------------------------------------------------------

#define VISUALIZE_SCALAR_KERNEL_INSTANCES(name, readTerm) \
    VISUALIZE_SCALAR_KERNEL_BODY(name##_uint8_x4, readTerm, uint8_x4) \

#define VISUALIZE_SCALAR_KERNEL_SELECTOR(name) \
    template <typename Dst> inline const GpuKernelLink& name(); \
    template <> inline const GpuKernelLink& name<uint8_x4>() {return name##_uint8_x4;} \

#define VISUALIZE_SCALAR_KERNEL_ALL(name, readTerm) \
    DEV_ONLY(VISUALIZE_SCALAR_KERNEL_INSTANCES(name, readTerm)) \
    HOST_ONLY(VISUALIZE_SCALAR_KERNEL_SELECTOR(name))

//----------------------------------------------------------------

VISUALIZE_SCALAR_KERNEL_ALL(visualizeScalar1, devTex2D(visualizeScalarSampler1, srcXs * o.srcTexstep.X, srcYs * o.srcTexstep.Y))

VISUALIZE_SCALAR_KERNEL_ALL(visualizeScalar2_0, devTex2D(visualizeScalarSampler2, srcXs * o.srcTexstep.X, srcYs * o.srcTexstep.Y).x)
VISUALIZE_SCALAR_KERNEL_ALL(visualizeScalar2_1, devTex2D(visualizeScalarSampler2, srcXs * o.srcTexstep.X, srcYs * o.srcTexstep.Y).y)

VISUALIZE_SCALAR_KERNEL_ALL(visualizeScalar4_0, devTex2D(visualizeScalarSampler4, srcXs * o.srcTexstep.X, srcYs * o.srcTexstep.Y).x)
VISUALIZE_SCALAR_KERNEL_ALL(visualizeScalar4_1, devTex2D(visualizeScalarSampler4, srcXs * o.srcTexstep.X, srcYs * o.srcTexstep.Y).y)
VISUALIZE_SCALAR_KERNEL_ALL(visualizeScalar4_2, devTex2D(visualizeScalarSampler4, srcXs * o.srcTexstep.X, srcYs * o.srcTexstep.Y).z)
VISUALIZE_SCALAR_KERNEL_ALL(visualizeScalar4_3, devTex2D(visualizeScalarSampler4, srcXs * o.srcTexstep.X, srcYs * o.srcTexstep.Y).w)

//----------------------------------------------------------------

VISUALIZE_SCALAR_KERNEL_ALL(visualizeScalarCubic1, texCubic2D(visualizeScalarSampler1, point(srcXs, srcYs), o.srcTexstep))

VISUALIZE_SCALAR_KERNEL_ALL(visualizeScalarCubic2_0, texCubic2D(visualizeScalarSampler2, point(srcXs, srcYs), o.srcTexstep).x)
VISUALIZE_SCALAR_KERNEL_ALL(visualizeScalarCubic2_1, texCubic2D(visualizeScalarSampler2, point(srcXs, srcYs), o.srcTexstep).y)

VISUALIZE_SCALAR_KERNEL_ALL(visualizeScalarCubic4_0, texCubic2D(visualizeScalarSampler4, point(srcXs, srcYs), o.srcTexstep).x)
VISUALIZE_SCALAR_KERNEL_ALL(visualizeScalarCubic4_1, texCubic2D(visualizeScalarSampler4, point(srcXs, srcYs), o.srcTexstep).y)
VISUALIZE_SCALAR_KERNEL_ALL(visualizeScalarCubic4_2, texCubic2D(visualizeScalarSampler4, point(srcXs, srcYs), o.srcTexstep).z)
VISUALIZE_SCALAR_KERNEL_ALL(visualizeScalarCubic4_3, texCubic2D(visualizeScalarSampler4, point(srcXs, srcYs), o.srcTexstep).w)

//================================================================
//
// visualizeScalarMatrix
//
//================================================================

#if HOSTCODE

template <typename Type, typename Dst>
stdbool visualizeScalarMatrix
(
    const GpuMatrix<const Type>& src,
    const LinearTransform<Point<float32>>& coordBackTransform,
    int channel,
    const LinearTransform<float32>& valueTransform,
    InterpType upsampleType,
    BorderMode borderMode,
    const GpuMatrix<Dst>& result,
    stdPars(GpuProcessKit)
)
{
    const int vectorRank = VectorTypeRank<Type>::val;
    COMPILE_ASSERT(vectorRank == 1 || vectorRank == 2 || vectorRank == 4);

    if (kit.dataProcessing)
    {
        const GpuSamplerLink* sampler = 0;
        if (vectorRank == 1) sampler = &visualizeScalarSampler1;
        if (vectorRank == 2) sampler = &visualizeScalarSampler2;
        if (vectorRank == 4) sampler = &visualizeScalarSampler4;
        REQUIRE(sampler != 0);

        require
        (
            kit.gpuSamplerSetting.setSamplerImage
            (
                *sampler,
                src,
                borderMode,
                upsampleType == INTERP_LINEAR,
                true,
                true,
                stdPass
            )
        );
    }

    ////

    if (kit.dataProcessing)
    {

        LinearTransform<float32> usedValueTransform = valueTransform;

        if (TYPE_IS_BUILTIN_INT(VECTOR_BASE(Type)))
            usedValueTransform.C1 *= convertFloat32(typeMax<VECTOR_BASE(Type)>());

        ////

        REQUIRE(channel >= 0 && channel < vectorRank);

        ////

        const GpuKernelLink* kernelLink = 0;

        if (upsampleType != INTERP_CUBIC)
        {
            if (vectorRank == 1 && channel == 0) kernelLink = &visualizeScalar1<Dst>();

            if (vectorRank == 2 && channel == 0) kernelLink = &visualizeScalar2_0<Dst>();
            if (vectorRank == 2 && channel == 1) kernelLink = &visualizeScalar2_1<Dst>();

            if (vectorRank == 4 && channel == 0) kernelLink = &visualizeScalar4_0<Dst>();
            if (vectorRank == 4 && channel == 1) kernelLink = &visualizeScalar4_1<Dst>();
            if (vectorRank == 4 && channel == 2) kernelLink = &visualizeScalar4_2<Dst>();
            if (vectorRank == 4 && channel == 3) kernelLink = &visualizeScalar4_3<Dst>();
        }
        else
        {
            if (vectorRank == 1 && channel == 0) kernelLink = &visualizeScalarCubic1<Dst>();

            if (vectorRank == 2 && channel == 0) kernelLink = &visualizeScalarCubic2_0<Dst>();
            if (vectorRank == 2 && channel == 1) kernelLink = &visualizeScalarCubic2_1<Dst>();

            if (vectorRank == 4 && channel == 0) kernelLink = &visualizeScalarCubic4_0<Dst>();
            if (vectorRank == 4 && channel == 1) kernelLink = &visualizeScalarCubic4_1<Dst>();
            if (vectorRank == 4 && channel == 2) kernelLink = &visualizeScalarCubic4_2<Dst>();
            if (vectorRank == 4 && channel == 3) kernelLink = &visualizeScalarCubic4_3<Dst>();
        }

        REQUIRE(kernelLink != 0);

        ////

        Point<float32> srcTexstep = 1.f / convertFloat32(clampMin(src.size(), 1));

        require
        (
            kit.gpuKernelCalling.callKernel
            (
                divUpNonneg(result.size(), point(visualizeScalarThreadCountX, 1)),
                point(visualizeScalarThreadCountX, 1),
                areaOf(result),
                *kernelLink,
                VisualizeScalar<Dst>{coordBackTransform, usedValueTransform, srcTexstep, result},
                kit.gpuCurrentStream,
                stdPass
            )
        );
    }

    returnTrue;
}

#endif

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Upconvert value
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// upconvertValueThreadCountX
//
//================================================================

static const Space upconvertValueThreadCountX = 64;

//================================================================
//
// UpconvertValue
//
//================================================================

template <typename Dst>
struct UpconvertValue
{
    LinearTransform<Point<float32>> coordBackTransform;
    LinearTransform<float32> valueTransform;
    Point<float32> srcTexstep;
    GpuMatrix<Dst> dst;
};

//================================================================
//
// srcSampler
//
//================================================================

devDefineSampler(upconvertValueSampler1, DevSampler2D, DevSamplerFloat, 1)
devDefineSampler(upconvertValueSampler2, DevSampler2D, DevSamplerFloat, 2)
devDefineSampler(upconvertValueSampler4, DevSampler2D, DevSamplerFloat, 4)

//================================================================
//
// upconvertValue
//
//================================================================

#define UPCONVERT_VALUE_KERNEL_BODY(name, readTerm, Dst) \
    \
    devDefineKernel(name, UpconvertValue<Dst>, o) \
    { \
        MATRIX_EXPOSE_EX(o.dst, dst); \
        \
        Space X = devGroupX * upconvertValueThreadCountX + devThreadX; \
        Space Y = devGroupY; \
        \
        if_not (X < dstSizeX) return; \
        \
        float32 Xs = X + 0.5f; \
        float32 Ys = Y + 0.5f; \
        \
        float32 srcXs = Xs * o.coordBackTransform.C1.X + o.coordBackTransform.C0.X; \
        float32 srcYs = Ys * o.coordBackTransform.C1.Y + o.coordBackTransform.C0.Y; \
        \
        using DstFloat = VECTOR_REBASE_(Dst, float32); \
        DstFloat srcValue = readTerm; \
        \
        DstFloat dstValue = o.valueTransform.C1 * srcValue + o.valueTransform.C0; \
        \
        storeNorm(MATRIX_POINTER(dst, X, Y), dstValue); \
    }

//----------------------------------------------------------------

#define UPCONVERT_VALUE_KERNEL_INSTANCES(name, readTerm) \
    UPCONVERT_VALUE_KERNEL_BODY(name##_float16, readTerm(upconvertValueSampler1), float16) \
    UPCONVERT_VALUE_KERNEL_BODY(name##_float16_x2, readTerm(upconvertValueSampler2), float16_x2) \
    UPCONVERT_VALUE_KERNEL_BODY(name##_float16_x4, readTerm(upconvertValueSampler4), float16_x4) \
    UPCONVERT_VALUE_KERNEL_BODY(name##_uint8_x4, readTerm(upconvertValueSampler4), uint8_x4) \

#define UPCONVERT_VALUE_KERNEL_SELECTOR(name) \
    template <typename Dst> inline const GpuKernelLink& name(); \
    template <> inline const GpuKernelLink& name<float16>() {return name##_float16;} \
    template <> inline const GpuKernelLink& name<float16_x2>() {return name##_float16_x2;} \
    template <> inline const GpuKernelLink& name<float16_x4>() {return name##_float16_x4;} \
    template <> inline const GpuKernelLink& name<uint8_x4>() {return name##_uint8_x4;}

#define UPCONVERT_VALUE_KERNEL_ALL(name, readTerm) \
    DEV_ONLY(UPCONVERT_VALUE_KERNEL_INSTANCES(name, readTerm)) \
    HOST_ONLY(UPCONVERT_VALUE_KERNEL_SELECTOR(name))

//----------------------------------------------------------------

#define UPCONVERT_VALUE_READ_NORMAL(sampler) \
    devTex2D(sampler, srcXs * o.srcTexstep.X, srcYs * o.srcTexstep.Y)

UPCONVERT_VALUE_KERNEL_ALL(upconvertValue, UPCONVERT_VALUE_READ_NORMAL)

////

#define UPCONVERT_VALUE_READ_CUBIC(sampler) \
    texCubic2D(sampler, point(srcXs, srcYs), o.srcTexstep)

UPCONVERT_VALUE_KERNEL_ALL(upconvertValueCubic, UPCONVERT_VALUE_READ_CUBIC)

//================================================================
//
// upconvertValueMatrix
//
//================================================================

#if HOSTCODE

template <typename Type, typename Dst>
stdbool upconvertValueMatrix
(
    const GpuMatrix<const Type>& src,
    const LinearTransform<Point<float32>>& coordBackTransform,
    const LinearTransform<float32>& valueTransform,
    InterpType upsampleType,
    BorderMode borderMode,
    const GpuMatrix<Dst>& result,
    stdPars(GpuProcessKit)
)
{
    const int vectorRank = VectorTypeRank<Type>::val;
    COMPILE_ASSERT(vectorRank == 1 || vectorRank == 2 || vectorRank == 4);

    if (kit.dataProcessing)
    {
        const GpuSamplerLink* sampler = 0;
        if (vectorRank == 1) sampler = &upconvertValueSampler1;
        if (vectorRank == 2) sampler = &upconvertValueSampler2;
        if (vectorRank == 4) sampler = &upconvertValueSampler4;
        REQUIRE(sampler != 0);

        require
        (
            kit.gpuSamplerSetting.setSamplerImage
            (
                *sampler,
                src,
                borderMode,
                upsampleType == INTERP_LINEAR,
                true,
                true,
                stdPass
            )
        );
    }

    ////

    if (kit.dataProcessing)
    {

        LinearTransform<float32> usedValueTransform = valueTransform;

        if (TYPE_IS_BUILTIN_INT(VECTOR_BASE(Type)))
            usedValueTransform.C1 *= convertFloat32(typeMax<VECTOR_BASE(Type)>());

        ////

        const GpuKernelLink* kernelLink = 0;

        if (upsampleType != INTERP_CUBIC)
            kernelLink = &upconvertValue<Dst>();
        else
            kernelLink = &upconvertValueCubic<Dst>();

        REQUIRE(kernelLink != 0);

        ////

        Point<float32> srcTexstep = 1.f / convertFloat32(clampMin(src.size(), 1));

        require
        (
            kit.gpuKernelCalling.callKernel
            (
                divUpNonneg(result.size(), point(upconvertValueThreadCountX, 1)),
                point(upconvertValueThreadCountX, 1),
                areaOf(result),
                *kernelLink,
                UpconvertValue<Dst>{coordBackTransform, usedValueTransform, srcTexstep, result},
                kit.gpuCurrentStream,
                stdPass
            )
        );
    }

    returnTrue;
}

#endif

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// GpuImageConsoleThunk
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Host part
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

#if HOSTCODE

//================================================================
//
// getElementAtUserPoint
//
//================================================================

template <typename Type, typename Kit>
stdbool getElementAtUserPoint(const GpuMatrix<const Type>& image, const LinearTransform<Point<float32>>& transform, Point<Space>& resultIdx, Type& resultValue, stdPars(Kit))
{
    Point<float32> dstPos = convertFloat32(kit.userPoint.position) + 0.5f; // to space format
    Point<float32> srcPos = ltApply(dstPos, transform);
    Point<Space> srcIdx = convertToNearestIndex(srcPos);

    require(allv(image.size() >= 1));
    srcIdx = clampRange(srcIdx, point(0), image.size() - 1);

    resultIdx = srcIdx;
    require(readGpuElement(image, srcIdx, resultValue, stdPass));

    returnTrue;
}

//================================================================
//
// ScalarVisualizationParams
//
//================================================================

template <typename Type>
struct ScalarVisualizationParams
{
    GpuMatrix<const Type> img;
    int channel;
    LinearTransform<Point<float32>> coordBack;
    LinearTransform<float32> valueTransform;
    InterpType upsampleType;
    BorderMode borderMode;
    bool overlayCentering;
};

//================================================================
//
// ScalarVisualizationProvider
//
//================================================================

template <typename Type>
class ScalarVisualizationProvider : public GpuImageProviderBgr32, private ScalarVisualizationParams<Type>
{

    using Base = ScalarVisualizationParams<Type>;

    // Thanks GCC 5.4
    using Base::img;
    using Base::channel;
    using Base::coordBack;
    using Base::valueTransform;
    using Base::upsampleType;
    using Base::borderMode;
    using Base::overlayCentering;

public:

    ScalarVisualizationProvider(const ScalarVisualizationParams<Type>& params, const GpuProcessKit& kit)
        : Base(params), kit(kit) {}

    stdbool saveImage(const GpuMatrix<uint8_x4>& dest, stdNullPars) const;

private:

    GpuProcessKit kit;

};

//================================================================
//
// ScalarVisualizationProvider::saveImage
//
//================================================================

template <typename Type>
stdbool ScalarVisualizationProvider<Type>::saveImage(const GpuMatrix<uint8_x4>& dest, stdNullPars) const
{
    auto newCoordBack = coordBack;

    ////

    if (overlayCentering)
    {
        Point<float32> srcCenter = 0.5f * convertFloat32(img.size());
        Point<float32> dstCenter = 0.5f * convertFloat32(dest.size());

        newCoordBack.C0 = srcCenter - coordBack.C1 * dstCenter;

        if (allv(coordBack.C1 == point(1.f)))
            newCoordBack.C0 = convertFloat32(convertNearest<Space>(newCoordBack.C0));
    }

    ////

    require((visualizeScalarMatrix<Type, uint8_x4>(img, newCoordBack, channel, valueTransform, upsampleType, borderMode, dest, stdPass)));

    ////

    returnTrue;
}

//================================================================
//
// GpuImageConsoleThunk::addMatrixExImpl
//
//================================================================

template <typename Type>
stdbool GpuImageConsoleThunk::addMatrixExImpl
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
)
{
    REQUIRE(upsampleSize >= 0);

    Point<Space> outputSize = upsampleSize;

    if (allv(outputSize == point(0)))
        outputSize = convertUp<Space>(convertFloat32(img.size()) * upsampleFactor);

    //
    //
    //

    LinearTransform<float32> valueTransform = ltByTwoPoints(minVal, 0.f, maxVal, 1.f);
    if (minVal == maxVal) valueTransform = ltOutputZero<float32>();

    //
    //
    //

    REQUIRE(upsampleFactor >= 1.f);

    auto coordBack = linearTransform(1.f / upsampleFactor, point(0.f));

    ////

    if (hint.target == ImgOutputConsole)
    {
        GPU_MATRIX_ALLOC(gpuMatrix, uint8_x4, outputSize);

        ////

        require((visualizeScalarMatrix<Type, uint8_x4>(img, coordBack, channel, valueTransform, upsampleType, borderMode, gpuMatrix, stdPass)));

        ////

        require(baseConsole.addImageBgr(gpuMatrix, ImgOutputHint(hint).setDesc(paramMsg(STR("%0 [%1, %2]"), hint.desc, fltf(minVal, 3), fltf(maxVal, 3))), stdPass));

    }
    else if (hint.target == ImgOutputOverlay)
    {

        ScalarVisualizationProvider<Type> outputProvider(ScalarVisualizationParams<Type>{img, channel, coordBack, valueTransform, upsampleType, borderMode, hint.overlayCentering}, kit);

        require(baseConsole.overlaySetImageBgr(outputSize, outputProvider, paramMsg(STR("%0 [%1, %2] ~%3 bits"), hint.desc, 
            fltf(minVal, 3), fltf(maxVal, 3), fltf(-nativeLog2(maxVal - minVal), 1)), stdPass));

    }
    else
    {

        REQUIRE(false);

    }

    //----------------------------------------------------------------
    //
    // Text value
    //
    //----------------------------------------------------------------

    auto printTextValue = [&] () -> stdbool
    {
        Point<Space> userIdx{};
        Type userValue{};
        require(getElementAtUserPoint(img, coordBack, userIdx, userValue, stdPass));

        if (getTextEnabled())
            require(printMsgL(kit, STR("Value[%0] = %1"), userIdx, fltg(convertFloat32(userValue), 5)));

        returnTrue;
    };

    if (hint.target == ImgOutputOverlay && kit.userPoint.valid)
        errorBlock(printTextValue());

    //// 

    returnTrue;
}

//================================================================
//
// instantiations
//
//================================================================

#define TMP_MACRO(Type, _) \
    \
    template \
    stdbool GpuImageConsoleThunk::addMatrixExImpl<Type> \
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
    );

IMAGE_CONSOLE_FOREACH_SCALAR_TYPE(TMP_MACRO, _)
IMAGE_CONSOLE_FOREACH_VECTOR_TYPE(TMP_MACRO, _)

#undef TMP_MACRO

//================================================================
//
// VectorVisualizationParams
//
//================================================================

template <typename Vector>
struct VectorVisualizationParams
{
    GpuMatrix<const Vector> image;
    float32 valueFactor;
    float32 textFactor;
    float32 arrowFactor;
    Point<float32> coordBackMul;
    InterpType upsampleType;
    BorderMode borderMode;
    bool overlayCentering;
    bool grayMode;
    bool textOutputEnabled;
};

//================================================================
//
// VectorVisualizationProvider
//
//================================================================

template <typename Vector>
class VectorVisualizationProvider : public GpuImageProviderBgr32, private VectorVisualizationParams<Vector>
{

    using Base = VectorVisualizationParams<Vector>;

    // Thanks GCC 5.4
    using Base::image;
    using Base::valueFactor;
    using Base::textFactor;
    using Base::arrowFactor;
    using Base::coordBackMul;
    using Base::upsampleType;
    using Base::borderMode;
    using Base::overlayCentering;
    using Base::grayMode;
    using Base::textOutputEnabled;

public:

    stdbool saveImage(const GpuMatrix<uint8_x4>& dest, stdNullPars) const;

    using Kit = GpuImageConsoleThunk::Kit;

    VectorVisualizationProvider(const VectorVisualizationParams<Vector>& params, const Kit& kit)
        : Base(params), kit(kit) {}

private:

    Kit kit;

};

//================================================================
//
// VectorVisualizationProvider::saveImage
//
//================================================================

template <typename Vector>
stdbool VectorVisualizationProvider<Vector>::saveImage(const GpuMatrix<uint8_x4>& dest, stdNullPars) const
{
    Point<float32> coordBackAdd = point(0.f);

    if (overlayCentering)
    {
        Point<float32> srcCenter = 0.5f * convertFloat32(image.size());
        Point<float32> dstCenter = 0.5f * convertFloat32(dest.size());
        coordBackAdd = srcCenter - coordBackMul * dstCenter;

        if (allv(coordBackMul == point(1.f)))
            coordBackAdd = convertFloat32(convertNearest<Space>(coordBackAdd));
    }

    ////

    require(visualizeVectorImage(image, dest, linearTransform(coordBackMul, coordBackAdd), valueFactor, 
        upsampleType, borderMode, grayMode, stdPass));

    ////

    auto drawArrow = [&] () -> stdbool
    {
        Vector userValue{};
        Point<Space> userIndex{};
        require(getElementAtUserPoint(image, linearTransform(coordBackMul, coordBackAdd), userIndex, userValue, stdPass));

        Point<float32> vectorValue = point(convertFloat32(userValue.x), convertFloat32(userValue.y));
        require(def(vectorValue));

        if (textOutputEnabled)
            printMsgL(kit, STR("Value[%0] = %1"), userIndex, fltfs(textFactor * vectorValue, 2));

        ////

        Point<float32> dstPos = convertFloat32(kit.userPoint.position) + 0.5f; // to space format
        require(imposeVectorArrow(dest, dstPos, arrowFactor * vectorValue, stdPass));
        returnTrue;
    };

    if (kit.userPoint.valid)
        errorBlock(drawArrow());

    ////

    returnTrue;
}

//================================================================
//
// GpuImageConsoleThunk::addVectorImageGeneric
//
//================================================================

template <typename Vector>
stdbool GpuImageConsoleThunk::addVectorImageGeneric
(
    const GpuMatrix<const Vector>& image,
    float32 maxVector,
    const Point<float32>& upsampleFactor,
    InterpType upsampleType,
    const Point<Space>& upsampleSize,
    BorderMode borderMode,
    const ImgOutputHint& hint,
    stdNullPars
)
{
    REQUIRE(upsampleFactor >= 1.f);
    Point<float32> coordBackMul = 1.f / upsampleFactor;

    REQUIRE(def(maxVector));
    float32 valueFactor = nativeRecipZero(maxVector);

    ////

    REQUIRE(upsampleSize >= 0);

    Point<Space> outputSize = upsampleSize;

    if (allv(upsampleSize == point(0)))
        outputSize = convertUp<Space>(convertFloat32(image.size()) * upsampleFactor);

    //----------------------------------------------------------------
    //
    // Report max vector
    //
    //----------------------------------------------------------------

    if (getTextEnabled())
        printMsgL(kit, STR("Max Vector = %0 (octave %1)"), fltf(absv(maxVector), 3), fltfs(logf(absv(maxVector)) / logf(2.f), 1));

    //----------------------------------------------------------------
    //
    // X/Y display
    //
    //----------------------------------------------------------------

    if (vectorDisplayMode == VectorDisplayX || vectorDisplayMode == VectorDisplayY)
    {
        require(addMatrixChan(image, vectorDisplayMode == VectorDisplayY, -maxVector, +maxVector, upsampleFactor, upsampleType, upsampleSize, borderMode, hint, stdPass));
        returnTrue;
    }

    //
    //
    //

    if (hint.target == ImgOutputOverlay)
    {
        VectorVisualizationProvider<Vector> sourceBgr(VectorVisualizationParams<Vector>{image, valueFactor, hint.textFactor, hint.arrowFactor, coordBackMul, upsampleType, borderMode, 
            hint.overlayCentering, vectorDisplayMode == VectorDisplayMagnitude, getTextEnabled()}, kit);

        require(baseConsole.overlaySetImageBgr(outputSize, sourceBgr, hint.desc, stdPass));
    }

    //
    //
    //

    if (hint.target == ImgOutputConsole)
    {
        GPU_MATRIX_ALLOC(result, uint8_x4, outputSize);
    
        require(visualizeVectorImage(image, result, linearTransform(coordBackMul, point(0.f)), valueFactor, upsampleType, borderMode, 
            vectorDisplayMode == VectorDisplayMagnitude, stdPass));

        ////

        require(baseConsole.addImageBgr(result, hint, stdPass));

    }

    ////

    returnTrue;
}

//----------------------------------------------------------------

INSTANTIATE_FUNC(GpuImageConsoleThunk::addVectorImageGeneric<float16_x2>)
INSTANTIATE_FUNC(GpuImageConsoleThunk::addVectorImageGeneric<float16_x4>)

INSTANTIATE_FUNC(GpuImageConsoleThunk::addVectorImageGeneric<float32_x2>)
INSTANTIATE_FUNC(GpuImageConsoleThunk::addVectorImageGeneric<float32_x4>)

//================================================================
//
// Yuv420ConvertProvider
//
//================================================================

template <typename Pixel>
class Yuv420ConvertProvider : public GpuImageProviderBgr32
{

public:

    stdbool saveImage(const GpuMatrix<uint8_x4>& dest, stdNullPars) const
    {
        REQUIRE(equalSize(image.luma, dest));
        require(convertYuv420ToBgr<Pixel>(image.luma, image.chroma, nullptr, nullptr, point(0), make_uint8_x4(0, 0, 0, 0), dest, stdPass)); // 0.22 ms
        returnTrue;
    }

    Yuv420ConvertProvider(const GpuPackedYuv<const Pixel>& image, const GpuProcessKit& kit)
        : image(image), kit(kit) {}

private:

    GpuPackedYuv<const Pixel> image;
    GpuProcessKit kit;

};

//================================================================
//
// GpuImageConsoleThunk::addYuvImage420Func
//
//================================================================

template <typename Type>
stdbool GpuImageConsoleThunk::addYuvImage420Func
(
    const GpuPackedYuv<const Type>& image,
    const ImgOutputHint& hint,
    stdNullPars
)
{
    //
    //
    //

    if (hint.target == ImgOutputOverlay)
    {
        require(baseConsole.overlaySetImageBgr(image.luma.size(), Yuv420ConvertProvider<Type>(image, kit), hint, stdPass));
    }

    //
    //
    //

    if (hint.target == ImgOutputConsole)
    {
        GPU_MATRIX_ALLOC(tmp, uint8_x4, image.luma.size());
        require(convertYuv420ToBgr<Type>(image.luma, image.chroma, nullptr, nullptr, point(0), make_uint8_x4(0, 0, 0, 0), tmp, stdPass));
    
        require(baseConsole.addImageBgr(tmp, hint, stdPass));
    }

    ////

    returnTrue;
}

//----------------------------------------------------------------

#define TMP_MACRO(Type, _) \
    INSTANTIATE_FUNC_EX(GpuImageConsoleThunk::addYuvImage420Func<Type>, Type)

IMAGE_CONSOLE_FOREACH_YUV420_TYPE(TMP_MACRO, _)

#undef TMP_MACRO

//================================================================
//
// UnpackedColorConvertProvider
//
//================================================================

template <typename Type>
class UnpackedColorConvertProvider : public GpuImageProviderBgr32, private ScalarVisualizationParams<Type>
{

    using Base = ScalarVisualizationParams<Type>;

    // Thanks GCC 5.4
    using Base::img;
    using Base::channel;
    using Base::coordBack;
    using Base::valueTransform;
    using Base::upsampleType;
    using Base::borderMode;
    using Base::overlayCentering;

public:

    UnpackedColorConvertProvider(const ScalarVisualizationParams<Type>& params, ColorMode colorMode, const GpuProcessKit& kit)
        : Base(params), colorMode(colorMode), kit(kit) {}

    stdbool saveImage(const GpuMatrix<uint8_x4>& dest, stdNullPars) const;

private:

    ColorMode colorMode;
    GpuProcessKit kit;

};

#endif

//================================================================
//
// convertPackedYuvToRgb
//
//================================================================

GPUTOOL_2D_BEG
(                   
    convertPackedYuvToRgb,
    PREP_EMPTY,
    ((const float16_x4, src))
    ((uint8_x4, dst)),
    PREP_EMPTY
)
#if DEVCODE
{
    float32_x4 srcValue = loadNorm(src);
    float32_x4 dstValue = convertYPbPrToBgr(srcValue.x, srcValue.y, srcValue.z);
    storeNorm(dst, dstValue);
}
#endif
GPUTOOL_2D_END

//================================================================
//
// UnpackedColorConvertProvider::saveImage
//
//================================================================

#if HOSTCODE

template <typename Type>
stdbool UnpackedColorConvertProvider<Type>::saveImage(const GpuMatrix<uint8_x4>& dest, stdNullPars) const
{
    auto newCoordBack = coordBack;

    ////

    if (overlayCentering)
    {
        Point<float32> srcCenter = 0.5f * convertFloat32(img.size());
        Point<float32> dstCenter = 0.5f * convertFloat32(dest.size());

        newCoordBack.C0 = srcCenter - coordBack.C1 * dstCenter;

        if (allv(coordBack.C1 == point(1.f)))
            newCoordBack.C0 = convertFloat32(convertNearest<Space>(newCoordBack.C0));
    }

    ////

    if (colorMode == ColorRgb)
    {
        require((upconvertValueMatrix<Type, uint8_x4>(img, newCoordBack, valueTransform, upsampleType, borderMode, dest, stdPass)));
    }
    else if (colorMode == ColorYuv)
    {
        GPU_MATRIX_ALLOC(tmp, float16_x4, dest.size());
        require((upconvertValueMatrix<Type, float16_x4>(img, newCoordBack, valueTransform, upsampleType, borderMode, tmp, stdPass)));
        require(convertPackedYuvToRgb(tmp, dest, stdPass));
    }
    else
    {
        REQUIRE(false);
    }

    ////

    returnTrue;
}

//================================================================
//
// GpuImageConsoleThunk::addColorImageFunc
//
//================================================================

template <typename Type>
stdbool GpuImageConsoleThunk::addColorImageFunc
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
)
{
    REQUIRE(upsampleSize >= 0);

    Point<Space> outputSize = upsampleSize;

    if (allv(outputSize == point(0)))
        outputSize = convertUp<Space>(convertFloat32(img.size()) * upsampleFactor);


    //
    //
    //

    LinearTransform<float32> valueTransform = ltByTwoPoints(minVal, 0.f, maxVal, 1.f);
    if (minVal == maxVal) valueTransform = ltOutputZero<float32>();

    //
    //
    //

    REQUIRE(upsampleFactor >= 1.f);
    auto coordBack = linearTransform(1.f / upsampleFactor, point(0.f));

    ////

    if (hint.target == ImgOutputOverlay)
    {
        UnpackedColorConvertProvider<Type> outputProvider
        (
            ScalarVisualizationParams<Type>{img, 0, coordBack, valueTransform, upsampleType, borderMode, hint.overlayCentering}, 
            colorMode, kit
        );

        require(baseConsole.overlaySetImageBgr(outputSize, outputProvider, paramMsg(STR("%0 [%1, %2] ~%3 bits"), hint.desc, 
            fltf(minVal, 3), fltf(maxVal, 3), fltf(-nativeLog2(maxVal - minVal), 1)), stdPass));
    }

    //----------------------------------------------------------------
    //
    // Text value
    //
    //----------------------------------------------------------------

    auto printTextValue = [&] () -> stdbool
    {
        Point<Space> userIdx{};
        Type userValue{};
        require(getElementAtUserPoint(img, coordBack, userIdx, userValue, stdPass));

        if (getTextEnabled())
        {
            using Base = VECTOR_BASE(Type);
            int hexDigits = divUp<int>(sizeof(Base) * CHAR_BIT, 4);
            require(printMsgL(kit, STR("Value[%0] = %1 (hex %2)"), userIdx, fltg(convertFloat32(userValue), 5), hex(userValue, hexDigits)));
        }

        returnTrue;
    };

    if (hint.target == ImgOutputOverlay && kit.userPoint.valid)
        errorBlock(printTextValue());

    ////

    returnTrue;
}

//----------------------------------------------------------------

#define TMP_MACRO(Type, _) \
    INSTANTIATE_FUNC_EX(GpuImageConsoleThunk::addColorImageFunc<Type>, Type)

IMAGE_CONSOLE_FOREACH_X4_TYPE(TMP_MACRO, _)

#undef TMP_MACRO

//----------------------------------------------------------------

#endif

}
