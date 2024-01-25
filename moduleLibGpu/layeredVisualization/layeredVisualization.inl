#if HOSTCODE
#include "layeredVisualization.h"
#endif

#include "computeVectorVisualization.h"
#include "gpuDevice/loadstore/loadNorm.h"
#include "gpuDevice/loadstore/storeNorm.h"
#include "gpuSupport/gpuTexTools.h"
#include "gpuSupport/gpuTool.h"
#include "mathFuncs/bsplineShapes.h"
#include "mathFuncs/gaussApprox.h"
#include "mathFuncs/rotationMath.h"
#include "numbers/mathIntrinsics.h"
#include "vectorTypes/vectorOperations.h"
#include "vectorTypes/vectorType.h"
#include "prepTools/prepIterate.h"
#include "imageRead/positionTools.h"

#if HOSTCODE
#include "prepTools/prepEnum.h"
#include "userOutput/paramMsg.h"
#include "dataAlloc/gpuMatrixMemory.h"
#include "gpuMatrixSet/gpuMatrixSet.h"
#include "imageConsole/gpuImageConsole.h"
#include "diagTools/readGpuElement.h"
#include "userOutput/printMsgEx.h"
#include "dataAlloc/gpuLayeredMatrixMemory.h"
#include "kits/userPoint.h"
#endif

namespace layeredVisualization {

//================================================================
//
// Arguments
//
//================================================================

#if !(defined(VISUALIZATION_MAX_LAYERS) && defined(VECTOR_TYPE) && defined(PRESENCE_TYPE))
    #error Arguments required
#endif

//----------------------------------------------------------------

#define VectorType VECTOR_TYPE
#define PresenceType PRESENCE_TYPE

//================================================================
//
// PRESENCE_EPSILON
//
//================================================================

#define PRESENCE_EPSILON 1e-6f

//================================================================
//
// layeredVisualization
//
//================================================================

#define PREP_ITER_DIMS 1
#define PREP_ITER_ARGS_0 (1, VISUALIZATION_MAX_LAYERS, "layeredVisualization/layeredVisualizationKernels.inl")

# include PREP_ITERATE_ND

#undef PREP_ITER_DIMS
#undef PREP_ITER_ARGS_0

//----------------------------------------------------------------

#if HOSTCODE

//================================================================
//
// displayVectorSet
//
//================================================================

stdbool displayVectorSet
(
    const GpuLayeredMatrix<const VectorType>& vectors,
    const GpuLayeredMatrix<const PresenceType>& presences,
    bool independentPresenceMode,
    const Point<float32>& pos,
    float32 vectorFactor,
    Space matrixSize,
    float32 thickness,
    const ImgOutputHint& hint,
    stdPars(GpuModuleProcessKit)
)
{
    REQUIRE(thickness > 0);
    GPU_MATRIX_ALLOC(tmp, uint8_x4, point(matrixSize));

    ////

    switch (vectors.layers())
    {
        #define TMP_MACRO(layersIdx, _) \
            TMP_MACRO_EX(PREP_INC(layersIdx))

        #define TMP_MACRO_EX(n) \
            case n: \
            require(PREP_PASTE_UNDER4(renderVectorSet, n, VectorType, PresenceType)(GPU_LAYERED_MATRIX_PASS(n, vectors), GPU_LAYERED_MATRIX_PASS(n, presences), tmp, independentPresenceMode, pos, vectorFactor, thickness, stdPass)); \
            break;

        PREP_FOR1(VISUALIZATION_MAX_LAYERS, TMP_MACRO, _)

        #undef TMP_MACRO
        #undef TMP_MACRO_EX

        default:
            REQUIRE(false);
    }

    ////

    require(kit.gpuImageConsole.addImageBgr(tmp, ImgOutputHint(hint).setTargetConsole().setDesc(paramMsg(STR("%0 Set"), hint.desc)).setID(hint.id ^ 0xCD4B3287), stdPass));

    returnTrue;
}

//================================================================
//
// convertIndependentPresenceToAdditive
//
//================================================================

stdbool convertIndependentPresenceToAdditive
(
    const GpuLayeredMatrix<const VectorType>& srcVector,
    const GpuLayeredMatrix<const PresenceType>& srcPresence,
    const GpuLayeredMatrix<PresenceType>& dstPresence,
    float32 vectorProximity,
    stdPars(GpuModuleProcessKit)
)
{
    REQUIRE(equalLayers(srcVector, srcPresence, dstPresence));
    Space layers = srcVector.layers();

    ////

    #define TMP_MACRO(n) \
        \
        require \
        ( \
            PREP_PASTE3(convertIndependentPresenceToAdditive, n, PresenceType) \
            ( \
                GPU_LAYERED_MATRIX_PASS(n, srcVector), \
                GPU_LAYERED_MATRIX_PASS(n, srcPresence), \
                GPU_LAYERED_MATRIX_PASS(n, dstPresence), \
                square(fastRecipZero(vectorProximity)), \
                stdPass \
            ) \
        ); \

    #define TMP_MACRO2(n, _) \
        if (layers == n) {TMP_MACRO(n)} else

    PREP_FOR1_FROM1_TO_COUNT(VISUALIZATION_MAX_LAYERS, TMP_MACRO2, _)
    REQUIRE(false);

    #undef TMP_MACRO
    #undef TMP_MACRO2

    returnTrue;
}

//================================================================
//
// LayeredVectorProvider
//
//================================================================

template <typename V, typename P>
class LayeredVectorProvider;

//----------------------------------------------------------------

template <>
class LayeredVectorProvider<VectorType, PresenceType> : public GpuImageProviderBgr32
{

public:

    stdbool saveImage(const GpuMatrixAP<uint8_x4>& dest, stdParsNull) const;

public:

    inline LayeredVectorProvider
    (
        const GpuLayeredMatrix<const VectorType>& vectorValue,
        const GpuLayeredMatrix<const PresenceType>& vectorPresence,
        float32 maxVector,
        const Point<float32>& upsampleFactor,
        bool upsampleInterpolation,
        const GpuModuleProcessKit& kit
    )
        :
        vectorValue(vectorValue),
        vectorPresence(vectorPresence),
        maxVector(maxVector),
        upsampleFactor(upsampleFactor),
        upsampleInterpolation(upsampleInterpolation),
        kit(kit)
    {
    }

private:

    const GpuLayeredMatrix<const VectorType>& vectorValue;
    const GpuLayeredMatrix<const PresenceType>& vectorPresence;
    const float32 maxVector;
    const Point<float32> upsampleFactor;
    const bool upsampleInterpolation;
    const GpuModuleProcessKit kit;

};

//================================================================
//
// LayeredVectorProvider::saveImage
//
//================================================================

stdbool LayeredVectorProvider<VectorType, PresenceType>::saveImage(const GpuMatrixAP<uint8_x4>& dest, stdParsNull) const
{
    REQUIRE(upsampleFactor >= 1);
    Point<float32> divUpsampleFactor = 1.f / upsampleFactor;

    REQUIRE(maxVector != 0);
    float32 divMaxVector = 1.f / maxVector;

    ////

    REQUIRE(vectorValue.layers() == vectorPresence.layers());

    switch (vectorValue.layers())
    {
        #define TMP_MACRO_EX(n) \
            case n: \
            require(PREP_PASTE_UNDER4(upsampleVectorVisualizationFunc, n, VectorType, PresenceType) \
                (GPU_LAYERED_MATRIX_PASS(n, vectorValue), GPU_LAYERED_MATRIX_PASS(n, vectorPresence), dest, divUpsampleFactor, divMaxVector, upsampleInterpolation, stdPass)); \
            break;

        #define TMP_MACRO(idx, _) \
            TMP_MACRO_EX(PREP_INC(idx))

        PREP_FOR1(VISUALIZATION_MAX_LAYERS, TMP_MACRO, _)

        #undef TMP_MACRO
        #undef TMP_MACRO_EX

        default:
            REQUIRE(false);
    }

    ////

    returnTrue;
}

//================================================================
//
// visualizeLayeredVector
//
//================================================================

template <>
stdbool visualizeLayeredVector
(
    const GpuLayeredMatrix<const VectorType>& vectorValue,
    const GpuLayeredMatrix<const PresenceType>& vectorPresence,
    bool independentPresenceMode,
    float32 maxVector,
    const Point<float32>& upsampleFactor,
    const Point<Space>& upsampleSize,
    bool upsampleInterpolation,
    const ImgOutputHint& hint,
    stdPars(GpuModuleProcessKit)
)
{
    REQUIRE(equalLayers(vectorValue, vectorPresence));
    Space layers = vectorValue.layers();
    REQUIRE(layers >= 1);

    REQUIRE(equalSize(vectorValue, vectorPresence));
    Point<Space> size = vectorValue.size();

    Space displayedLayer = kit.display.circularIndex(layers);

    //----------------------------------------------------------------
    //
    // Image
    //
    //----------------------------------------------------------------

    float32 vectorProximity = 0.2f * maxVector;

    ////

    if_not (independentPresenceMode)
    {
        LayeredVectorProvider<VectorType, PresenceType> provider(vectorValue, vectorPresence, maxVector, upsampleFactor, upsampleInterpolation, kit);
        require(kit.gpuImageConsole.overlaySetImageBgr(upsampleSize, provider, hint, stdPass));
    }
    else
    {
        GPU_LAYERED_MATRIX_ALLOC(additivePresence, PresenceType, layers, size);
        require(convertIndependentPresenceToAdditive(vectorValue, vectorPresence, additivePresence, vectorProximity, stdPass));

        LayeredVectorProvider<VectorType, PresenceType> provider(vectorValue, additivePresence, maxVector, upsampleFactor, upsampleInterpolation, kit);
        require(kit.gpuImageConsole.overlaySetImageBgr(upsampleSize, provider, hint, stdPass));
    }

    //----------------------------------------------------------------
    //
    // Vector set
    //
    //----------------------------------------------------------------

    breakBlock_
    {
        breakRequire(kit.userPoint.valid);

        ////

        Point<float32> userPosScreen = kit.userPoint.floatPos;
        Point<float32> userPosData = userPosScreen * (1.f / upsampleFactor);

        require(displayVectorSet(vectorValue, makeConst(vectorPresence), independentPresenceMode, userPosData, 1 / maxVector, 192, 2.5f, hint, stdPass));

        ////

        Point<Space> srcInt = convertNearest<Space>(userPosData - 0.5f); // Back to grid and round

        Point<Space> imageSize = vectorValue.getLayer(0).size();
        breakRequire(allv(imageSize >= 1));
        srcInt = clampRange(srcInt, point(0), imageSize - 1);

        float32 totalPresence = 0;

        for_count (r, layers)
        {
            PresenceType presence = zeroOf<PresenceType>();

            require(readGpuElement<PresenceType>(vectorPresence.getLayer(r), srcInt, presence, stdPass));
            totalPresence += convertFloat32(presence);

            VectorType vector = convertNearest<VectorType>(0);
            require(readGpuElement<VectorType>(vectorValue.getLayer(r), srcInt, vector, stdPass));

            printMsgL(kit, STR("%0 of vector %1"), fltf(convertFloat32(presence), 2), fltf(hint.textFactor * convertNearest<Point<float32>>(vector), 1));
        }

        printMsgL(kit, STR("%0 total"), fltf(convertFloat32(totalPresence), 2));
    }

    ////

    returnTrue;
}

//================================================================
//
// visualizeLayeredVector
//
//================================================================

#ifdef PRESENCE_ONCE

template <>
stdbool visualizeLayeredVector
(
    const GpuLayeredMatrix<const VectorType>& vectorValue,
    float32 maxVector,
    const Point<float32>& upsampleFactor,
    const Point<Space>& upsampleSize,
    bool upsampleInterpolation,
    const ImgOutputHint& hint,
    stdPars(GpuModuleProcessKit)
)
{
    Space layers = vectorValue.layers();
    REQUIRE(layers >= 1);

    Point<Space> size = vectorValue.size();
    GPU_LAYERED_MATRIX_ALLOC(vectorPresence, float16, layers, size);

    for_count (r, layers)
        require(gpuMatrixSet(vectorPresence.getLayer(r), convertNearest<float16>(1.f / layers), stdPass));

    require((visualizeLayeredVector<VectorType, float16>)(vectorValue, vectorPresence, false, maxVector, upsampleFactor, upsampleSize, upsampleInterpolation, hint, stdPass));

    returnTrue;
}

#endif

//----------------------------------------------------------------

#endif

}
