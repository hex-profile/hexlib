#if HOSTCODE
#include "gaussPresentationTest.h"
#endif

#include "computeVectorVisualization.h"
#include "gpuDevice/loadstore/storeNorm.h"
#include "gpuSupport/gpuTexTools.h"
#include "gpuSupport/gpuTool.h"
#include "imageRead/positionTools.h"
#include "mathFuncs/bsplineShapes.h"
#include "mathFuncs/gaussApprox.h"
#include "mathFuncs/rotationMath.h"
#include "vectorTypes/vectorOperations.h"

#if HOSTCODE
#include "cfgTools/multiSwitch.h"
#include "storage/classThunks.h"
#include "dataAlloc/gpuMatrixMemory.h"
#include "cfgTools/rangeValueControl.h"
#include "imageConsole/gpuImageConsole.h"
#endif

namespace gaussPresentationTest {

//================================================================
//
// filterShape
//
//================================================================

sysinline float32 filterTaps(bool alternative)
{
    return
        alternative ?
        3.f :
        3.f;
}

//----------------------------------------------------------------

sysinline float32 filterShape(const Point<float32>& dist, float32 gaussSigma, bool alternative)
{
    using Bspline = Bspline<3>;

    float32 box = (absv(dist.X) <= 0.5f && absv(dist.Y) <= 0.5f) ? 1.f : 0.f;
    float32 tent = (1 - saturatev(absv(dist.X))) * (1 - saturatev(absv(dist.Y)));
    float32 gauss = gaussExpoApprox<4>(vectorLengthSq(dist) / square(gaussSigma));
    float32 bspline = Bspline::func(dist.X) * Bspline::func(dist.Y);

    float32 result = alternative ? bspline : gauss;

    float32 filterRadius = 0.5f * filterTaps(alternative);

    if_not (maxv(absv(dist.X), absv(dist.Y)) <= filterRadius)
        result = 0;

    return result;
}

//================================================================
//
// fillEdgeImage
//
//================================================================

GPUTOOL_2D_BEG
(
    fillEdgeImage,
    PREP_EMPTY,
    ((float32, dst)),
    ((Point<float32>, rotation))
)
#if DEVCODE
{
    Point<float32> imageSize = convertFloat32(vGlobSize);
    Point<float32> center = 0.5f * imageSize;
    Point<float32> ofs = point(Xs, Ys) - center;

    Point<float32> rotatedOfs = complexMul(ofs, complexConjugate(rotation));

    storeNorm(dst, rotatedOfs.X < 0 ? 1.f : 0.f);
}
#endif
GPUTOOL_2D_END

//================================================================
//
// downsampleByFilter
//
//================================================================

GPUTOOL_2D_BEG
(
    downsampleByFilter,
    ((const float32, src, INTERP_NEAREST, BORDER_MIRROR)),
    ((float32, dst)),
    ((float32, downsampleFactor))
    ((float32, filterScale))
    ((float32, gaussSigma))
    ((bool, alternative))
)
#if DEVCODE
{
    Point<float32> dstPos = point(Xs, Ys);
    Point<float32> srcCenterPos = dstPos * downsampleFactor;

    ////

    Space taps = convertUp<Space>(filterTaps(alternative) * filterScale * downsampleFactor - 1e-6f);
    float32 divFilterScale = fastRecip(downsampleFactor * filterScale);

    Point<float32> filterOrg = computeFilterStartPos(srcCenterPos, taps);

    ////

    float32 sumSpatialWeight = 0;
    float32 sumSpatialWeightPresence = 0;

    for_count (iY, taps)
    {
        for_count (iX, taps)
        {
            Point<float32> srcReadPos = point(filterOrg.X + iX, filterOrg.Y + iY);
            auto value = tex2D(srcSampler, srcReadPos * srcTexstep);

            Point<float32> dist = srcReadPos - srcCenterPos;
            float32 spatialWeight = filterShape(dist * divFilterScale, gaussSigma, alternative);

            sumSpatialWeight += spatialWeight;
            sumSpatialWeightPresence += spatialWeight * value;
        }
    }

    ////

    float32 divSumWeight = 1.f / sumSpatialWeight;

    storeNorm(dst, divSumWeight * sumSpatialWeightPresence);
}
#endif
GPUTOOL_2D_END

//================================================================
//
// upsampleByFilter
//
//================================================================

GPUTOOL_2D_BEG
(
    upsampleByFilter,
    ((const float32, src, INTERP_NEAREST, BORDER_ZERO)),
    ((float32, dst)),
    ((float32, upsampleFactor))
    ((Point<float32>, srcSize))
    ((Point<float32>, dstSize))
    ((float32, filterScale))
    ((float32, gaussSigma))
    ((bool, alternative))
)
#if DEVCODE
{

    ////

    Point<float32> dstPos = point(Xs, Ys) - 0.5f * dstSize;
    Point<float32> srcCenterPos = dstPos / upsampleFactor + 0.5f * srcSize;

    ////

    Space taps = convertUp<Space>(filterTaps(alternative) * filterScale - 1e-6f);
    float32 divFilterScale = fastRecip(filterScale);

    Point<float32> filterOrg = computeFilterStartPos(srcCenterPos, taps);

    ////

    float32 sumWeightValue = 0;
    float32 sumWeight = 0;


    for_count (iY, taps)
    {
        for_count (iX, taps)
        {
            Point<float32> srcReadPos = point(filterOrg.X + iX, filterOrg.Y + iY);
            auto value = tex2D(srcSampler, srcReadPos * srcTexstep);

            Point<float32> dist = srcReadPos - srcCenterPos;
            float32 weight = filterShape(dist * divFilterScale, gaussSigma, alternative);

            sumWeight += weight;
            sumWeightValue += weight * value;
        }
    }

    ////

    float32 divSumWeight = 1.f / sumWeight;
    float32 result = divSumWeight * sumWeightValue;

    storeNorm(dst, result);
}
#endif
GPUTOOL_2D_END

//================================================================
//
// GaussPresentationTestImpl
//
//================================================================

#if HOSTCODE

class GaussPresentationTestImpl
{

public:

    GaussPresentationTestImpl();
    void serialize(const ModuleSerializeKit& kit);
    bool active() const {return displaySwitch != DisplayNothing;}
    stdbool realloc(stdPars(GpuModuleReallocKit)) {returnTrue;}
    bool reallocValid() const {return true;}
    stdbool process(const ProcessParams& o, stdPars(GpuModuleProcessKit));

private:

    enum DisplayType {DisplayNothing, DisplayHiresInput, DisplayHiresOutput, DisplayCount};
    ExclusiveMultiSwitch<DisplayType, DisplayCount, 0x0D09EB4D> displaySwitch;

    RangeValueControl<float32> rotationAngle;
    NumericVarStaticEx<float32, int, 1, 64, 8> downsampleFactor;
    NumericVarStaticEx<float32, int, 1, 64, 8> upsampleFactor;

    NumericVarStaticEx<float32, int, 0, 64, 1> downsampleScale;
    NumericVarStaticEx<float32, int, 0, 64, 1> upsampleScale;

    NumericVarStaticEx<float32, int, 0, 64, 0> gaussSigma;
};

//================================================================
//
// GaussPresentationTestImpl::GaussPresentationTestImpl
//
//================================================================

GaussPresentationTestImpl::GaussPresentationTestImpl()
    :
    rotationAngle(0, 1, 0, 1.f/512, RangeValueCircular)
{
    gaussSigma = 0.6f;
}

//================================================================
//
// GaussPresentationTestImpl::serialize
//
//================================================================

void GaussPresentationTestImpl::serialize(const ModuleSerializeKit& kit)
{
    displaySwitch.serialize
    (
        kit, STR("Display"),
        {STR("<Nothing>"), STR("")},
        {STR("Hires Input"), STR("")},
        {STR("Hires Output"), STR("")}
    );

    rotationAngle.serialize(kit, STR("Rotation Angle"), STR("7"), STR("8"));
    downsampleFactor.serialize(kit, STR("Downsample Factor"));
    upsampleFactor.serialize(kit, STR("Upsample Factor"));
    downsampleScale.serialize(kit, STR("Downsample Scale"));
    upsampleScale.serialize(kit, STR("Upsample Scale"));
    gaussSigma.serialize(kit, STR("Gauss Sigma"));
}

//================================================================
//
// GaussPresentationTestImpl::process
//
//================================================================

stdbool GaussPresentationTestImpl::process(const ProcessParams& o, stdPars(GpuModuleProcessKit))
{
    DisplayType displayType = kit.verbosity >= Verbosity::On ? displaySwitch : DisplayNothing;

    if (displayType == DisplayNothing)
        returnTrue;

    ////

    Point<Space> hiSize = kit.display.screenSize;

    //----------------------------------------------------------------
    //
    // Hires
    //
    //----------------------------------------------------------------

    GPU_MATRIX_ALLOC(hiresData, float32, hiSize);
    require(fillEdgeImage(hiresData, circleCCW(rotationAngle()), stdPass));

    ////

    if (displayType == DisplayHiresInput)
    {
        require
        (
            kit.gpuImageConsole.addMatrixEx
            (
                hiresData,
                0, kit.display.factor, point(1.f),
                INTERP_NEAREST, kit.display.screenSize, BORDER_CLAMP,
                STR("Hires Input"),
                stdPass
            )
        );
    }

    //----------------------------------------------------------------
    //
    // Downsample
    //
    //----------------------------------------------------------------

    Point<Space> loSize = convertDown<Space>(convertFloat32(hiSize) / downsampleFactor());

    GPU_MATRIX_ALLOC(loresData, float32, loSize);

    require(downsampleByFilter(hiresData, loresData, downsampleFactor, downsampleScale, gaussSigma, kit.alternative, stdPass));

    //----------------------------------------------------------------
    //
    // Upsample
    //
    //----------------------------------------------------------------

    GPU_MATRIX_ALLOC(upsampledData, float32, hiSize);

    ///

    require
    (
        upsampleByFilter
        (
            loresData,
            upsampledData,
            upsampleFactor,
            convertFloat32(hiSize) / downsampleFactor(),
            convertFloat32(hiSize),
            upsampleScale,
            gaussSigma,
            kit.alternative,
            stdPass
        )
    );

    if (displayType == DisplayHiresOutput)
    {
        require
        (
            kit.gpuImageConsole.addMatrixEx
            (
                upsampledData,
                0, kit.display.factor, point(1.f),
                INTERP_NEAREST, kit.display.screenSize, BORDER_CLAMP,
                STR("Hires Output"),
                stdPass
            )
        );
    }

    ////

    returnTrue;
}

//================================================================
//
// Thunks
//
//================================================================

CLASSTHUNK_CONSTRUCT_DESTRUCT(GaussPresentationTest)
CLASSTHUNK_VOID1(GaussPresentationTest, serialize, const ModuleSerializeKit&)
CLASSTHUNK_BOOL_CONST0(GaussPresentationTest, active)
CLASSTHUNK_BOOL_STD0(GaussPresentationTest, realloc, GpuModuleReallocKit)
CLASSTHUNK_BOOL_CONST0(GaussPresentationTest, reallocValid)
CLASSTHUNK_BOOL_STD1(GaussPresentationTest, process, const ProcessParams&, GpuModuleProcessKit)

#endif

//----------------------------------------------------------------

}
