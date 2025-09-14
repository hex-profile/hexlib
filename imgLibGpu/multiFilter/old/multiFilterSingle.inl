#if HOSTCODE
#include "dataAlloc/gpuLayeredMatrixMemory.h"
#include "dataAlloc/gpuMatrixMemory.h"
#include "errorLog/errorLog.h"
#include "gpuAppliedApi/gpuAppliedApi.h"
#include "gpuMatrixSet/gpuMatrixSet.h"
#include "gpuProcessKit.h"
#include "numbers/divRound.h"
#include "prepTools/prepEnum.h"
#endif

#include "data/gpuMatrix.h"
#include "gpuDevice/gpuDevice.h"
#include "gpuDevice/loadstore/loadNorm.h"
#include "gpuDevice/loadstore/storeNorm.h"
#include "gpuSupport/gpuTemplateKernel.h"
#include "gpuSupport/gpuTexTools.h"
#include "gpuSupport/parallelLoop.h"
#include "imageRead/positionTools.h"
#include "kit/kit.h"
#include "mapDownsampleIndexToSource.h"
#include "prepTools/prepIncDec.h"
#include "vectorTypes/vectorOperations.h"

#ifdef FUNCSPACE
namespace FUNCSPACE {
#endif

#ifndef FUNCNAME
    #error
#endif

namespace PREP_PASTE(FUNCNAME, Space) {

//================================================================
//
// threadCountX
// threadCountY
//
//================================================================

static const Space horThreadCountX = 32;
static const Space horThreadCountY = 8;
sysinline Point<Space> horThreadCount() {return point(horThreadCountX, horThreadCountY);}

static const Space verThreadCountX = 32;
static const Space verThreadCountY = 8;
sysinline Point<Space> verThreadCount() {return point(verThreadCountX, verThreadCountY);}

//================================================================
//
// srcSampler
//
//================================================================

#if DEVCODE

devDefineSampler(PREP_PASTE(FUNCNAME, srcSampler_x1), DevSampler2D, DevSamplerFloat, 1)
devDefineSampler(PREP_PASTE(FUNCNAME, srcSampler_x2), DevSampler2D, DevSamplerFloat, 2)

////

#define TMP_MACRO(k, _) \
    devDefineSampler(PREP_PASTE(FUNCNAME, intermSampler##k##_x1), DevSampler2D, DevSamplerFloat, 1) \
    devDefineSampler(PREP_PASTE(FUNCNAME, intermSampler##k##_x2), DevSampler2D, DevSamplerFloat, 2)

PREP_FOR(FILTER_COUNT, TMP_MACRO, _)

#undef TMP_MACRO

#endif

//================================================================
//
// LINEAR_COMBINATION
//
//================================================================

#ifndef LINEAR_COMBINATION

    #define LINEAR_ONLY(s)
    #define LINEAR_ONLY_COMMA(s)

#else

    #define LINEAR_ONLY(s) s
    #define LINEAR_ONLY_COMMA(s) s,

#endif

//================================================================
//
// IntermParams
//
//================================================================

template <typename Dst>
struct IntermParams
{
    Point<float32> srcTexstep;
    Point<Space> dstSize;
    GpuMatrix<Dst> dst[FILTER_COUNT];
};

//================================================================
//
// FinalParams
//
//================================================================

template <typename Dst>
struct FinalParams
{
    Point<float32> srcTexstep;
    Point<Space> dstSize;

#ifndef LINEAR_COMBINATION

    GpuMatrix<Dst> dst[FILTER_COUNT];

#else

    float32 filterMixCoeffs[FILTER_COUNT];

    GpuMatrix<const Dst> dstMixImage;
    float32 dstMixCoeff;

    GpuMatrix<Dst> dst;

#endif

};

//================================================================
//
// Instances
//
//================================================================

#define RANK 1

#define HORIZONTAL 1
# include "multiFilterSingleKernel.inl"
#undef HORIZONTAL

#define HORIZONTAL 0
# include "multiFilterSingleKernel.inl"
#undef HORIZONTAL

#undef RANK

//----------------------------------------------------------------

#define RANK 2

#define HORIZONTAL 1
# include "multiFilterSingleKernel.inl"
#undef HORIZONTAL

#define HORIZONTAL 0
# include "multiFilterSingleKernel.inl"
#undef HORIZONTAL

#undef RANK

//================================================================
//
// Kernel instantiations (Interm)
//
//================================================================

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, IntermHor1), IntermParams, PREP_PASTE(FUNCNAME, IntermHorLink), PREP_PASTE(FUNCNAME, IntermHor8u), (uint8));
GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, IntermVer1), IntermParams, PREP_PASTE(FUNCNAME, IntermVerLink), PREP_PASTE(FUNCNAME, IntermVer8u), (uint8));

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, IntermHor1), IntermParams, PREP_PASTE(FUNCNAME, IntermHorLink), PREP_PASTE(FUNCNAME, IntermHor8s), (int8));
GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, IntermVer1), IntermParams, PREP_PASTE(FUNCNAME, IntermVerLink), PREP_PASTE(FUNCNAME, IntermVer8s), (int8));

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, IntermHor2), IntermParams, PREP_PASTE(FUNCNAME, IntermHorLink), PREP_PASTE(FUNCNAME, IntermHor8s_x2), (int8_x2));
GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, IntermVer2), IntermParams, PREP_PASTE(FUNCNAME, IntermVerLink), PREP_PASTE(FUNCNAME, IntermVer8s_x2), (int8_x2));

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, IntermHor1), IntermParams, PREP_PASTE(FUNCNAME, IntermHorLink), PREP_PASTE(FUNCNAME, IntermHor16f), (float16));
GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, IntermVer1), IntermParams, PREP_PASTE(FUNCNAME, IntermVerLink), PREP_PASTE(FUNCNAME, IntermVer16f), (float16));

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, IntermHor2), IntermParams, PREP_PASTE(FUNCNAME, IntermHorLink), PREP_PASTE(FUNCNAME, IntermHor16f_x2), (float16_x2));
GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, IntermVer2), IntermParams, PREP_PASTE(FUNCNAME, IntermVerLink), PREP_PASTE(FUNCNAME, IntermVer16f_x2), (float16_x2));

//================================================================
//
// Kernel instantiations (Final)
//
//================================================================

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, FinalHor1), FinalParams, PREP_PASTE(FUNCNAME, FinalHorLink), PREP_PASTE(FUNCNAME, FinalHor8u), (uint8));
GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, FinalVer1), FinalParams, PREP_PASTE(FUNCNAME, FinalVerLink), PREP_PASTE(FUNCNAME, FinalVer8u), (uint8));

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, FinalHor1), FinalParams, PREP_PASTE(FUNCNAME, FinalHorLink), PREP_PASTE(FUNCNAME, FinalHor8s), (int8));
GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, FinalVer1), FinalParams, PREP_PASTE(FUNCNAME, FinalVerLink), PREP_PASTE(FUNCNAME, FinalVer8s), (int8));

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, FinalHor2), FinalParams, PREP_PASTE(FUNCNAME, FinalHorLink), PREP_PASTE(FUNCNAME, FinalHor8s_x2), (int8_x2));
GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, FinalVer2), FinalParams, PREP_PASTE(FUNCNAME, FinalVerLink), PREP_PASTE(FUNCNAME, FinalVer8s_x2), (int8_x2));

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, FinalHor1), FinalParams, PREP_PASTE(FUNCNAME, FinalHorLink), PREP_PASTE(FUNCNAME, FinalHor16f), (float16));
GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, FinalVer1), FinalParams, PREP_PASTE(FUNCNAME, FinalVerLink), PREP_PASTE(FUNCNAME, FinalVer16f), (float16));

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, FinalHor2), FinalParams, PREP_PASTE(FUNCNAME, FinalHorLink), PREP_PASTE(FUNCNAME, FinalHor16f_x2), (float16_x2));
GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, FinalVer2), FinalParams, PREP_PASTE(FUNCNAME, FinalVerLink), PREP_PASTE(FUNCNAME, FinalVer16f_x2), (float16_x2));

//================================================================
//
// DIR
//
//================================================================

#ifndef HORIZONTAL_FIRST
    #error
#elif HORIZONTAL_FIRST
    #define DIR(h, v) h
#else
    #define DIR(h, v) v
#endif

//----------------------------------------------------------------

}

//================================================================
//
// Main func
//
//================================================================

#if HOSTCODE

template <typename Src, typename Dst>
void FUNCNAME
(
    const GpuMatrix<const Src>& src,

#ifndef LINEAR_COMBINATION
    PREP_ENUM_INDEXED(FILTER_COUNT, const GpuMatrix<Dst>& dst),
#else
    const GpuMatrix<const Dst>& dstMixImage,
    float32 dstMixCoeff,
    const GpuMatrix<Dst>& dst,
    const float32* filterMixCoeffs,
#endif

    BorderMode borderMode,
    stdPars(GpuProcessKit)
)
{
    using namespace PREP_PASTE(FUNCNAME, Space);

    ////

    Point<Space> srcSize = src.size();

#ifndef LINEAR_COMBINATION
    Point<Space> dstSize = dst0.size();
    REQUIRE(equalSize(PREP_ENUM_INDEXED(FILTER_COUNT, dst)));
#else
    Point<Space> dstSize = dst.size();
    REQUIRE(equalSize(dst, dstMixImage));
#endif

    Point<Space> intermSize = srcSize;
    intermSize.DIR(X, Y) = dstSize.DIR(X, Y);
    GPU_LAYERED_MATRIX_ALLOC(interm, Dst, FILTER_COUNT, intermSize);

    ////

    if_not (kit.dataProcessing)
        return;

    ////

    const int srcRank = VectorTypeRank<Src>::val;

    //----------------------------------------------------------------
    //
    // Interm
    //
    //----------------------------------------------------------------

    const GpuSamplerLink* srcSampler = (srcRank == 1) ? soft_cast<const GpuSamplerLink*>(&PREP_PASTE(FUNCNAME, srcSampler_x1)) : &PREP_PASTE(FUNCNAME, srcSampler_x2);
    kit.gpuSamplerSetting.setSamplerImage(*srcSampler, src, borderMode, LinearInterpolation{false}, ReadNormalizedFloat{true}, NormalizedCoords{true}, stdPass);

    ////

    Point<Space> intermLaunchSize = srcSize;
    intermLaunchSize.DIR(X, Y) = dstSize.DIR(X, Y);

    Point<Space> intermThreadCount = DIR(horThreadCount(), verThreadCount());

    ////

    IntermParams<Dst> intermParams;
    intermParams.srcTexstep = computeTexstep(srcSize);
    intermParams.dstSize = intermSize;

    for_count (k, FILTER_COUNT)
        intermParams.dst[k] = interm.getLayer(k);

    ////

    kit.gpuKernelCalling.callKernel
    (
        divUpNonneg(intermLaunchSize, intermThreadCount),
        intermThreadCount,
        areaOf(intermSize),
        PREP_PASTE3(FUNCNAME, Interm, DIR(HorLink, VerLink))<Dst>(),
        intermParams,
        kit.gpuCurrentStream,
        stdPass
    );

    //----------------------------------------------------------------
    //
    // Final
    //
    //----------------------------------------------------------------

    #define TMP_MACRO(k, _) \
        { \
            const GpuSamplerLink* sampler = 0; \
            if (srcRank == 1) sampler = &PREP_PASTE(FUNCNAME, intermSampler##k##_x1); \
            if (srcRank == 2) sampler = &PREP_PASTE(FUNCNAME, intermSampler##k##_x2); \
            REQUIRE(sampler != 0); \
            kit.gpuSamplerSetting.setSamplerImage(*sampler, makeConst(interm.getLayer(k)), borderMode, \
                LinearInterpolation{false}, ReadNormalizedFloat{true}, NormalizedCoords{true}, stdPass); \
        }

    PREP_FOR(FILTER_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO

    ////

    Point<Space> finalLaunchSize = dstSize;
    finalLaunchSize.DIR(Y, X) = dstSize.DIR(Y, X);

    Point<Space> finalThreadCount = DIR(verThreadCount(), horThreadCount());

    ////

    FinalParams<Dst> finalParams;
    finalParams.srcTexstep = computeTexstep(intermSize);
    finalParams.dstSize = dstSize;

#ifndef LINEAR_COMBINATION

    #define TMP_MACRO(k, _) \
        finalParams.dst[k] = dst##k;

    PREP_FOR(FILTER_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO

#else

    for_count (k, FILTER_COUNT)
        finalParams.filterMixCoeffs[k] = filterMixCoeffs[k];

    finalParams.dstMixImage = dstMixImage;
    finalParams.dstMixCoeff = dstMixCoeff;

    finalParams.dst = dst;

#endif

    ////

    kit.gpuKernelCalling.callKernel
    (
        divUpNonneg(finalLaunchSize, finalThreadCount),
        finalThreadCount,
        areaOf(dstSize),
        PREP_PASTE3(FUNCNAME, Final, DIR(VerLink, HorLink))<Dst>(),
        finalParams,
        kit.gpuCurrentStream,
        stdPass
    );
}

#endif

//----------------------------------------------------------------

#if HOSTCODE

INSTANTIATE_FUNC_EX((FUNCNAME<uint8, uint8>), FUNCNAME)
INSTANTIATE_FUNC_EX((FUNCNAME<int8, int8>), FUNCNAME)
INSTANTIATE_FUNC_EX((FUNCNAME<uint8, float16>), FUNCNAME)

INSTANTIATE_FUNC_EX((FUNCNAME<int8_x2, int8_x2>), FUNCNAME)
INSTANTIATE_FUNC_EX((FUNCNAME<int8_x2, float16_x2>), FUNCNAME)

INSTANTIATE_FUNC_EX((FUNCNAME<float16, float16>), FUNCNAME)
INSTANTIATE_FUNC_EX((FUNCNAME<float16_x2, float16_x2>), FUNCNAME)

#endif

//================================================================
//
// Undefs
//
//================================================================

#undef DIR
#undef LINEAR_ONLY
#undef LINEAR_ONLY_COMMA

//----------------------------------------------------------------

#ifdef FUNCSPACE
}
#endif
