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
#include "kit/kit.h"
#include "mapDownsampleIndexToSource.h"
#include "prepTools/prepIncDec.h"
#include "vectorTypes/vectorOperations.h"

namespace PREP_PASTE(FUNCNAME, Space) {

//================================================================
//
// TASK_COUNT
//
//================================================================

#ifndef TASK_COUNT
    #error
#endif

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

#define TMP_MACRO(t, _) \
    devDefineSampler(PREP_PASTE(FUNCNAME, srcSampler1##t), DevSampler2D, DevSamplerFloat, 1) \
    devDefineSampler(PREP_PASTE(FUNCNAME, srcSampler2##t), DevSampler2D, DevSamplerFloat, 2)

PREP_FOR(TASK_COUNT, TMP_MACRO, _)

#undef TMP_MACRO

////

#define TMP_MACRO(t, k, _) \
    devDefineSampler(PREP_PASTE(FUNCNAME, intermSampler##k##1##t), DevSampler2D, DevSamplerFloat, 1) \
    devDefineSampler(PREP_PASTE(FUNCNAME, intermSampler##k##2##t), DevSampler2D, DevSamplerFloat, 2)

PREP_FOR_2D(TASK_COUNT, FILTER_COUNT, TMP_MACRO, _)

#undef TMP_MACRO

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

    GpuMatrix<Dst> dst[TASK_COUNT][FILTER_COUNT];
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

    GpuMatrix<Dst> dst[TASK_COUNT][FILTER_COUNT];

#else

    float32 filterMixCoeffs[FILTER_COUNT];

    GpuMatrix<const Dst> dstMixImage[TASK_COUNT];
    float32 dstMixCoeff[TASK_COUNT];
    GpuMatrix<Dst> dst[TASK_COUNT];

#endif

};

//================================================================
//
// Instances
//
//================================================================

#define RANK 1

#define HORIZONTAL 1
# include "multiFilterKernelMultiple.inl"
#undef HORIZONTAL

#define HORIZONTAL 0
# include "multiFilterKernelMultiple.inl"
#undef HORIZONTAL

#undef RANK

//----------------------------------------------------------------

#define RANK 2

#define HORIZONTAL 1
# include "multiFilterKernelMultiple.inl"
#undef HORIZONTAL

#define HORIZONTAL 0
# include "multiFilterKernelMultiple.inl"
#undef HORIZONTAL

#undef RANK

//================================================================
//
// Kernel instantiations (Interm)
//
//================================================================

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, IntermParallelHor1), IntermParams, PREP_PASTE(FUNCNAME, IntermHorLink), PREP_PASTE(FUNCNAME, IntermHor8u), (uint8));
GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, IntermParallelVer1), IntermParams, PREP_PASTE(FUNCNAME, IntermVerLink), PREP_PASTE(FUNCNAME, IntermVer8u), (uint8));
                                                                                                                                              

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, IntermParallelHor2), IntermParams, PREP_PASTE(FUNCNAME, IntermHorLink), PREP_PASTE(FUNCNAME, IntermHor8s_x2), (int8_x2));
GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, IntermParallelVer2), IntermParams, PREP_PASTE(FUNCNAME, IntermVerLink), PREP_PASTE(FUNCNAME, IntermVer8s_x2), (int8_x2));
                                                                                                                                              

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, IntermParallelHor1), IntermParams, PREP_PASTE(FUNCNAME, IntermHorLink), PREP_PASTE(FUNCNAME, IntermHor16f), (float16));
GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, IntermParallelVer1), IntermParams, PREP_PASTE(FUNCNAME, IntermVerLink), PREP_PASTE(FUNCNAME, IntermVer16f), (float16));
                                                                                                                                              

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, IntermParallelHor2), IntermParams, PREP_PASTE(FUNCNAME, IntermHorLink), PREP_PASTE(FUNCNAME, IntermHor16f_x2), (float16_x2));
GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, IntermParallelVer2), IntermParams, PREP_PASTE(FUNCNAME, IntermVerLink), PREP_PASTE(FUNCNAME, IntermVer16f_x2), (float16_x2));

//================================================================
//
// Kernel instantiations (Final)
//
//================================================================

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, FinalParallelHor1), FinalParams, PREP_PASTE(FUNCNAME, FinalHorLink), PREP_PASTE(FUNCNAME, FinalHor8u), (uint8));
GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, FinalParallelVer1), FinalParams, PREP_PASTE(FUNCNAME, FinalVerLink), PREP_PASTE(FUNCNAME, FinalVer8u), (uint8));

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, FinalParallelHor2), FinalParams, PREP_PASTE(FUNCNAME, FinalHorLink), PREP_PASTE(FUNCNAME, FinalHor8s_x2), (int8_x2));
GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, FinalParallelVer2), FinalParams, PREP_PASTE(FUNCNAME, FinalVerLink), PREP_PASTE(FUNCNAME, FinalVer8s_x2), (int8_x2));

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, FinalParallelHor1), FinalParams, PREP_PASTE(FUNCNAME, FinalHorLink), PREP_PASTE(FUNCNAME, FinalHor16f), (float16));
GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, FinalParallelVer1), FinalParams, PREP_PASTE(FUNCNAME, FinalVerLink), PREP_PASTE(FUNCNAME, FinalVer16f), (float16));

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, FinalParallelHor2), FinalParams, PREP_PASTE(FUNCNAME, FinalHorLink), PREP_PASTE(FUNCNAME, FinalHor16f_x2), (float16_x2));
GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, FinalParallelVer2), FinalParams, PREP_PASTE(FUNCNAME, FinalVerLink), PREP_PASTE(FUNCNAME, FinalVer16f_x2), (float16_x2));

//================================================================
//
// DIR
//
//================================================================

#undef DIR

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
// FUNC_PARAMETERS
//
//================================================================

#undef FUNC_PARAMETERS

#ifndef LINEAR_COMBINATION

    #define FUNC_PARAMETERS(t, _) \
        const GpuMatrix<const Src>& src##t, \
        PREP_ENUM_LR(FILTER_COUNT, const GpuMatrix<Dst>& dst, t), 

#else

    #define FUNC_PARAMETERS(t, _) \
        const GpuMatrix<const Src>& src##t, \
        const GpuMatrix<const Dst>& dstMixImage##t, \
        float32 dstMixCoeff##t, \
        const GpuMatrix<Dst>& dst##t,

#endif

//================================================================
//
// Main func
//
//================================================================

#if HOSTCODE

template <typename Src, typename Dst>
stdbool FUNCNAME
(
    PREP_FOR1(TASK_COUNT, FUNC_PARAMETERS, _)
    LINEAR_ONLY_COMMA(const float32* filterMixCoeffs)
    BorderMode borderMode, 
    stdPars(GpuProcessKit)
)
{
    using namespace PREP_PASTE(FUNCNAME, Space);

    ////

    Point<Space> srcSize = src0.size();
    #define TMP_MACRO(t, _) REQUIRE(equalSize(src##t, srcSize));
    #undef TMP_MACRO

    ////

#ifndef LINEAR_COMBINATION
    Point<Space> dstSize = dst00.size();
    #define TMP_MACRO(t, k, _) REQUIRE(equalSize(dst##k##t, dstSize));
    PREP_FOR_2D(TASK_COUNT, FILTER_COUNT, TMP_MACRO, _)
    #undef TMP_MACRO
#else
    Point<Space> dstSize = dst0.size();
    #define TMP_MACRO(t, _) REQUIRE(equalSize(dst##t, dstMixImage##t, dstSize));
    #undef TMP_MACRO
#endif

    ////

    Point<Space> intermSize = srcSize;
    intermSize.DIR(X, Y) = dstSize.DIR(X, Y);

    GPU_LAYERED_MATRIX_ALLOC(interm, Dst, TASK_COUNT * FILTER_COUNT, intermSize);

    ////

    if_not (kit.dataProcessing)
        returnTrue;

    ////

    const int srcRank = VectorTypeRank<Src>::val;

    //----------------------------------------------------------------
    //
    // Interm
    //
    //----------------------------------------------------------------

    #define TMP_MACRO(t, _) \
        { \
            const GpuSamplerLink* srcSampler = (srcRank == 1) ? soft_cast<const GpuSamplerLink*>(&PREP_PASTE(FUNCNAME, srcSampler1##t)) : &PREP_PASTE(FUNCNAME, srcSampler2##t); \
            require(kit.gpuSamplerSetting.setSamplerImage(*srcSampler, src##t, borderMode, false, true, true, stdPass)); \
        }

    PREP_FOR(TASK_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO

    ////

    IntermParams<Dst> intermParams;
    intermParams.srcTexstep = computeTexstep(srcSize);
    intermParams.dstSize = intermSize;

    for (Space t = 0; t < TASK_COUNT; ++t)
    {
        for (Space k = 0; k < FILTER_COUNT; ++k)
            intermParams.dst[t][k] = interm.getLayer(k + t * FILTER_COUNT);
    }

    ////

    Point<Space> intermLaunchSize = srcSize;
    intermLaunchSize.DIR(X, Y) = dstSize.DIR(X, Y);

    Point<Space> intermThreadCount = DIR(horThreadCount(), verThreadCount());
    Point<Space> intermGroupCount = divUpNonneg(intermLaunchSize, intermThreadCount);

    require
    (
        kit.gpuKernelCalling.callKernel
        (
            point3D(intermGroupCount.X, intermGroupCount.Y, TASK_COUNT),
            intermThreadCount,
            areaOf(intermSize),
            PREP_PASTE3(FUNCNAME, Interm, DIR(HorLink, VerLink))<Dst>(),
            intermParams,
            kit.gpuCurrentStream,
            stdPass
        )
    );

    //----------------------------------------------------------------
    //
    // Final
    //
    //----------------------------------------------------------------

    #define TMP_MACRO(t, k, _) \
        { \
            const GpuSamplerLink* sampler = 0; \
            if (srcRank == 1) sampler = &PREP_PASTE(FUNCNAME, intermSampler##k##1##t); \
            if (srcRank == 2) sampler = &PREP_PASTE(FUNCNAME, intermSampler##k##2##t); \
            REQUIRE(sampler != 0); \
            require(kit.gpuSamplerSetting.setSamplerImage(*sampler, makeConst(interm.getLayer(k + t * FILTER_COUNT)), borderMode, false, true, true, stdPass)); \
        }

    PREP_FOR_2D(TASK_COUNT, FILTER_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO

    ////

    FinalParams<Dst> finalParams;
    finalParams.srcTexstep = computeTexstep(intermSize);
    finalParams.dstSize = dstSize;

#ifndef LINEAR_COMBINATION

    #define TMP_MACRO(t, k, _) \
        finalParams.dst[t][k] = dst##k##t;

    PREP_FOR_2D(TASK_COUNT, FILTER_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO

#else

    for (Space k = 0; k < FILTER_COUNT; ++k)
        finalParams.filterMixCoeffs[k] = filterMixCoeffs[k];

    #define TMP_MACRO(t, _) \
        finalParams.dstMixImage[t] = dstMixImage##t; \
        finalParams.dstMixCoeff[t] = dstMixCoeff##t; \
        finalParams.dst[t] = dst##t;

    PREP_FOR(TASK_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO

#endif

    ////

    Point<Space> finalThreadCount = DIR(verThreadCount(), horThreadCount());

    Point<Space> finalLaunchSize = dstSize;
    finalLaunchSize.DIR(Y, X) = dstSize.DIR(Y, X);

    Point<Space> finalGroupCount = divUpNonneg(finalLaunchSize, finalThreadCount);

    ////

    require
    (
        kit.gpuKernelCalling.callKernel
        (
            point3D(finalGroupCount.X, finalGroupCount.Y, TASK_COUNT),
            finalThreadCount,
            areaOf(dstSize),
            PREP_PASTE3(FUNCNAME, Final, DIR(VerLink, HorLink))<Dst>(),
            finalParams,
            kit.gpuCurrentStream,                                          
            stdPass
        )
    );

    ////

    returnTrue;
}

#endif

//----------------------------------------------------------------

#if HOSTCODE 

INSTANTIATE_FUNC((FUNCNAME<uint8, uint8>))
INSTANTIATE_FUNC((FUNCNAME<uint8, float16>))

INSTANTIATE_FUNC((FUNCNAME<int8_x2, int8_x2>))
INSTANTIATE_FUNC((FUNCNAME<int8_x2, float16_x2>))

INSTANTIATE_FUNC((FUNCNAME<float16, float16>))
INSTANTIATE_FUNC((FUNCNAME<float16_x2, float16_x2>))

#endif
