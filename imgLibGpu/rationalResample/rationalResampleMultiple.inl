#if HOSTCODE
#include "gpuAppliedApi/gpuAppliedApi.h"
#include "numbers/divRound.h"
#include "errorLog/errorLog.h"
#include "dataAlloc/gpuMatrixMemory.h"
#endif

#include "data/gpuMatrix.h"
#include "gpuDevice/gpuDevice.h"
#include "gpuDevice/loadstore/storeNorm.h"
#include "gpuSupport/gpuTemplateKernel.h"
#include "gpuSupport/gpuTexTools.h"
#include "gpuSupport/parallelLoop.h"
#include "kit/kit.h"
#include "numbers/int/intType.h"
#include "rndgen/rndgenFloat.h"
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
// OUTPUT_FACTOR
//
//================================================================

#ifdef OUTPUT_FACTOR
    #define OUTPUT_FACTOR_ONLY(s) s
    #define OUTPUT_FACTOR_ONLY_COMMA(s) s,
#else 
    #define OUTPUT_FACTOR_ONLY(s)
    #define OUTPUT_FACTOR_ONLY_COMMA(s)
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
    devDefineSampler(PREP_PASTE3(FUNCNAME, srcSampler1, t), DevSampler2D, DevSamplerFloat, 1) \
    devDefineSampler(PREP_PASTE3(FUNCNAME, srcSampler2, t), DevSampler2D, DevSamplerFloat, 2)

PREP_FOR(TASK_COUNT, TMP_MACRO, _)

#undef TMP_MACRO

//================================================================
//
// ResampleParams
//
//================================================================

template <typename Dst>
struct ResampleParams
{
    Point<float32> srcTexstep; 
    GpuMatrix<Dst> dst[TASK_COUNT];
    OUTPUT_FACTOR_ONLY(float32 outputFactor;)
};

//================================================================
//
// Instances
//
//================================================================

#define RANK 1

#define HORIZONTAL 1
# include "rationalResampleKernelMultiple.inl"
#undef HORIZONTAL

#define HORIZONTAL 0
# include "rationalResampleKernelMultiple.inl"
#undef HORIZONTAL

#undef RANK

//----------------------------------------------------------------

#define RANK 2

#define HORIZONTAL 1
# include "rationalResampleKernelMultiple.inl"
#undef HORIZONTAL

#define HORIZONTAL 0
# include "rationalResampleKernelMultiple.inl"
#undef HORIZONTAL

#undef RANK

//================================================================
//
// Kernel instantiations
//
//================================================================

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, Hor1), ResampleParams, PREP_PASTE(FUNCNAME, HorLink), PREP_PASTE(FUNCNAME, Hor8u), (uint8));
GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, Ver1), ResampleParams, PREP_PASTE(FUNCNAME, VerLink), PREP_PASTE(FUNCNAME, Ver8u), (uint8));

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, Hor1), ResampleParams, PREP_PASTE(FUNCNAME, HorLink), PREP_PASTE(FUNCNAME, Hor8s), (int8));
GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, Ver1), ResampleParams, PREP_PASTE(FUNCNAME, VerLink), PREP_PASTE(FUNCNAME, Ver8s), (int8));

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, Hor2), ResampleParams, PREP_PASTE(FUNCNAME, HorLink), PREP_PASTE(FUNCNAME, Hor8s_x2), (int8_x2));
GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, Ver2), ResampleParams, PREP_PASTE(FUNCNAME, VerLink), PREP_PASTE(FUNCNAME, Ver8s_x2), (int8_x2));
                                                                                                                                          

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, Hor1), ResampleParams, PREP_PASTE(FUNCNAME, HorLink), PREP_PASTE(FUNCNAME, Hor16f), (float16));
GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, Ver1), ResampleParams, PREP_PASTE(FUNCNAME, VerLink), PREP_PASTE(FUNCNAME, Ver16f), (float16));

GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, Hor2), ResampleParams, PREP_PASTE(FUNCNAME, HorLink), PREP_PASTE(FUNCNAME, Hor16f_x2), (float16_x2));
GPU_TEMPLATE_KERNEL_INST(((typename, Dst)), PREP_PASTE(FUNCNAME, Ver2), ResampleParams, PREP_PASTE(FUNCNAME, VerLink), PREP_PASTE(FUNCNAME, Ver16f_x2), (float16_x2));

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
// Main func
//
//================================================================

#if HOSTCODE

template <typename Src, typename Interm, typename Dst>
stdbool FUNCNAME
(
    #define TMP_MACRO(t, _) \
        const GpuMatrix<const Src>& src##t, \
        const GpuMatrix<Dst>& dst##t,

    PREP_FOR(TASK_COUNT, TMP_MACRO, _)
    

    #undef TMP_MACRO

    OUTPUT_FACTOR_ONLY_COMMA(const Point<float32>& outputFactor)
    BorderMode borderMode,
    stdPars(GpuProcessKit)
)
{
    stdBegin;

    using namespace PREP_PASTE(FUNCNAME, Space);

    ////

    Point<Space> srcSize = src0.size();
    Point<Space> dstSize = dst0.size();

    #define TMP_MACRO(t, _) \
        REQUIRE(equalSize(srcSize, src##t.size())); \
        REQUIRE(equalSize(dstSize, dst##t.size()));

    PREP_FOR(TASK_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO

    //----------------------------------------------------------------
    //
    // Common samplers
    //
    //----------------------------------------------------------------

    const int srcRank = VectorTypeRank<Src>::val;

    #define TMP_MACRO(t, _) \
        const GpuSamplerLink* srcSampler##t = (srcRank == 1) ? soft_cast<const GpuSamplerLink*>(&PREP_PASTE3(FUNCNAME, srcSampler1, t)) : &PREP_PASTE3(FUNCNAME, srcSampler2, t);

    PREP_FOR(TASK_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO

    //----------------------------------------------------------------
    //
    // Interm pass
    //
    //----------------------------------------------------------------

    Point<Space> intermSize = srcSize;
    intermSize.DIR(X, Y) = dstSize.DIR(X, Y);

    #define TMP_MACRO(t, _) \
        GPU_MATRIX_ALLOC(interm##t, Interm, intermSize);

    PREP_FOR(TASK_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO

    ////

    if_not (kit.dataProcessing) // all allocations done
        return true;

    ////

    Point<Space> dstSizeInPacks = divUpNonneg(dstSize, point(PACK_SIZE));

    ////

    Point<Space> intermLaunchSize = srcSize;
    intermLaunchSize.DIR(X, Y) = dstSizeInPacks.DIR(X, Y);

    Point<Space> intermThreadCount = DIR(horThreadCount(), verThreadCount());
    Point<Space> intermGroupCount = divUpNonneg(intermLaunchSize, intermThreadCount);

    ////

    ResampleParams<Interm> intermParams;
    intermParams.srcTexstep = computeTexstep(srcSize);
    OUTPUT_FACTOR_ONLY(intermParams.outputFactor = outputFactor.DIR(X, Y));

    #define TMP_MACRO(t, _) \
        require(kit.gpuSamplerSetting.setSamplerImage(*srcSampler##t, src##t, borderMode, false, true, true, stdPass)); \
        intermParams.dst[t] = interm##t;

    PREP_FOR(TASK_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO


    ////

    require
    (
        kit.gpuKernelCalling.callKernel
        (
            point3D(intermGroupCount.X, intermGroupCount.Y, TASK_COUNT),
            intermThreadCount,
            areaOf(intermSize),
            PREP_PASTE(FUNCNAME, DIR(HorLink, VerLink))<Interm>(),
            intermParams,
            kit.gpuCurrentStream,
            stdPass
        )
    );

    //----------------------------------------------------------------
    //
    // Final pass
    //
    //----------------------------------------------------------------

    Point<Space> finalLaunchSize = dstSize;
    finalLaunchSize.DIR(Y, X) = dstSizeInPacks.DIR(Y, X);

    Point<Space> finalThreadCount = DIR(verThreadCount(), horThreadCount());
    Point<Space> finalGroupCount = divUpNonneg(finalLaunchSize, finalThreadCount);

    ////

    ResampleParams<Dst> finalParams;
    finalParams.srcTexstep = computeTexstep(intermSize);
    OUTPUT_FACTOR_ONLY(finalParams.outputFactor = outputFactor.DIR(Y, X));
    

    #define TMP_MACRO(t, _) \
        require(kit.gpuSamplerSetting.setSamplerImage(*srcSampler##t, makeConst(interm##t), borderMode, false, true, true, stdPass)); \
        finalParams.dst[t] = dst##t;

    PREP_FOR(TASK_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO

    ////

    require
    (
        kit.gpuKernelCalling.callKernel
        (
            point3D(finalGroupCount.X, finalGroupCount.Y, TASK_COUNT),
            finalThreadCount,
            areaOf(dstSize),
            PREP_PASTE(FUNCNAME, DIR(VerLink, HorLink))<Dst>(),
            finalParams,
            kit.gpuCurrentStream,                                          
            stdPass
        )
    );

    ////

    stdEnd;
}

//----------------------------------------------------------------

#endif
