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

#define SIGNATURE PREP_PASTE_UNDER5(FUNCNAME, RANK, SRC_TYPE, INTERM_TYPE, DST_TYPE)

namespace SIGNATURE {

//================================================================
//
// Check parameters.
//
//================================================================

#if !(defined(SIGNATURE) && defined(FUNCNAME) && defined(RANK) && defined(SRC_TYPE) && defined(INTERM_TYPE) && defined(DST_TYPE))
    #error Parameters need to be defined.
#endif
                                         
//================================================================
//
// TASK_COUNT
//
//================================================================

#ifndef TASK_COUNT
    #define TASK_COUNT 1
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
    devDefineSampler(PREP_PASTE3(SIGNATURE, srcSampler, t), DevSampler2D, DevSamplerFloat, RANK) \

PREP_FOR(TASK_COUNT, TMP_MACRO, _)

#undef TMP_MACRO

////

#define TMP_MACRO(t, k, _) \
    devDefineSampler(PREP_PASTE4(SIGNATURE, intermSampler, k, t), DevSampler2D, DevSamplerFloat, RANK)

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

struct IntermParams
{
    Point<float32> srcTexstep;
    Point<Space> dstSize; 

    GpuMatrix<INTERM_TYPE> dst[TASK_COUNT][FILTER_COUNT];
};

//================================================================
//
// FinalParams
//
//================================================================

struct FinalParams
{

    Point<float32> srcTexstep;
    Point<Space> dstSize; 

#ifndef LINEAR_COMBINATION

    GpuMatrix<DST_TYPE> dst[TASK_COUNT][FILTER_COUNT];

#else

    float32 filterMixCoeffs[FILTER_COUNT];

    GpuMatrix<const DST_TYPE> dstMixImage[TASK_COUNT];
    float32 dstMixCoeff[TASK_COUNT];
    GpuMatrix<DST_TYPE> dst[TASK_COUNT];

#endif

};

//================================================================
//
// Instances
//
//================================================================

#define HORIZONTAL 1
# include "multiFilterKernel.inl"
#undef HORIZONTAL

#define HORIZONTAL 0
# include "multiFilterKernel.inl"
#undef HORIZONTAL

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

#define FUNC_PARAMETERS(t, types) \
    FUNC_PARAMETERS_EX(t, PREP_ARG3_0 types, PREP_ARG3_1 types, PREP_ARG3_2 types)

//----------------------------------------------------------------

#ifndef LINEAR_COMBINATION

    #define FUNC_PARAMETERS_EX(t, Src, Interm, Dst) \
        const GpuMatrix<const Src>& src##t, \
        PREP_ENUM_LR(FILTER_COUNT, const GpuMatrix<Dst>& dst, t), 

#else

    #define FUNC_PARAMETERS_EX(t, Src, Interm, Dst) \
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

//----------------------------------------------------------------

template <typename Src, typename Interm, typename Dst>
stdbool FUNCNAME
(
    PREP_FOR1(TASK_COUNT, FUNC_PARAMETERS, (Src, Interm, Dst))
    LINEAR_ONLY_COMMA(const float32* filterMixCoeffs)
    BorderMode borderMode, 
    stdPars(GpuProcessKit)
);

//----------------------------------------------------------------

template <>
stdbool FUNCNAME<SRC_TYPE, INTERM_TYPE, DST_TYPE>
(
    PREP_FOR1(TASK_COUNT, FUNC_PARAMETERS, (SRC_TYPE, INTERM_TYPE, DST_TYPE))
    LINEAR_ONLY_COMMA(const float32* filterMixCoeffs)
    BorderMode borderMode, 
    stdPars(GpuProcessKit)
)
{
    using namespace SIGNATURE;

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

    GPU_LAYERED_MATRIX_ALLOC(interm, INTERM_TYPE, TASK_COUNT * FILTER_COUNT, intermSize);

    ////

    if_not (kit.dataProcessing)
        returnTrue;

    ////

    REQUIRE(VectorTypeRank<SRC_TYPE>::val == RANK);
    REQUIRE(VectorTypeRank<INTERM_TYPE>::val == RANK);
    REQUIRE(VectorTypeRank<DST_TYPE>::val == RANK);

    //----------------------------------------------------------------
    //
    // Interm
    //
    //----------------------------------------------------------------

    #define TMP_MACRO(t, _) \
        { \
            const GpuSamplerLink* srcSampler = &PREP_PASTE3(SIGNATURE, srcSampler, t); \
            require(kit.gpuSamplerSetting.setSamplerImage(*srcSampler, src##t, borderMode, false, true, true, stdPass)); \
        }

    PREP_FOR(TASK_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO

    ////

    IntermParams intermParams;
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
            PREP_PASTE3(SIGNATURE, Interm, DIR(Hor, Ver)),
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
            const GpuSamplerLink* sampler = &PREP_PASTE4(SIGNATURE, intermSampler, k, t); \
            require(kit.gpuSamplerSetting.setSamplerImage(*sampler, makeConst(interm.getLayer(k + t * FILTER_COUNT)), borderMode, false, true, true, stdPass)); \
        }

    PREP_FOR_2D(TASK_COUNT, FILTER_COUNT, TMP_MACRO, _)

    #undef TMP_MACRO

    ////

    FinalParams finalParams;
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
            PREP_PASTE3(SIGNATURE, Final, DIR(Ver, Hor)),
            finalParams,
            kit.gpuCurrentStream,                                          
            stdPass
        )
    );

    ////

    returnTrue;
}

//----------------------------------------------------------------

INSTANTIATE_FUNC_EX((FUNCNAME<SRC_TYPE, INTERM_TYPE, DST_TYPE>), SIGNATURE)

//----------------------------------------------------------------

#endif

//================================================================
//
// Undefs
//
//================================================================

#undef FUNC_PARAMETERS
#undef FUNC_PARAMETERS_EX
#undef DIR
#undef LINEAR_ONLY
#undef LINEAR_ONLY_COMMA

//----------------------------------------------------------------

#ifdef FUNCSPACE
}
#endif
