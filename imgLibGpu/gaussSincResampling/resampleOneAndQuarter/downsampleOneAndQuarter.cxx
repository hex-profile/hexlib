#include "downsampleOneAndQuarter.h"

#include "gaussSincResampling/gaussSincResamplingSettings.h"
#include "gaussSincResampling/common/allTypes.h"

using namespace gaussSincResampling;

//================================================================
//
// downsampleOneAndQuarterConservative
//
//================================================================

COMPILE_ASSERT(GAUSS_SINC_RESAMPLING_QUALITY == 0);
static devConstant float32 conservativeFilter0[] = {+0.00241186f, -0.00565753f, +0.00000000f, +0.01435584f, -0.01565120f, -0.01660933f, +0.05355094f, -0.01984877f, -0.11210227f, +0.22673374f, +0.60919981f, +0.35535279f, -0.07326762f, -0.06169844f, +0.05600034f, -0.00000000f, -0.02450656f, +0.01135782f, +0.00509503f, -0.00678908f, +0.00099145f, +0.00200081f, -0.00113536f, -0.00018091f, +0.00039665f};
static devConstant float32 conservativeFilter1[] = {+0.00000000f, +0.00297905f, -0.00354263f, -0.00403734f, +0.01364123f, -0.00508064f, -0.02653388f, +0.04002093f, +0.01775756f, -0.11813595f, +0.10336638f, +0.56109975f, +0.47220693f, +0.00000000f, -0.09806011f, +0.04415521f, +0.02038911f, -0.02886426f, +0.00456581f, +0.01010837f, -0.00634871f, -0.00112687f, +0.00276548f, -0.00074093f, -0.00058448f};
static devConstant float32 conservativeFilter2[] = {-0.00058448f, -0.00074093f, +0.00276548f, -0.00112687f, -0.00634871f, +0.01010837f, +0.00456581f, -0.02886426f, +0.02038911f, +0.04415521f, -0.09806011f, +0.00000000f, +0.47220693f, +0.56109975f, +0.10336638f, -0.11813595f, +0.01775756f, +0.04002093f, -0.02653388f, -0.00508064f, +0.01364123f, -0.00403734f, -0.00354263f, +0.00297905f, +0.00000000f};
static devConstant float32 conservativeFilter3[] = {+0.00039665f, -0.00018091f, -0.00113536f, +0.00200081f, +0.00099145f, -0.00678908f, +0.00509503f, +0.01135782f, -0.02450656f, -0.00000000f, +0.05600034f, -0.06169844f, -0.07326762f, +0.35535279f, +0.60919981f, +0.22673374f, -0.11210227f, -0.01984877f, +0.05355094f, -0.01660933f, -0.01565120f, +0.01435584f, +0.00000000f, -0.00565753f, +0.00241186f};
static const Space conservativeFilterSrcShift = -10;

//----------------------------------------------------------------

#define FUNCSPACE gaussSincResampling

#define PACK_SIZE 4
#define PACK_TO_SRC_FACTOR 5

#define FILTER0 conservativeFilter0
#define FILTER1 conservativeFilter1
#define FILTER2 conservativeFilter2
#define FILTER3 conservativeFilter3
#define FILTER_SRC_SHIFT conservativeFilterSrcShift

#define HORIZONTAL_FIRST 1

//----------------------------------------------------------------

#undef TASK_COUNT
#define TASK_COUNT 1

#undef FUNCNAME
#define FUNCNAME downsampleOneAndQuarterConservative1

# include "rationalResample/rationalResampleMultiple.inl"

////

#undef TASK_COUNT
#define TASK_COUNT 2

#undef FUNCNAME
#define FUNCNAME downsampleOneAndQuarterConservative2

# include "rationalResample/rationalResampleMultiple.inl"

////

#undef TASK_COUNT
#define TASK_COUNT 3

#undef FUNCNAME
#define FUNCNAME downsampleOneAndQuarterConservative3

# include "rationalResample/rationalResampleMultiple.inl"

////

#undef TASK_COUNT
#define TASK_COUNT 4

#undef FUNCNAME
#define FUNCNAME downsampleOneAndQuarterConservative4

# include "rationalResample/rationalResampleMultiple.inl"

////

#undef TASK_COUNT

//================================================================
//
// Host part
//
//================================================================

#if HOSTCODE

namespace gaussSincResampling {

//================================================================
//
// downsampleOneAndQuarterConservativeMultitask
//
//================================================================

template <typename Src, typename Interm, typename Dst>              
stdbool downsampleOneAndQuarterConservativeMultitask(const GpuLayeredMatrix<const Src>& src, const GpuLayeredMatrix<Dst>& dst, BorderMode borderMode, stdPars(GpuProcessKit))
{
    REQUIRE(equalLayers(src, dst));
    auto layers = dst.layers();

    ////

    #define TMP_PASS(i, _) \
        src.getLayer(i), dst.getLayer(i)

    #define TMP_MACRO(n, _) \
        \
        else if (layers == n) \
        { \
            auto func = downsampleOneAndQuarterConservative##n<Src, Interm, Dst>; \
            require(func(PREP_ENUM(n, TMP_PASS, _), borderMode, stdPass)); \
        }

    ////

    #define MAX_TASKS 4
    
    if (layers == 0)
    {
    }
    PREP_FOR2_FROM1_TO_COUNT(MAX_TASKS, TMP_MACRO, _)
    else
    {
        REQUIRE(false);
    }

    #undef TMP_MACRO
    #undef TMP_PASS

    returnTrue;
}

//----------------------------------------------------------------

#define TMP_MACRO(Src, Interm, Dst, _) \
    INSTANTIATE_FUNC_EX((downsampleOneAndQuarterConservativeMultitask<Src, Interm, Dst>), Dst)

FOREACH_TYPE(TMP_MACRO)

#undef TMP_MACRO

//----------------------------------------------------------------

}

#endif
