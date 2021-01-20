#include "downsampleOneAndHalf.h"

#include "gaussSincResampling/gaussSincResamplingSettings.h"
#include "gaussSincResampling/common/allTypes.h"

using namespace gaussSincResampling;

//================================================================
//
// downsampleOneAndHalfConservative
//
//================================================================

COMPILE_ASSERT(GAUSS_SINC_RESAMPLING_QUALITY == 0);
static devConstant float32 conservativeFilter0[] = {+0.00110935f, +0.00224565f, -0.00295087f, -0.00482240f, +0.00710520f, +0.00945891f, -0.01576861f, -0.01727902f, +0.03333590f, +0.03072488f, -0.07264324f, -0.06101802f, +0.22472360f, +0.49828885f, +0.39333031f, +0.05504308f, -0.09948469f, -0.01653026f, +0.04718953f, +0.00547456f, -0.02404282f, -0.00137333f, +0.01186880f, +0.00000000f, -0.00546444f, +0.00028981f, +0.00230354f, -0.00023435f, -0.00087992f};
static devConstant float32 conservativeFilter1[] = {-0.00087992f, -0.00023435f, +0.00230354f, +0.00028981f, -0.00546444f, -0.00000000f, +0.01186880f, -0.00137333f, -0.02404282f, +0.00547456f, +0.04718953f, -0.01653026f, -0.09948469f, +0.05504308f, +0.39333031f, +0.49828885f, +0.22472360f, -0.06101802f, -0.07264324f, +0.03072488f, +0.03333590f, -0.01727902f, -0.01576861f, +0.00945891f, +0.00710520f, -0.00482240f, -0.00295087f, +0.00224565f, +0.00110935f};
static const Space conservativeFilterSrcShift = -13;

//----------------------------------------------------------------

#define FUNCSPACE gaussSincResampling

#define PACK_SIZE 2
#define PACK_TO_SRC_FACTOR 3

#define FILTER0 conservativeFilter0
#define FILTER1 conservativeFilter1
#define FILTER_SRC_SHIFT conservativeFilterSrcShift

#define HORIZONTAL_FIRST 1

//----------------------------------------------------------------

#undef TASK_COUNT
#define TASK_COUNT 1

#undef FUNCNAME
#define FUNCNAME downsampleOneAndHalfConservative1

# include "rationalResample/rationalResampleMultiple.inl"

////

#undef TASK_COUNT
#define TASK_COUNT 2

#undef FUNCNAME
#define FUNCNAME downsampleOneAndHalfConservative2

# include "rationalResample/rationalResampleMultiple.inl"

////

#undef TASK_COUNT
#define TASK_COUNT 3

#undef FUNCNAME
#define FUNCNAME downsampleOneAndHalfConservative3

# include "rationalResample/rationalResampleMultiple.inl"

////

#undef TASK_COUNT
#define TASK_COUNT 4

#undef FUNCNAME
#define FUNCNAME downsampleOneAndHalfConservative4

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
// downsampleOneAndHalfConservativeMultitask
//
//================================================================

template <typename Src, typename Interm, typename Dst>              
stdbool downsampleOneAndHalfConservativeMultitask(const GpuLayeredMatrix<const Src>& src, const GpuLayeredMatrix<Dst>& dst, BorderMode borderMode, stdPars(GpuProcessKit))
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
            auto func = downsampleOneAndHalfConservative##n<Src, Interm, Dst>; \
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
    INSTANTIATE_FUNC_EX((downsampleOneAndHalfConservativeMultitask<Src, Interm, Dst>), Dst)

FOREACH_TYPE(TMP_MACRO)

#undef TMP_MACRO

//----------------------------------------------------------------

}

#endif
