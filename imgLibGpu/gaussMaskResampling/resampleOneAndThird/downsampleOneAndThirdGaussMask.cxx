#include "downsampleOneAndThirdGaussMask.h"

#include "gpuDevice/gpuDevice.h"
#include "prepTools/prepFor.h"
#include "gaussMaskResampling/common/allTypes.h"

using namespace gaussMaskResampling;

//================================================================
//
// downsampleOneAndThirdGaussMaskInitial
//
//================================================================

static devConstant float32 initialFilter0[] = {+0.01273839f, +0.17222247f, +0.48806752f, +0.28992447f, +0.03609979f, +0.00094219f, +0.00000515f, +0.00000001f};
static devConstant float32 initialFilter1[] = {+0.00003479f, +0.00377785f, +0.08598342f, +0.41020394f, +0.41020394f, +0.08598342f, +0.00377785f, +0.00003479f};
static devConstant float32 initialFilter2[] = {+0.00000001f, +0.00000515f, +0.00094219f, +0.03609979f, +0.28992447f, +0.48806752f, +0.17222247f, +0.01273839f};
static const Space initialFilterShift = -2;

#define FILTER0 initialFilter0
#define FILTER1 initialFilter1
#define FILTER2 initialFilter2
#define FILTER_SRC_SHIFT initialFilterShift

////

#define FUNCSPACE gaussMaskResampling

#define PACK_SIZE 3
#define PACK_TO_SRC_FACTOR 4

#define HORIZONTAL_FIRST 1

////

#undef FUNCNAME
#define FUNCNAME downsampleOneAndThirdGaussMaskInitial1

#undef TASK_COUNT
#define TASK_COUNT 1
# include "rationalResample/rationalResampleMultiple.inl"

////

#undef FUNCNAME
#define FUNCNAME downsampleOneAndThirdGaussMaskInitial2

#undef TASK_COUNT
#define TASK_COUNT 2

# include "rationalResample/rationalResampleMultiple.inl"

////

#undef FUNCNAME
#define FUNCNAME downsampleOneAndThirdGaussMaskInitial3

#undef TASK_COUNT
#define TASK_COUNT 3

# include "rationalResample/rationalResampleMultiple.inl"

////

#undef FUNCNAME
#define FUNCNAME downsampleOneAndThirdGaussMaskInitial4

#undef TASK_COUNT
#define TASK_COUNT 4

# include "rationalResample/rationalResampleMultiple.inl"

//================================================================
//
// downsampleOneAndThirdGaussMaskSustaining
//
//================================================================

static devConstant float32 sustainingFilter0[] = {+0.06608526f, +0.71472518f, +0.21733108f, +0.00185803f, +0.00000045f, +0.00000000f};
static devConstant float32 sustainingFilter1[] = {+0.00001081f, +0.01367310f, +0.48631609f, +0.48631609f, +0.01367310f, +0.00001081f};
static devConstant float32 sustainingFilter2[] = {+0.00000000f, +0.00000045f, +0.00185803f, +0.21733108f, +0.71472518f, +0.06608526f};
static const Space sustainingSrcShift = -1;

//----------------------------------------------------------------

#undef FUNCNAME

#undef FILTER0
#undef FILTER1
#undef FILTER2
#undef FILTER_SRC_SHIFT

#define FILTER0 sustainingFilter0
#define FILTER1 sustainingFilter1
#define FILTER2 sustainingFilter2
#define FILTER_SRC_SHIFT sustainingSrcShift

////

#undef FUNCNAME
#define FUNCNAME downsampleOneAndThirdGaussMaskSustaining1

#undef TASK_COUNT
#define TASK_COUNT 1

# include "rationalResample/rationalResampleMultiple.inl"

////

#undef FUNCNAME
#define FUNCNAME downsampleOneAndThirdGaussMaskSustaining2

#undef TASK_COUNT
#define TASK_COUNT 2

# include "rationalResample/rationalResampleMultiple.inl"

////

#undef FUNCNAME
#define FUNCNAME downsampleOneAndThirdGaussMaskSustaining3

#undef TASK_COUNT
#define TASK_COUNT 3

# include "rationalResample/rationalResampleMultiple.inl"

////

#undef FUNCNAME
#define FUNCNAME downsampleOneAndThirdGaussMaskSustaining4

#undef TASK_COUNT
#define TASK_COUNT 4

# include "rationalResample/rationalResampleMultiple.inl"

//================================================================
//
// Host part
//
//================================================================

#if HOSTCODE

namespace gaussMaskResampling {

//================================================================
//
// downsampleOneAndThirdGaussMaskMultitask
//
//================================================================

template <typename Src, typename Interm, typename Dst>              
stdbool downsampleOneAndThirdGaussMaskMultitask(const GpuLayeredMatrix<const Src>& src, const GpuLayeredMatrix<Dst>& dst, BorderMode borderMode, bool initial, stdPars(GpuProcessKit))
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
            auto func = initial ? \
                downsampleOneAndThirdGaussMaskInitial##n<Src, Interm, Dst> : \
                downsampleOneAndThirdGaussMaskSustaining##n<Src, Interm, Dst>; \
            \
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
    INSTANTIATE_FUNC_EX((downsampleOneAndThirdGaussMaskMultitask<Src, Interm, Dst>), Dst)

FOREACH_TYPE(TMP_MACRO)

#undef TMP_MACRO

//----------------------------------------------------------------

}

#endif
