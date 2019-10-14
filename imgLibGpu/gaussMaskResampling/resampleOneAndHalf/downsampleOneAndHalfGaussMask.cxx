#include "downsampleOneAndHalfGaussMask.h"

#include "gpuDevice/gpuDevice.h"
#include "prepTools/prepFor.h"

using namespace gaussMaskResampling;

//================================================================
//
// downsampleOneAndHalfGaussMaskInitial
//
//================================================================

static devConstant float32 initialFilter0[] = {+0.01948875f, +0.16907312f, +0.42677512f, +0.31344228f, +0.06698076f, +0.00416463f, +0.00007534f};
static devConstant float32 initialFilter1[] = {+0.00007534f, +0.00416463f, +0.06698076f, +0.31344228f, +0.42677512f, +0.16907312f, +0.01948875f};
static const Space initialSrcShift = -2;

//----------------------------------------------------------------

#define FUNCSPACE gaussMaskResampling

#define PACK_SIZE 2
#define PACK_TO_SRC_FACTOR 3

#define FILTER0 initialFilter0
#define FILTER1 initialFilter1
#define FILTER_SRC_SHIFT initialSrcShift

#define HORIZONTAL_FIRST 1

#define FOREACH_TYPE(action) \
    \
    action(uint8, uint8, uint8, 1) \
    action(float16, float16, float16, 1) \
    action(float32, float32, float32, 1) \
    action(float16_x4, float16_x4, float16_x4, 4)

////

#undef FUNCNAME
#define FUNCNAME downsampleOneAndHalfGaussMaskInitial1

#undef TASK_COUNT
#define TASK_COUNT 1
# include "rationalResample/rationalResampleMultiple.inl"

////

#undef FUNCNAME
#define FUNCNAME downsampleOneAndHalfGaussMaskInitial2

#undef TASK_COUNT
#define TASK_COUNT 2

# include "rationalResample/rationalResampleMultiple.inl"

////

#undef FUNCNAME
#define FUNCNAME downsampleOneAndHalfGaussMaskInitial3

#undef TASK_COUNT
#define TASK_COUNT 3

# include "rationalResample/rationalResampleMultiple.inl"

////

#undef FUNCNAME
#define FUNCNAME downsampleOneAndHalfGaussMaskInitial4

#undef TASK_COUNT
#define TASK_COUNT 4

# include "rationalResample/rationalResampleMultiple.inl"

////

COMPILE_ASSERT(downsampleOneAndHalfMaxTasks == 4);

//================================================================
//
// downsampleOneAndHalfGaussMaskSustaining
//
//================================================================

static devConstant float32 sustainingFilter0[] = {+0.00214486f, +0.10479062f, +0.55481288f, +0.31832579f, +0.01979239f, +0.00013336f, +0.00000010f};
static devConstant float32 sustainingFilter1[] = {+0.00000010f, +0.00013336f, +0.01979239f, +0.31832579f, +0.55481288f, +0.10479062f, +0.00214486f};
static const Space sustainingSrcShift = -2;

//----------------------------------------------------------------

#undef FUNCNAME

#undef FILTER0
#undef FILTER1
#undef FILTER_SRC_SHIFT

#define FILTER0 sustainingFilter0
#define FILTER1 sustainingFilter1
#define FILTER_SRC_SHIFT sustainingSrcShift

////

#undef FUNCNAME
#define FUNCNAME downsampleOneAndHalfGaussMaskSustaining1

#undef TASK_COUNT
#define TASK_COUNT 1

# include "rationalResample/rationalResampleMultiple.inl"

////

#undef FUNCNAME
#define FUNCNAME downsampleOneAndHalfGaussMaskSustaining2

#undef TASK_COUNT
#define TASK_COUNT 2

# include "rationalResample/rationalResampleMultiple.inl"

////

#undef FUNCNAME
#define FUNCNAME downsampleOneAndHalfGaussMaskSustaining3

#undef TASK_COUNT
#define TASK_COUNT 3

# include "rationalResample/rationalResampleMultiple.inl"

////

#undef FUNCNAME
#define FUNCNAME downsampleOneAndHalfGaussMaskSustaining4

#undef TASK_COUNT
#define TASK_COUNT 4

# include "rationalResample/rationalResampleMultiple.inl"

////

COMPILE_ASSERT(downsampleOneAndHalfMaxTasks == 4);

//================================================================
//
// Host part
//
//================================================================

#if HOSTCODE

namespace gaussMaskResampling {

//================================================================
//
// downsampleOneAndHalfGaussMaskMultitask
//
//================================================================

template <typename Src, typename Interm, typename Dst>              
stdbool downsampleOneAndHalfGaussMaskMultitask(const GpuLayeredMatrix<const Src>& src, const GpuLayeredMatrix<Dst>& dst, BorderMode borderMode, bool initial, stdPars(GpuProcessKit))
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
                downsampleOneAndHalfGaussMaskInitial##n<Src, Interm, Dst> : \
                downsampleOneAndHalfGaussMaskSustaining##n<Src, Interm, Dst>; \
            \
            require(func(PREP_ENUM(n, TMP_PASS, _), borderMode, stdPass)); \
        }

    ////

    #define MAX_TASKS 4
    COMPILE_ASSERT(MAX_TASKS == downsampleOneAndHalfMaxTasks);

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
    INSTANTIATE_FUNC_EX((downsampleOneAndHalfGaussMaskMultitask<Src, Interm, Dst>), Dst)

FOREACH_TYPE(TMP_MACRO)

#undef TMP_MACRO

//----------------------------------------------------------------

}

#endif
