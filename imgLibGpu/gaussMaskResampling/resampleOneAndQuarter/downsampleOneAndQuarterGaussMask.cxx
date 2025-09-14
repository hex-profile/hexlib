#include "downsampleOneAndQuarterGaussMask.h"

#include "gpuDevice/gpuDevice.h"
#include "prepTools/prepFor.h"
#include "gaussMaskResampling/common/allTypes.h"

using namespace gaussMaskResampling;

//================================================================
//
// downsampleOneAndQuarterGaussMaskInitial
//
//================================================================

static devConstant float32 initialFilter0[] = {+0.00960880f, +0.17270208f, +0.52462259f, +0.26935022f, +0.02337268f, +0.00034278f, +0.00000085f, +0.00000000f, +0.00000000f};
static devConstant float32 initialFilter1[] = {+0.00002131f, +0.00353471f, +0.09908366f, +0.46943045f, +0.37589052f, +0.05087125f, +0.00116360f, +0.00000450f, +0.00000000f};
static devConstant float32 initialFilter2[] = {+0.00000000f, +0.00000450f, +0.00116360f, +0.05087125f, +0.37589052f, +0.46943045f, +0.09908366f, +0.00353471f, +0.00002131f};
static devConstant float32 initialFilter3[] = {+0.00000000f, +0.00000000f, +0.00000085f, +0.00034278f, +0.02337268f, +0.26935022f, +0.52462259f, +0.17270208f, +0.00960880f};
static const Space initialFilterShift = -2;

#define FILTER0 initialFilter0
#define FILTER1 initialFilter1
#define FILTER2 initialFilter2
#define FILTER3 initialFilter3
#define FILTER_SRC_SHIFT initialFilterShift

////

#define FUNCSPACE gaussMaskResampling

#define PACK_SIZE 4
#define PACK_TO_SRC_FACTOR 5

#define HORIZONTAL_FIRST 1

////

#undef FUNCNAME
#define FUNCNAME downsampleOneAndQuarterGaussMaskInitial1

#undef TASK_COUNT
#define TASK_COUNT 1
# include "rationalResample/rationalResampleMultiple.inl"

////

#undef FUNCNAME
#define FUNCNAME downsampleOneAndQuarterGaussMaskInitial2

#undef TASK_COUNT
#define TASK_COUNT 2

# include "rationalResample/rationalResampleMultiple.inl"

////

#undef FUNCNAME
#define FUNCNAME downsampleOneAndQuarterGaussMaskInitial3

#undef TASK_COUNT
#define TASK_COUNT 3

# include "rationalResample/rationalResampleMultiple.inl"

////

#undef FUNCNAME
#define FUNCNAME downsampleOneAndQuarterGaussMaskInitial4

#undef TASK_COUNT
#define TASK_COUNT 4

# include "rationalResample/rationalResampleMultiple.inl"

//================================================================
//
// downsampleOneAndQuarterGaussMaskSustaining
//
//================================================================

static devConstant float32 sustainingFilter0[] = {+0.03796609f, +0.83140174f, +0.13048539f, +0.00014677f, +0.00000000f, +0.00000000f, +0.00000000f};
static devConstant float32 sustainingFilter1[] = {+0.00000081f, +0.00854574f, +0.64317744f, +0.34693474f, +0.00134122f, +0.00000004f, +0.00000000f};
static devConstant float32 sustainingFilter2[] = {+0.00000000f, +0.00000004f, +0.00134122f, +0.34693474f, +0.64317744f, +0.00854574f, +0.00000081f};
static devConstant float32 sustainingFilter3[] = {+0.00000000f, +0.00000000f, +0.00000000f, +0.00014677f, +0.13048539f, +0.83140174f, +0.03796609f};
static const Space sustainingSrcShift = -1;

//----------------------------------------------------------------

#undef FUNCNAME

#undef FILTER0
#undef FILTER1
#undef FILTER2
#undef FILTER3
#undef FILTER_SRC_SHIFT

#define FILTER0 sustainingFilter0
#define FILTER1 sustainingFilter1
#define FILTER2 sustainingFilter2
#define FILTER3 sustainingFilter3
#define FILTER_SRC_SHIFT sustainingSrcShift

////

#undef FUNCNAME
#define FUNCNAME downsampleOneAndQuarterGaussMaskSustaining1

#undef TASK_COUNT
#define TASK_COUNT 1

# include "rationalResample/rationalResampleMultiple.inl"

////

#undef FUNCNAME
#define FUNCNAME downsampleOneAndQuarterGaussMaskSustaining2

#undef TASK_COUNT
#define TASK_COUNT 2

# include "rationalResample/rationalResampleMultiple.inl"

////

#undef FUNCNAME
#define FUNCNAME downsampleOneAndQuarterGaussMaskSustaining3

#undef TASK_COUNT
#define TASK_COUNT 3

# include "rationalResample/rationalResampleMultiple.inl"

////

#undef FUNCNAME
#define FUNCNAME downsampleOneAndQuarterGaussMaskSustaining4

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
// downsampleOneAndQuarterGaussMaskMultitask
//
//================================================================

template <typename Src, typename Interm, typename Dst>
void downsampleOneAndQuarterGaussMaskMultitask(const GpuLayeredMatrix<const Src>& src, const GpuLayeredMatrix<Dst>& dst, BorderMode borderMode, bool initial, stdPars(GpuProcessKit))
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
                downsampleOneAndQuarterGaussMaskInitial##n<Src, Interm, Dst> : \
                downsampleOneAndQuarterGaussMaskSustaining##n<Src, Interm, Dst>; \
            \
            func(PREP_ENUM(n, TMP_PASS, _), borderMode, stdPass); \
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
}

//----------------------------------------------------------------

#define TMP_MACRO(Src, Interm, Dst, _) \
    INSTANTIATE_FUNC_EX((downsampleOneAndQuarterGaussMaskMultitask<Src, Interm, Dst>), Dst)

FOREACH_TYPE(TMP_MACRO)

#undef TMP_MACRO

//----------------------------------------------------------------

}

#endif
