#include "downsampleOne.h"

#include "gaussSincResampling/gaussSincResamplingSettings.h"
#include "gaussSincResampling/common/allTypes.h"

using namespace gaussSincResampling;

//================================================================
//
// downsampleOneConservative
//
//================================================================

#if GAUSS_SINC_RESAMPLING_HQ == 0

    static const Space conservativeFilterSrcShift = -8;
    static devConstant float32 conservativeFilter0[] = {+0.00225560f, -0.00849235f, +0.01519044f, -0.01300677f, -0.01125501f, +0.06495925f, -0.13853066f, +0.20425735f, +0.76924904f, +0.20425735f, -0.13853066f, +0.06495925f, -0.01125501f, -0.01300677f, +0.01519044f, -0.00849235f, +0.00225560f, +0.00059045f, -0.00097857f, +0.00053805f, -0.00015468f};
    static devConstant float32 conservativeFilter1[] = {+0.00059001f, +0.00225392f, -0.00848602f, +0.01517913f, -0.01299709f, -0.01124663f, +0.06491088f, -0.13842752f, +0.20410527f, +0.76867627f, +0.20410527f, -0.13842752f, +0.06491088f, -0.01124663f, -0.01299709f, +0.01517913f, -0.00848602f, +0.00225392f, +0.00059001f, -0.00097784f, +0.00053765f};
    static devConstant float32 conservativeFilter2[] = {-0.00097932f, +0.00059091f, +0.00225734f, -0.00849890f, +0.01520217f, -0.01301681f, -0.01126370f, +0.06500941f, -0.13863762f, +0.20441506f, +0.76984296f, +0.20441506f, -0.13863762f, +0.06500941f, -0.01126370f, -0.01301681f, +0.01520217f, -0.00849890f, +0.00225734f, +0.00059091f, -0.00097932f};
    static devConstant float32 conservativeFilter3[] = {+0.00053765f, -0.00097784f, +0.00059001f, +0.00225392f, -0.00848602f, +0.01517913f, -0.01299709f, -0.01124663f, +0.06491088f, -0.13842752f, +0.20410527f, +0.76867627f, +0.20410527f, -0.13842752f, +0.06491088f, -0.01124663f, -0.01299709f, +0.01517913f, -0.00848602f, +0.00225392f, +0.00059001f};

#elif GAUSS_SINC_RESAMPLING_HQ == 1

    static const Space conservativeFilterSrcShift = -13;
    static devConstant float32 conservativeFilter0[] = {+0.00117016f, -0.00000000f, -0.00269356f, +0.00686988f, -0.01147575f, +0.01415814f, -0.01150576f, +0.00000000f, +0.02248062f, -0.05515257f, +0.09358300f, -0.13031013f, +0.15687118f, +0.83286314f, +0.15687118f, -0.13031013f, +0.09358300f, -0.05515257f, +0.02248062f, +0.00000000f, -0.01150576f, +0.01415814f, -0.01147575f, +0.00686988f, -0.00269356f, -0.00000000f, +0.00117016f, -0.00129349f, +0.00093184f, -0.00049188f};
    static devConstant float32 conservativeFilter1[] = {-0.00129452f, +0.00117110f, -0.00000000f, -0.00269572f, +0.00687539f, -0.01148496f, +0.01416950f, -0.01151499f, +0.00000000f, +0.02249866f, -0.05519682f, +0.09365807f, -0.13041467f, +0.15699703f, +0.83353130f, +0.15699703f, -0.13041467f, +0.09365807f, -0.05519682f, +0.02249866f, +0.00000000f, -0.01151499f, +0.01416950f, -0.01148496f, +0.00687539f, -0.00269572f, -0.00000000f, +0.00117110f, -0.00129452f, +0.00093259f};
    static devConstant float32 conservativeFilter2[] = {+0.00093259f, -0.00129452f, +0.00117110f, -0.00000000f, -0.00269572f, +0.00687539f, -0.01148496f, +0.01416950f, -0.01151499f, +0.00000000f, +0.02249866f, -0.05519682f, +0.09365807f, -0.13041467f, +0.15699703f, +0.83353130f, +0.15699703f, -0.13041467f, +0.09365807f, -0.05519682f, +0.02249866f, +0.00000000f, -0.01151499f, +0.01416950f, -0.01148496f, +0.00687539f, -0.00269572f, -0.00000000f, +0.00117110f, -0.00129452f};
    static devConstant float32 conservativeFilter3[] = {-0.00049188f, +0.00093184f, -0.00129349f, +0.00117016f, -0.00000000f, -0.00269356f, +0.00686988f, -0.01147575f, +0.01415814f, -0.01150576f, +0.00000000f, +0.02248062f, -0.05515257f, +0.09358300f, -0.13031013f, +0.15687118f, +0.83286314f, +0.15687118f, -0.13031013f, +0.09358300f, -0.05515257f, +0.02248062f, +0.00000000f, -0.01150576f, +0.01415814f, -0.01147575f, +0.00686988f, -0.00269356f, -0.00000000f, +0.00117016f};

#else

    #error

#endif

//----------------------------------------------------------------

#define FUNCSPACE gaussSincResampling

#define PACK_SIZE 4
#define PACK_TO_SRC_FACTOR 4

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
#define FUNCNAME downsampleOneConservative1

# include "rationalResample/rationalResampleMultiple.inl"

////

#undef TASK_COUNT
#define TASK_COUNT 2

#undef FUNCNAME
#define FUNCNAME downsampleOneConservative2

# include "rationalResample/rationalResampleMultiple.inl"

////

#undef TASK_COUNT
#define TASK_COUNT 3

#undef FUNCNAME
#define FUNCNAME downsampleOneConservative3

# include "rationalResample/rationalResampleMultiple.inl"

////

#undef TASK_COUNT
#define TASK_COUNT 4

#undef FUNCNAME
#define FUNCNAME downsampleOneConservative4

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
// downsampleOneConservativeMultitask
//
//================================================================

template <typename Src, typename Interm, typename Dst>
void downsampleOneConservativeMultitask(const GpuLayeredMatrix<const Src>& src, const GpuLayeredMatrix<Dst>& dst, BorderMode borderMode, stdPars(GpuProcessKit))
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
            auto func = downsampleOneConservative##n<Src, Interm, Dst>; \
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
    INSTANTIATE_FUNC_EX((downsampleOneConservativeMultitask<Src, Interm, Dst>), Dst)

FOREACH_TYPE(TMP_MACRO)

#undef TMP_MACRO

//----------------------------------------------------------------

}

#endif
