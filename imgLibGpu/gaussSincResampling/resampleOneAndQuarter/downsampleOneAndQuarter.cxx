#include "downsampleOneAndQuarter.h"

#include "gaussSincResampling/gaussSincResamplingSettings.h"
#include "gaussSincResampling/common/allTypes.h"

using namespace gaussSincResampling;

//================================================================
//
// downsampleOneAndQuarterConservative
//
//================================================================

#if GAUSS_SINC_RESAMPLING_HQ == 0

    static const Space conservativeFilterSrcShift = -10;
    static devConstant float32 conservativeFilter0[] = {+0.00241186f, -0.00565753f, +0.00000000f, +0.01435584f, -0.01565120f, -0.01660933f, +0.05355094f, -0.01984877f, -0.11210227f, +0.22673374f, +0.60919981f, +0.35535279f, -0.07326762f, -0.06169844f, +0.05600034f, +0.00000000f, -0.02450656f, +0.01135782f, +0.00509503f, -0.00678908f, +0.00099145f, +0.00200081f, -0.00113536f, -0.00018091f, +0.00039665f};
    static devConstant float32 conservativeFilter1[] = {+0.00000000f, +0.00297905f, -0.00354263f, -0.00403734f, +0.01364123f, -0.00508064f, -0.02653388f, +0.04002093f, +0.01775756f, -0.11813595f, +0.10336638f, +0.56109975f, +0.47220693f, +0.00000000f, -0.09806011f, +0.04415521f, +0.02038911f, -0.02886426f, +0.00456581f, +0.01010837f, -0.00634871f, -0.00112687f, +0.00276548f, -0.00074093f, -0.00058448f};
    static devConstant float32 conservativeFilter2[] = {-0.00058448f, -0.00074093f, +0.00276548f, -0.00112687f, -0.00634871f, +0.01010837f, +0.00456581f, -0.02886426f, +0.02038911f, +0.04415521f, -0.09806011f, +0.00000000f, +0.47220693f, +0.56109975f, +0.10336638f, -0.11813595f, +0.01775756f, +0.04002093f, -0.02653388f, -0.00508064f, +0.01364123f, -0.00403734f, -0.00354263f, +0.00297905f, +0.00000000f};
    static devConstant float32 conservativeFilter3[] = {+0.00039665f, -0.00018091f, -0.00113536f, +0.00200081f, +0.00099145f, -0.00678908f, +0.00509503f, +0.01135782f, -0.02450656f, +0.00000000f, +0.05600034f, -0.06169844f, -0.07326762f, +0.35535279f, +0.60919981f, +0.22673374f, -0.11210227f, -0.01984877f, +0.05355094f, -0.01660933f, -0.01565120f, +0.01435584f, +0.00000000f, -0.00565753f, +0.00241186f};

#elif GAUSS_SINC_RESAMPLING_HQ == 1

    static const Space conservativeFilterSrcShift = -16;
    static devConstant float32 conservativeFilter0[] = {+0.00138257f, +0.00071226f, -0.00369157f, +0.00370542f, +0.00183756f, -0.00919019f, +0.00892903f, +0.00430319f, -0.02102515f, +0.02009971f, +0.00962787f, -0.04745908f, +0.04686397f, +0.02414972f, -0.13887300f, +0.19765679f, +0.65840175f, +0.34869457f, -0.11624576f, -0.02660204f, +0.06936987f, -0.03734485f, -0.01030883f, +0.02935433f, -0.01645488f, -0.00460497f, +0.01307443f, -0.00722718f, -0.00197903f, +0.00546690f, -0.00292773f, -0.00077416f, +0.00205975f, -0.00106023f, -0.00026900f, +0.00068580f, -0.00033785f};
    static devConstant float32 conservativeFilter1[] = {-0.00120946f, +0.00046417f, +0.00179101f, -0.00340888f, +0.00125633f, +0.00466387f, -0.00856019f, +0.00305114f, +0.01099540f, -0.01968695f, +0.00689057f, +0.02460968f, -0.04426150f, +0.01588978f, +0.06028773f, -0.12316723f, +0.05893659f, +0.59965426f, +0.49039480f, -0.04953807f, -0.08067616f, +0.07549146f, -0.01473324f, -0.03021393f, +0.03142771f, -0.00644615f, -0.01347531f, +0.01402503f, -0.00284351f, -0.00582621f, +0.00590734f, -0.00116146f, -0.00229973f, +0.00224716f, -0.00042486f, -0.00080751f, +0.00075631f};
    static devConstant float32 conservativeFilter2[] = {+0.00075631f, -0.00080751f, -0.00042486f, +0.00224716f, -0.00229973f, -0.00116146f, +0.00590734f, -0.00582621f, -0.00284351f, +0.01402503f, -0.01347531f, -0.00644615f, +0.03142771f, -0.03021393f, -0.01473324f, +0.07549146f, -0.08067616f, -0.04953807f, +0.49039480f, +0.59965426f, +0.05893659f, -0.12316723f, +0.06028773f, +0.01588978f, -0.04426150f, +0.02460968f, +0.00689057f, -0.01968695f, +0.01099540f, +0.00305114f, -0.00856019f, +0.00466387f, +0.00125633f, -0.00340888f, +0.00179101f, +0.00046417f, -0.00120946f};
    static devConstant float32 conservativeFilter3[] = {-0.00033785f, +0.00068580f, -0.00026900f, -0.00106023f, +0.00205975f, -0.00077416f, -0.00292773f, +0.00546690f, -0.00197903f, -0.00722718f, +0.01307443f, -0.00460497f, -0.01645488f, +0.02935433f, -0.01030883f, -0.03734485f, +0.06936987f, -0.02660204f, -0.11624576f, +0.34869457f, +0.65840175f, +0.19765679f, -0.13887300f, +0.02414972f, +0.04686397f, -0.04745908f, +0.00962787f, +0.02009971f, -0.02102515f, +0.00430319f, +0.00892903f, -0.00919019f, +0.00183756f, +0.00370542f, -0.00369157f, +0.00071226f, +0.00138257f};

#else

    #error

#endif

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
