#include "downsampleOneAndThird.h"

#include "gaussSincResampling/gaussSincResamplingSettings.h"
#include "gaussSincResampling/common/allTypes.h"

using namespace gaussSincResampling;

//================================================================
//
// downsampleOneAndThirdConservative
//
//================================================================

#if GAUSS_SINC_RESAMPLING_HQ == 0

    static const Space conservativeFilterSrcShift = -11;
    static devConstant float32 conservativeFilter0[] = {+0.00279382f, -0.00189957f, -0.00577745f, +0.00893510f, +0.00704868f, -0.02512208f, +0.00226968f, +0.05288493f, -0.04318244f, -0.09520463f, +0.22758823f, +0.56765431f, +0.37624931f, -0.02939805f, -0.08827668f, +0.03900945f, +0.02635732f, -0.02651133f, -0.00353944f, +0.01303643f, -0.00253176f, -0.00463247f, +0.00237087f, +0.00104430f, -0.00112618f, -0.00004035f};
    static devConstant float32 conservativeFilter1[] = {-0.00087323f, +0.00218743f, +0.00071130f, -0.00630163f, +0.00292815f, +0.01283250f, -0.01584928f, -0.01710868f, +0.04642745f, +0.00437693f, -0.11153895f, +0.08352015f, +0.49868786f, +0.49868786f, +0.08352015f, -0.11153895f, +0.00437693f, +0.04642745f, -0.01710868f, -0.01584928f, +0.01283250f, +0.00292815f, -0.00630163f, +0.00071130f, +0.00218743f, -0.00087323f};
    static devConstant float32 conservativeFilter2[] = {-0.00004035f, -0.00112618f, +0.00104430f, +0.00237087f, -0.00463247f, -0.00253176f, +0.01303643f, -0.00353944f, -0.02651133f, +0.02635732f, +0.03900945f, -0.08827668f, -0.02939805f, +0.37624931f, +0.56765431f, +0.22758823f, -0.09520463f, -0.04318244f, +0.05288493f, +0.00226968f, -0.02512208f, +0.00704868f, +0.00893510f, -0.00577745f, -0.00189957f, +0.00279382f};

#elif GAUSS_SINC_RESAMPLING_HQ == 1

    static const Space conservativeFilterSrcShift = -17;
    static devConstant float32 conservativeFilter0[] = {+0.00139293f, +0.00082048f, -0.00346799f, +0.00206949f, +0.00410963f, -0.00778512f, +0.00070301f, +0.01250898f, -0.01352633f, -0.00743244f, +0.02963833f, -0.01694323f, -0.03293797f, +0.06309435f, -0.00607210f, -0.12687529f, +0.20273400f, +0.61308744f, +0.37866958f, -0.07471824f, -0.06949349f, +0.07002456f, -0.00358472f, -0.03747370f, +0.02429004f, +0.00807840f, -0.01952398f, +0.00671884f, +0.00773606f, -0.00852066f, +0.00044754f, +0.00462271f, -0.00288878f, -0.00091055f, +0.00205988f, -0.00065735f, -0.00069679f, +0.00070250f};
    static devConstant float32 conservativeFilter1[] = {-0.00116539f, +0.00032465f, +0.00191397f, -0.00261614f, -0.00082947f, +0.00557412f, -0.00417813f, -0.00547802f, +0.01257800f, -0.00323294f, -0.01772498f, +0.02276304f, +0.00687493f, -0.04486135f, +0.03358280f, +0.04596261f, -0.11904887f, +0.04071998f, +0.52884120f, +0.52884120f, +0.04071998f, -0.11904887f, +0.04596261f, +0.03358280f, -0.04486135f, +0.00687493f, +0.02276304f, -0.01772498f, -0.00323294f, +0.01257800f, -0.00547802f, -0.00417813f, +0.00557412f, -0.00082947f, -0.00261614f, +0.00191397f, +0.00032465f, -0.00116539f};
    static devConstant float32 conservativeFilter2[] = {+0.00070250f, -0.00069679f, -0.00065735f, +0.00205988f, -0.00091055f, -0.00288878f, +0.00462271f, +0.00044754f, -0.00852066f, +0.00773606f, +0.00671884f, -0.01952398f, +0.00807840f, +0.02429004f, -0.03747370f, -0.00358472f, +0.07002456f, -0.06949349f, -0.07471824f, +0.37866958f, +0.61308744f, +0.20273400f, -0.12687529f, -0.00607210f, +0.06309435f, -0.03293797f, -0.01694323f, +0.02963833f, -0.00743244f, -0.01352633f, +0.01250898f, +0.00070301f, -0.00778512f, +0.00410963f, +0.00206949f, -0.00346799f, +0.00082048f, +0.00139293f};

#else

    #error

#endif

//----------------------------------------------------------------

#define FUNCSPACE gaussSincResampling

#define PACK_SIZE 3
#define PACK_TO_SRC_FACTOR 4

#define FILTER0 conservativeFilter0
#define FILTER1 conservativeFilter1
#define FILTER2 conservativeFilter2
#define FILTER_SRC_SHIFT conservativeFilterSrcShift

#define HORIZONTAL_FIRST 1

//----------------------------------------------------------------

#undef TASK_COUNT
#define TASK_COUNT 1

#undef FUNCNAME
#define FUNCNAME downsampleOneAndThirdConservative1

# include "rationalResample/rationalResampleMultiple.inl"

////

#undef TASK_COUNT
#define TASK_COUNT 2

#undef FUNCNAME
#define FUNCNAME downsampleOneAndThirdConservative2

# include "rationalResample/rationalResampleMultiple.inl"

////

#undef TASK_COUNT
#define TASK_COUNT 3

#undef FUNCNAME
#define FUNCNAME downsampleOneAndThirdConservative3

# include "rationalResample/rationalResampleMultiple.inl"

////

#undef TASK_COUNT
#define TASK_COUNT 4

#undef FUNCNAME
#define FUNCNAME downsampleOneAndThirdConservative4

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
// downsampleOneAndThirdConservativeMultitask
//
//================================================================

template <typename Src, typename Interm, typename Dst>
void downsampleOneAndThirdConservativeMultitask(const GpuLayeredMatrix<const Src>& src, const GpuLayeredMatrix<Dst>& dst, BorderMode borderMode, stdPars(GpuProcessKit))
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
            auto func = downsampleOneAndThirdConservative##n<Src, Interm, Dst>; \
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
    INSTANTIATE_FUNC_EX((downsampleOneAndThirdConservativeMultitask<Src, Interm, Dst>), Dst)

FOREACH_TYPE(TMP_MACRO)

#undef TMP_MACRO

//----------------------------------------------------------------

}

#endif
