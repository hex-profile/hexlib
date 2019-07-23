#if HOSTCODE
#include "upsampleOneAndHalf.h"
#endif

#include "gaussSincResampling/gaussSincResamplingSettings.h"

//================================================================
//
// upsampleOneAndHalfConservative
//
//================================================================

COMPILE_ASSERT(GAUSS_SINC_RESAMPLING_QUALITY == 0);
static devConstant float32 conservativeFilter0[] = {+0.00166414f, +0.00043475f, -0.00723411f, +0.01780444f, -0.02365456f, +0.00821240f, +0.04609053f, -0.14923741f, +0.33710882f, +0.74748520f, +0.08257036f, -0.10897243f, +0.07078922f, -0.02592033f, -0.00206014f, +0.01065854f, -0.00819722f, +0.00336871f, -0.00035155f, -0.00055936f};
static devConstant float32 conservativeFilter1[] = {-0.00141998f, +0.00345938f, -0.00443153f, -0.00000000f, +0.01420508f, -0.03610672f, +0.05006276f, -0.02482460f, -0.09163485f, +0.59069046f, +0.59069046f, -0.09163485f, -0.02482460f, +0.05006276f, -0.03610672f, +0.01420508f, +0.00000000f, -0.00443153f, +0.00345938f, -0.00141998f};
static devConstant float32 conservativeFilter2[] = {-0.00055936f, -0.00035155f, +0.00336871f, -0.00819722f, +0.01065854f, -0.00206014f, -0.02592033f, +0.07078922f, -0.10897243f, +0.08257036f, +0.74748520f, +0.33710882f, -0.14923741f, +0.04609053f, +0.00821240f, -0.02365456f, +0.01780444f, -0.00723411f, +0.00043475f, +0.00166414f};
static const Space conservativeFilterSrcShift = -9;

//----------------------------------------------------------------

#define FUNCSPACE gaussSincResampling
#define FUNCNAME upsampleOneAndHalfConservative

#define PACK_SIZE 3
#define PACK_TO_SRC_FACTOR 2

#define FILTER0 conservativeFilter0
#define FILTER1 conservativeFilter1
#define FILTER2 conservativeFilter2
#define FILTER_SRC_SHIFT conservativeFilterSrcShift

#define HORIZONTAL_FIRST 0

#define FOREACH_TYPE(action) \
    \
    action(int8, int8, int8, 1) \
    action(uint8, uint8, uint8, 1) \
    action(float16, float16, float16, 1) \
    \
    action(int8_x2, int8_x2, int8_x2, 2) \
    action(uint8_x2, uint8_x2, uint8_x2, 2) \
    action(float16_x2, float16_x2, float16_x2, 2)

# include "rationalResample/rationalResampleMultiple.inl"

//================================================================
//
// upsampleOneAndHalfBalanced
//
//================================================================

COMPILE_ASSERT(GAUSS_SINC_RESAMPLING_QUALITY == 0);
static devConstant float32 balancedFilter0[] = {+0.00173900f, -0.00411747f, +0.00898795f, -0.01834153f, +0.03593885f, -0.07198057f, +0.18364449f, +0.95286833f, -0.12640519f, +0.05655808f, -0.02877429f, +0.01455068f, -0.00698670f, +0.00311880f, -0.00127944f, +0.00047900f};
static devConstant float32 balancedFilter1[] = {-0.00186663f, +0.00468803f, -0.01079123f, +0.02298772f, -0.04609567f, +0.09006436f, -0.18746129f, +0.62847471f, +0.62847471f, -0.18746129f, +0.09006436f, -0.04609567f, +0.02298772f, -0.01079123f, +0.00468803f, -0.00186663f};
static devConstant float32 balancedFilter2[] = {+0.00047900f, -0.00127944f, +0.00311880f, -0.00698670f, +0.01455068f, -0.02877429f, +0.05655808f, -0.12640519f, +0.95286833f, +0.18364449f, -0.07198057f, +0.03593885f, -0.01834153f, +0.00898795f, -0.00411747f, +0.00173900f};
static const Space balancedFilterSrcShift = -7;

#undef FUNCNAME
#define FUNCNAME upsampleOneAndHalfBalanced

#undef FILTER0
#undef FILTER1
#undef FILTER2
#undef FILTER_SRC_SHIFT

#define FILTER0 balancedFilter0
#define FILTER1 balancedFilter1
#define FILTER2 balancedFilter2
#define FILTER_SRC_SHIFT balancedFilterSrcShift

# include "rationalResample/rationalResampleMultiple.inl"
