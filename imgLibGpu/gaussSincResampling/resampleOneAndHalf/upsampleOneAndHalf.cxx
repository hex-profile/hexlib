#if HOSTCODE
#include "upsampleOneAndHalf.h"
#endif

#include "gaussSincResampling/gaussSincResamplingSettings.h"
#include "gaussSincResampling/common/allTypes.h"

//================================================================
//
// upsampleOneAndHalfBalanced
//
//================================================================

#define FUNCSPACE gaussSincResampling

#define PACK_SIZE 3
#define PACK_TO_SRC_FACTOR 2

#define HORIZONTAL_FIRST 0

//----------------------------------------------------------------

COMPILE_ASSERT(GAUSS_SINC_RESAMPLING_QUALITY == 0);
static devConstant float32 balancedFilter0[] = {+0.00173900f, -0.00411747f, +0.00898795f, -0.01834153f, +0.03593885f, -0.07198057f, +0.18364449f, +0.95286833f, -0.12640519f, +0.05655808f, -0.02877429f, +0.01455068f, -0.00698670f, +0.00311880f, -0.00127944f, +0.00047900f};
static devConstant float32 balancedFilter1[] = {-0.00186663f, +0.00468803f, -0.01079123f, +0.02298772f, -0.04609567f, +0.09006436f, -0.18746129f, +0.62847471f, +0.62847471f, -0.18746129f, +0.09006436f, -0.04609567f, +0.02298772f, -0.01079123f, +0.00468803f, -0.00186663f};
static devConstant float32 balancedFilter2[] = {+0.00047900f, -0.00127944f, +0.00311880f, -0.00698670f, +0.01455068f, -0.02877429f, +0.05655808f, -0.12640519f, +0.95286833f, +0.18364449f, -0.07198057f, +0.03593885f, -0.01834153f, +0.00898795f, -0.00411747f, +0.00173900f};
static const Space balancedFilterSrcShift = -7;

#define FILTER0 balancedFilter0
#define FILTER1 balancedFilter1
#define FILTER2 balancedFilter2
#define FILTER_SRC_SHIFT balancedFilterSrcShift

#define FUNCNAME upsampleOneAndHalfBalanced

# include "rationalResample/rationalResampleMultiple.inl"
