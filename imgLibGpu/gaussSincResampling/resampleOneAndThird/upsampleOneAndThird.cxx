#if HOSTCODE
#include "upsampleOneAndThird.h"
#endif

#include "gaussSincResampling/gaussSincResamplingSettings.h"
#include "gaussSincResampling/common/allTypes.h"

//================================================================
//
// upsampleOneAndThirdBalanced
//
//================================================================

#define FUNCSPACE gaussSincResampling

#define PACK_SIZE 4
#define PACK_TO_SRC_FACTOR 3

#define HORIZONTAL_FIRST 0

//----------------------------------------------------------------

COMPILE_ASSERT(GAUSS_SINC_RESAMPLING_QUALITY == 0);
static devConstant float32 balancedFilter0[] = {-0.00304974f, +0.00667832f, -0.01366109f, +0.02679042f, -0.05348373f, +0.13352587f, +0.97444888f, -0.10100833f, +0.04464127f, -0.02267651f, +0.01148360f, -0.00552878f, +0.00247618f, -0.00101957f, +0.00038321f};
static devConstant float32 balancedFilter1[] = {+0.00387308f, -0.00900910f, +0.01936415f, -0.03907094f, +0.07635356f, -0.15618776f, +0.46015820f, +0.77765646f, -0.19243908f, +0.09045953f, -0.04624995f, +0.02319632f, -0.01098439f, +0.00482143f, -0.00194152f};
static devConstant float32 balancedFilter2[] = {-0.00194152f, +0.00482143f, -0.01098439f, +0.02319632f, -0.04624995f, +0.09045953f, -0.19243908f, +0.77765646f, +0.46015820f, -0.15618776f, +0.07635356f, -0.03907094f, +0.01936415f, -0.00900910f, +0.00387308f};
static devConstant float32 balancedFilter3[] = {+0.00038321f, -0.00101957f, +0.00247618f, -0.00552878f, +0.01148360f, -0.02267651f, +0.04464127f, -0.10100833f, +0.97444888f, +0.13352587f, -0.05348373f, +0.02679042f, -0.01366109f, +0.00667832f, -0.00304974f};
static const Space balancedFilterSrcShift = -6;

#define FILTER0 balancedFilter0
#define FILTER1 balancedFilter1
#define FILTER2 balancedFilter2
#define FILTER3 balancedFilter3
#define FILTER_SRC_SHIFT balancedFilterSrcShift

#define FUNCNAME upsampleOneAndThirdBalanced

# include "rationalResample/rationalResampleMultiple.inl"
