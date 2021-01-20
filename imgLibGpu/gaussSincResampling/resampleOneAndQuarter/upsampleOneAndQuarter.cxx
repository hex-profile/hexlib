#if HOSTCODE
#include "upsampleOneAndQuarter.h"
#endif

#include "gaussSincResampling/gaussSincResamplingSettings.h"
#include "gaussSincResampling/common/allTypes.h"

//================================================================
//
// upsampleOneAndQuarterBalanced
//
//================================================================

#define FUNCSPACE gaussSincResampling

#define PACK_SIZE 5
#define PACK_TO_SRC_FACTOR 4

#define HORIZONTAL_FIRST 0

//----------------------------------------------------------------

COMPILE_ASSERT(GAUSS_SINC_RESAMPLING_QUALITY == 0);
static devConstant float32 balancedFilter0[] = {-0.00241229f, +0.00529252f, -0.01084216f, +0.02127401f, -0.04239411f, +0.10456193f, +0.98382546f, -0.08367051f, +0.03668917f, -0.01861798f, +0.00943610f, -0.00455023f, +0.00204196f, -0.00084263f, +0.00031745f, -0.00010867f};
static devConstant float32 balancedFilter1[] = {+0.00317089f, -0.00742270f, +0.01604222f, -0.03249610f, +0.06354554f, -0.12887265f, +0.35761652f, +0.85318919f, -0.18014379f, +0.08336349f, -0.04256729f, +0.02141671f, -0.01019346f, +0.00450172f, -0.00182500f, +0.00067471f};
static devConstant float32 balancedFilter2[] = {-0.00186663f, +0.00468803f, -0.01079123f, +0.02298772f, -0.04609567f, +0.09006436f, -0.18746129f, +0.62847471f, +0.62847471f, -0.18746129f, +0.09006436f, -0.04609567f, +0.02298772f, -0.01079123f, +0.00468803f, -0.00186663f};
static devConstant float32 balancedFilter3[] = {+0.00067471f, -0.00182500f, +0.00450172f, -0.01019346f, +0.02141671f, -0.04256729f, +0.08336349f, -0.18014379f, +0.85318919f, +0.35761652f, -0.12887265f, +0.06354554f, -0.03249610f, +0.01604222f, -0.00742270f, +0.00317089f};
static devConstant float32 balancedFilter4[] = {-0.00010867f, +0.00031745f, -0.00084263f, +0.00204196f, -0.00455023f, +0.00943610f, -0.01861798f, +0.03668917f, -0.08367051f, +0.98382546f, +0.10456193f, -0.04239411f, +0.02127401f, -0.01084216f, +0.00529252f, -0.00241229f};
static const Space balancedFilterSrcShift = -6;

#define FILTER0 balancedFilter0
#define FILTER1 balancedFilter1
#define FILTER2 balancedFilter2
#define FILTER3 balancedFilter3
#define FILTER4 balancedFilter4
#define FILTER_SRC_SHIFT balancedFilterSrcShift

#define FUNCNAME upsampleOneAndQuarterBalanced

# include "rationalResample/rationalResampleMultiple.inl"
