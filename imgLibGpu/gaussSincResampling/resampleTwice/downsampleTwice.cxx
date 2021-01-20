#if HOSTCODE
#include "downsampleTwice.h"
#endif

#include "gaussSincResampling/gaussSincResamplingSettings.h"
#include "gaussSincResampling/common/allTypes.h"

//================================================================
//
// Instance
//
//================================================================

COMPILE_ASSERT(GAUSS_SINC_RESAMPLING_QUALITY == 0);
static devConstant float32 FILTER0[] = {-0.00036593f, +0.00109768f, +0.00182117f, -0.00034330f, -0.00378252f, -0.00299196f, +0.00400108f, +0.00905409f, +0.00147503f, -0.01411553f, -0.01546243f, +0.00946866f, +0.03456685f, +0.01598766f, -0.04477958f, -0.07273554f, +0.01456270f, +0.20204177f, +0.36050008f, +0.36050008f, +0.20204177f, +0.01456270f, -0.07273554f, -0.04477958f, +0.01598766f, +0.03456685f, +0.00946866f, -0.01546243f, -0.01411553f, +0.00147503f, +0.00905409f, +0.00400108f, -0.00299196f, -0.00378252f, -0.00034330f, +0.00182117f, +0.00109768f, -0.00036593f};
static const Space FILTER_SRC_SHIFT = -18;

//----------------------------------------------------------------

#define FUNCSPACE gaussSincResampling
#define FUNCNAME downsampleTwiceConservative

#define PACK_SIZE 1
#define PACK_TO_SRC_FACTOR 2

#define HORIZONTAL_FIRST 1

# include "rationalResample/rationalResampleMultiple.inl"
