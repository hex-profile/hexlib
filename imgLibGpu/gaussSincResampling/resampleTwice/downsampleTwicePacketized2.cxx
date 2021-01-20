#if HOSTCODE
#include "downsampleTwicePacketized2.h"
#endif

#include "gaussSincResampling/gaussSincResamplingSettings.h"
#include "gaussSincResampling/common/allTypes.h"

//================================================================
//
// Downsample 2X (packet 2X)
//
//================================================================

COMPILE_ASSERT(GAUSS_SINC_RESAMPLING_QUALITY == 0);
static devConstant float32 FILTER0[] = {+0.00109807f, +0.00182181f, -0.00034342f, -0.00378384f, -0.00299301f, +0.00400249f, +0.00905726f, +0.00147555f, -0.01412048f, -0.01546785f, +0.00947198f, +0.03457897f, +0.01599327f, -0.04479528f, -0.07276104f, +0.01456780f, +0.20211261f, +0.36062647f, +0.36062647f, +0.20211261f, +0.01456780f, -0.07276104f, -0.04479528f, +0.01599327f, +0.03457897f, +0.00947198f, -0.01546785f, -0.01412048f, +0.00147555f, +0.00905726f, +0.00400249f, -0.00299301f, -0.00378384f, -0.00034342f, +0.00182181f, +0.00109807f, -0.00036606f, -0.00071667f};
static devConstant float32 FILTER1[] = {-0.00071667f, -0.00036606f, +0.00109807f, +0.00182181f, -0.00034342f, -0.00378384f, -0.00299301f, +0.00400249f, +0.00905726f, +0.00147555f, -0.01412048f, -0.01546785f, +0.00947198f, +0.03457897f, +0.01599327f, -0.04479528f, -0.07276104f, +0.01456780f, +0.20211261f, +0.36062647f, +0.36062647f, +0.20211261f, +0.01456780f, -0.07276104f, -0.04479528f, +0.01599327f, +0.03457897f, +0.00947198f, -0.01546785f, -0.01412048f, +0.00147555f, +0.00905726f, +0.00400249f, -0.00299301f, -0.00378384f, -0.00034342f, +0.00182181f, +0.00109807f};
static const Space FILTER_SRC_SHIFT = -17;

//----------------------------------------------------------------

#define FUNCSPACE gaussSincResampling
#define FUNCNAME downsampleTwicePacketized2

#define PACK_SIZE 2
#define PACK_TO_SRC_FACTOR 4

#define HORIZONTAL_FIRST 1

#define TASK_COUNT 1

# include "rationalResample/rationalResampleMultiple.inl"
