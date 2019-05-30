#if HOSTCODE
#include "upsampleFourTimes.h"
#endif

#include "gpuDevice/gpuDevice.h"

#include "gaussSincResampling/gaussSincResamplingSettings.h"

//================================================================
//
// Instance
//
//================================================================

COMPILE_ASSERT(GAUSS_SINC_RESAMPLING_QUALITY == 0);

static devConstant float32 FILTER0[] = {+0.00387308f, -0.00900910f, +0.01936415f, -0.03907094f, +0.07635356f, -0.15618776f, +0.46015820f, +0.77765646f, -0.19243908f, +0.09045953f, -0.04624995f, +0.02319632f, -0.01098439f, +0.00482143f, -0.00194152f};
static devConstant float32 FILTER1[] = {+0.00128228f, -0.00304700f, +0.00667231f, -0.01364880f, +0.02676633f, -0.05343562f, +0.13340577f, +0.97357244f, -0.10091749f, +0.04460112f, -0.02265611f, +0.01147327f, -0.00552380f, +0.00247396f, -0.00101865f}; 
static devConstant float32 FILTER2[] = {-0.00101865f, +0.00247396f, -0.00552380f, +0.01147327f, -0.02265611f, +0.04460112f, -0.10091749f, +0.97357244f, +0.13340577f, -0.05343562f, +0.02676633f, -0.01364880f, +0.00667231f, -0.00304700f, +0.00128228f};
static devConstant float32 FILTER3[] = {-0.00194152f, +0.00482143f, -0.01098439f, +0.02319632f, -0.04624995f, +0.09045953f, -0.19243908f, +0.77765646f, +0.46015820f, -0.15618776f, +0.07635356f, -0.03907094f, +0.01936415f, -0.00900910f, +0.00387308f};
#define FILTER_SRC_SHIFT -7

//----------------------------------------------------------------

#define FUNCSPACE gaussSincResampling
#define FUNCNAME upsampleFourTimesBalanced

#define PACK_SIZE 4
#define PACK_TO_SRC_FACTOR 1

#define HORIZONTAL_FIRST 0

#define FOREACH_TYPE(action) \
    \
    action(int8, int8, int8, 1) \
    action(uint8, uint8, uint8, 1) \
    action(int16, int16, int16, 1) \
    action(uint16, uint16, uint16, 1) \
    action(float16, float16, float16, 1) \
    \
    action(int8_x2, int8_x2, int8_x2, 2) \
    action(uint8_x2, uint8_x2, uint8_x2, 2) \
    action(int16_x2, int16_x2, int16_x2, 2) \
    action(uint16_x2, uint16_x2, uint16_x2, 2) \
    action(float16_x2, float16_x2, float16_x2, 2)

# include "rationalResample/rationalResampleMultiple.inl"

////

#undef FUNCNAME
#define FUNCNAME upsampleFourTimesBalancedDual

#undef TASK_COUNT
#define TASK_COUNT 2

# include "rationalResample/rationalResampleMultiple.inl"
