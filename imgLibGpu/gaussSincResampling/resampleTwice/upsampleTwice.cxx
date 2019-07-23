#if HOSTCODE
#include "upsampleTwice.h"
#endif

#include "gaussSincResampling/gaussSincResamplingSettings.h"

//================================================================
//
// Instance
//
//================================================================


COMPILE_ASSERT(GAUSS_SINC_RESAMPLING_QUALITY == 0);
static devConstant float32 FILTER0[] = {+0.00265213f, -0.00623483f, +0.01352500f, -0.02747154f, +0.05375393f, -0.10846222f, +0.29078630f, +0.89693080f, -0.16504324f, +0.07548820f, -0.03850145f, +0.01940955f, -0.00926903f, +0.00411005f, -0.00167367f};
static devConstant float32 FILTER1[] = {-0.00167367f, +0.00411005f, -0.00926903f, +0.01940955f, -0.03850145f, +0.07548820f, -0.16504324f, +0.89693080f, +0.29078630f, -0.10846222f, +0.05375393f, -0.02747154f, +0.01352500f, -0.00623483f, +0.00265213f};
static const Space FILTER_SRC_SHIFT = -7;

//----------------------------------------------------------------

#define FUNCSPACE gaussSincResampling
#define FUNCNAME upsampleTwiceBalanced

#define PACK_SIZE 2
#define PACK_TO_SRC_FACTOR 1

#define HORIZONTAL_FIRST 1

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
