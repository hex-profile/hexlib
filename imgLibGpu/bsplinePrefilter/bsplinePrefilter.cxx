#if HOSTCODE
#include "bsplinePrefilter.h"
#endif

#include "gpuDevice/gpuDevice.h"

//================================================================
//
// bsplineCubicPrefilter
//
// Michael Unser prefilter for cubic bspline kernel.
//
//================================================================

#define FILTER_CORE15 -0.00017177f, +0.00064107f, -0.00239251f, +0.00892898f, -0.03332342f, +0.12436468f, -0.46413531f, +1.73217655f, -0.46413531f, +0.12436468f, -0.03332342f, +0.00892898f, -0.00239251f, +0.00064107f, -0.00017177f
#define FILTER_SRC_SHIFT (-7)

static devConstant float32 preFilter0[] = {FILTER_CORE15, 0, 0, 0};
static devConstant float32 preFilter1[] = {0, FILTER_CORE15, 0, 0};
static devConstant float32 preFilter2[] = {0, 0, FILTER_CORE15, 0};
static devConstant float32 preFilter3[] = {0, 0, 0, FILTER_CORE15};

#define FILTER0 preFilter0
#define FILTER1 preFilter1
#define FILTER2 preFilter2
#define FILTER3 preFilter3

#define FUNCNAME bsplineCubicPrefilter

#define PACK_SIZE 4
#define PACK_TO_SRC_FACTOR 4

#define HORIZONTAL_FIRST 1

#define OUTPUT_FACTOR

#define FOREACH_TYPE(action) \
    \
    action(float16, float16, float16, 1) \
    action(float16_x2, float16_x2, float16_x2, 2) \
    action(float16_x4, float16_x4, float16_x4, 4) \
    \
    action(int8, float16, float16, 1) \
    action(int8_x2, float16_x2, float16_x2, 2) \
    action(int8_x4, float16_x4, float16_x4, 4) \
    \
    action(uint8, float16, float16, 1) \
    action(uint8_x2, float16_x2, float16_x2, 2) \
    action(uint8_x4, float16_x4, float16_x4, 4) \
    \
    action(float32, float32, float32, 1)

#define FOREACH_DST_TYPE(action) \
    \
    action(float16, float16, float16, 1) \
    action(float16_x2, float16_x2, float16_x2, 2) \
    action(float16_x4, float16_x4, float16_x4, 4) \
    \
    action(float32, float32, float32, 1)

# include "rationalResample/rationalResampleMultiple.inl"

#undef FOREACH_DST_TYPE
#undef FOREACH_TYPE

#undef FILTER0
#undef FILTER1
#undef FILTER2
#undef FILTER3

#undef FILTER_SRC_SHIFT
#undef FUNCNAME
#undef PACK_SIZE
#undef PACK_TO_SRC_FACTOR

//================================================================
//
// bsplineCubicUnprefilter
//
//================================================================

#define FILTER_CORE3 (1.f/6), (4.f/6), (1.f/6)
#define FILTER_SRC_SHIFT (-1)

static devConstant float32 FILTER0[] = {FILTER_CORE3, 0, 0, 0};
static devConstant float32 FILTER1[] = {0, FILTER_CORE3, 0, 0};
static devConstant float32 FILTER2[] = {0, 0, FILTER_CORE3, 0};
static devConstant float32 FILTER3[] = {0, 0, 0, FILTER_CORE3};

#define FUNCNAME bsplineCubicUnprefilter

#define PACK_SIZE 4
#define PACK_TO_SRC_FACTOR 4

#define FOREACH_TYPE(action) \
    \
    action(float16, float16, float16, 1) \
    action(float16_x2, float16_x2, float16_x2, 2) \
    action(float16_x4, float16_x4, float16_x4, 4) \
    \
    action(float16, float16, int8, 1) \
    action(float16_x2, float16_x2, int8_x2, 2) \
    action(float16_x4, float16_x4, int8_x4, 4) \
    \
    action(float16, float16, uint8, 1) \
    action(float16_x2, float16_x2, uint8_x2, 2) \
    action(float16_x4, float16_x4, uint8_x4, 4)

# include "rationalResample/rationalResampleMultiple.inl"
