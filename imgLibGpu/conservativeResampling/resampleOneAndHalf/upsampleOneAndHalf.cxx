#if HOSTCODE
#include "upsampleOneAndHalf.h"
#endif

#include "conservativeResampling/conservativeResamplingSettings.h"

//================================================================
//
// Instance
//
//================================================================

COMPILE_ASSERT(CONSERVATIVE_RESAMPLING_QUALITY == 0);
static devConstant float32 FILTER0[] = {+0.00166414f, +0.00043475f, -0.00723411f, +0.01780444f, -0.02365456f, +0.00821240f, +0.04609053f, -0.14923741f, +0.33710882f, +0.74748520f, +0.08257036f, -0.10897243f, +0.07078922f, -0.02592033f, -0.00206014f, +0.01065854f, -0.00819722f, +0.00336871f, -0.00035155f, -0.00055936f};
static devConstant float32 FILTER1[] = {-0.00141998f, +0.00345938f, -0.00443153f, -0.00000000f, +0.01420508f, -0.03610672f, +0.05006276f, -0.02482460f, -0.09163485f, +0.59069046f, +0.59069046f, -0.09163485f, -0.02482460f, +0.05006276f, -0.03610672f, +0.01420508f, +0.00000000f, -0.00443153f, +0.00345938f, -0.00141998f};
static devConstant float32 FILTER2[] = {-0.00055936f, -0.00035155f, +0.00336871f, -0.00819722f, +0.01065854f, -0.00206014f, -0.02592033f, +0.07078922f, -0.10897243f, +0.08257036f, +0.74748520f, +0.33710882f, -0.14923741f, +0.04609053f, +0.00821240f, -0.02365456f, +0.01780444f, -0.00723411f, +0.00043475f, +0.00166414f};
static const Space FILTER_SRC_SHIFT = -9;

//----------------------------------------------------------------

#define FUNCSPACE conservativeResampling
#define FUNCNAME upsampleOneAndHalf

#define PACK_SIZE 3
#define PACK_TO_SRC_FACTOR 2

#define HORIZONTAL_FIRST 0

#define FOREACH_TYPE(action) \
    \
    TMP_MACRO(int8, int8, int8, 1) \
    TMP_MACRO(uint8, uint8, uint8, 1) \
    TMP_MACRO(int16, int16, int16, 1) \
    TMP_MACRO(uint16, uint16, uint16, 1) \
    TMP_MACRO(float16, float16, float16, 1) \
    \
    TMP_MACRO(int8_x2, int8_x2, int8_x2, 2) \
    TMP_MACRO(uint8_x2, uint8_x2, uint8_x2, 2) \
    TMP_MACRO(int16_x2, int16_x2, int16_x2, 2) \
    TMP_MACRO(uint16_x2, uint16_x2, uint16_x2, 2) \
    TMP_MACRO(float16_x2, float16_x2, float16_x2, 2)

# include "rationalResample/rationalResampleMultiple.inl"
