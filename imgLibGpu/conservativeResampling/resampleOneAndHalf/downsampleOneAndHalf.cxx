#include "downsampleOneAndHalf.h"

#include "conservativeResampling/conservativeResamplingSettings.h"

//================================================================
//
// Instance
//
//================================================================

COMPILE_ASSERT(CONSERVATIVE_RESAMPLING_QUALITY == 0);
static devConstant float32 FILTER0[] = {+0.00110935f, +0.00224565f, -0.00295087f, -0.00482240f, +0.00710520f, +0.00945891f, -0.01576861f, -0.01727902f, +0.03333590f, +0.03072488f, -0.07264324f, -0.06101802f, +0.22472360f, +0.49828885f, +0.39333031f, +0.05504308f, -0.09948469f, -0.01653026f, +0.04718953f, +0.00547456f, -0.02404282f, -0.00137333f, +0.01186880f, +0.00000000f, -0.00546444f, +0.00028981f, +0.00230354f, -0.00023435f, -0.00087992f};
static devConstant float32 FILTER1[] = {-0.00087992f, -0.00023435f, +0.00230354f, +0.00028981f, -0.00546444f, -0.00000000f, +0.01186880f, -0.00137333f, -0.02404282f, +0.00547456f, +0.04718953f, -0.01653026f, -0.09948469f, +0.05504308f, +0.39333031f, +0.49828885f, +0.22472360f, -0.06101802f, -0.07264324f, +0.03072488f, +0.03333590f, -0.01727902f, -0.01576861f, +0.00945891f, +0.00710520f, -0.00482240f, -0.00295087f, +0.00224565f, +0.00110935f};
static const Space FILTER_SRC_SHIFT = -13;

//----------------------------------------------------------------

#define FUNCSPACE conservativeResampling
#define FUNCNAME downsampleOneAndHalf

#define PACK_SIZE 2
#define PACK_TO_SRC_FACTOR 3

#define HORIZONTAL_FIRST 1

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
