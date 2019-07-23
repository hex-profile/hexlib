#include "downsampleOneAndHalf.h"

#include "gaussSincResampling/gaussSincResamplingSettings.h"

//================================================================
//
// downsampleOneAndHalfConservative
//
//================================================================

COMPILE_ASSERT(GAUSS_SINC_RESAMPLING_QUALITY == 0);
static devConstant float32 conservativeFilter0[] = {+0.00110935f, +0.00224565f, -0.00295087f, -0.00482240f, +0.00710520f, +0.00945891f, -0.01576861f, -0.01727902f, +0.03333590f, +0.03072488f, -0.07264324f, -0.06101802f, +0.22472360f, +0.49828885f, +0.39333031f, +0.05504308f, -0.09948469f, -0.01653026f, +0.04718953f, +0.00547456f, -0.02404282f, -0.00137333f, +0.01186880f, +0.00000000f, -0.00546444f, +0.00028981f, +0.00230354f, -0.00023435f, -0.00087992f};
static devConstant float32 conservativeFilter1[] = {-0.00087992f, -0.00023435f, +0.00230354f, +0.00028981f, -0.00546444f, -0.00000000f, +0.01186880f, -0.00137333f, -0.02404282f, +0.00547456f, +0.04718953f, -0.01653026f, -0.09948469f, +0.05504308f, +0.39333031f, +0.49828885f, +0.22472360f, -0.06101802f, -0.07264324f, +0.03072488f, +0.03333590f, -0.01727902f, -0.01576861f, +0.00945891f, +0.00710520f, -0.00482240f, -0.00295087f, +0.00224565f, +0.00110935f};
static const Space conservativeFilterSrcShift = -13;

//----------------------------------------------------------------

#define FUNCSPACE gaussSincResampling
#define FUNCNAME downsampleOneAndHalfConservative

#define PACK_SIZE 2
#define PACK_TO_SRC_FACTOR 3

#define FILTER0 conservativeFilter0
#define FILTER1 conservativeFilter1
#define FILTER_SRC_SHIFT conservativeFilterSrcShift

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

//================================================================
//
// downsampleOneAndHalfBalanced
//
//================================================================

COMPILE_ASSERT(GAUSS_SINC_RESAMPLING_QUALITY == 0);
static devConstant float32 balancedFilter0[] = {+0.00115941f, +0.00207934f, -0.00718292f, +0.00599237f, +0.00970111f, -0.03068247f, +0.02396085f, +0.03770793f, -0.12477910f, +0.12243793f, +0.63528845f, +0.41832906f, -0.08427581f, -0.04799029f, +0.05994917f, -0.01918415f, -0.01222851f, +0.01530122f, -0.00465811f, -0.00274517f, +0.00312048f, -0.00085302f, -0.00044777f};
static devConstant float32 balancedFilter1[] = {-0.00044777f, -0.00085302f, +0.00312048f, -0.00274517f, -0.00465811f, +0.01530122f, -0.01222851f, -0.01918415f, +0.05994917f, -0.04799029f, -0.08427581f, +0.41832906f, +0.63528845f, +0.12243793f, -0.12477910f, +0.03770793f, +0.02396085f, -0.03068247f, +0.00970111f, +0.00599237f, -0.00718292f, +0.00207934f, +0.00115941f};
static const Space balancedFilterSrcShift = -10;

//----------------------------------------------------------------

#undef FUNCNAME
#define FUNCNAME downsampleOneAndHalfBalanced

#undef FILTER0
#undef FILTER1
#undef FILTER_SRC_SHIFT

#define FILTER0 balancedFilter0
#define FILTER1 balancedFilter1
#define FILTER_SRC_SHIFT balancedFilterSrcShift

# include "rationalResample/rationalResampleMultiple.inl"

