#if HOSTCODE
#include "upsampleTwice.h"
#endif

#include "conservativeResampling/conservativeResamplingSettings.h"

//================================================================
//
// Instance
//
//================================================================

COMPILE_ASSERT(CONSERVATIVE_RESAMPLING_QUALITY == 0);
static devConstant float32 FILTER0[] = {+0.00219537f, -0.00068659f, -0.00598392f, +0.01810818f, -0.02823106f, +0.01893732f, +0.03197533f, -0.14547108f, +0.40408355f, +0.72100016f, +0.02912540f, -0.08955916f, +0.06913369f, -0.03092485f, +0.00295006f, +0.00800217f, -0.00756503f, +0.00364235f, -0.00073186f};
static devConstant float32 FILTER1[] = {-0.00073186f, +0.00364235f, -0.00756503f, +0.00800217f, +0.00295006f, -0.03092485f, +0.06913369f, -0.08955916f, +0.02912540f, +0.72100016f, +0.40408355f, -0.14547108f, +0.03197533f, +0.01893732f, -0.02823106f, +0.01810818f, -0.00598392f, -0.00068659f, +0.00219537f};
static const Space FILTER_SRC_SHIFT = -9;

//----------------------------------------------------------------

#define FUNCSPACE conservativeResampling
#define FUNCNAME upsampleTwice

#define PACK_SIZE 2
#define PACK_TO_SRC_FACTOR 1

#define HORIZONTAL_FIRST 1

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
