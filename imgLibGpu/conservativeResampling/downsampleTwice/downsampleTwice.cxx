#if HOSTCODE
#include "downsampleTwice.h"
#endif

#include "conservativeResampling/conservativeResamplingFilters.h"

//================================================================
//
// Instance
//
//================================================================

#define FUNCNAME downsampleTwice

#define FILTER0 conservativeResampling::downsampleTwice::filter
#define FILTER_SRC_SHIFT conservativeResampling::downsampleTwice::filterSrcShift

#define PACK_SIZE 1
#define PACK_TO_SRC_FACTOR 2

#define HORIZONTAL_FIRST 1

#define TASK_COUNT 1

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
