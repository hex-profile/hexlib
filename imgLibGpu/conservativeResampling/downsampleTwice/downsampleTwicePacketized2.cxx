#if HOSTCODE
#include "downsampleTwicePacketized2.h"
#endif

#include "conservativeResampling/conservativeResamplingFilters.h"

//================================================================
//
// Instance
//
//================================================================

#define FUNCNAME downsampleTwicePacketized2

#define FILTER0 conservativeResampling::downsampleTwicePacketized2::filter0
#define FILTER1 conservativeResampling::downsampleTwicePacketized2::filter1
#define FILTER_SRC_SHIFT conservativeResampling::downsampleTwicePacketized2::filterSrcShift

#define PACK_SIZE 2
#define PACK_TO_SRC_FACTOR 4

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
