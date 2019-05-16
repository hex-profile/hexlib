#if HOSTCODE
#include "downsampleFourTimes.h"
#endif

#include "gpuDevice/gpuDevice.h"

//================================================================
//
// Instance
//
//================================================================

#define FUNCNAME downsampleFourTimes

static devConstant float32 FILTER0[] = {-0.00029084f, -0.00003406f, +0.00034787f, +0.00072855f, +0.00093066f, +0.00078977f, +0.00023691f, -0.00063277f, -0.00154314f, -0.00209884f, -0.00192455f, -0.00084336f, +0.00097526f, +0.00297641f, +0.00434261f, +0.00427403f, +0.00234801f, -0.00117904f, -0.00527881f, -0.00836851f, -0.00883129f, -0.00569827f, +0.00075606f, +0.00877999f, +0.01546327f, +0.01761671f, +0.01299459f, +0.00145780f, -0.01438467f, -0.02940619f, -0.03714952f, -0.03171398f, -0.00979290f, +0.02781749f, +0.07581280f, +0.12533387f, +0.16609455f, +0.18909354f, +0.18909354f, +0.16609455f, +0.12533387f, +0.07581280f, +0.02781749f, -0.00979290f, -0.03171398f, -0.03714952f, -0.02940619f, -0.01438467f, +0.00145780f, +0.01299459f, +0.01761671f, +0.01546327f, +0.00877999f, +0.00075606f, -0.00569827f, -0.00883129f, -0.00836851f, -0.00527881f, -0.00117904f, +0.00234801f, +0.00427403f, +0.00434261f, +0.00297641f, +0.00097526f, -0.00084336f, -0.00192455f, -0.00209884f, -0.00154314f, -0.00063277f, +0.00023691f, +0.00078977f, +0.00093066f, +0.00072855f, +0.00034787f, -0.00003406f, -0.00029084f};
#define FILTER_SRC_SHIFT (-36)

#define PACK_SIZE 1
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

////

#undef FUNCNAME 
#define FUNCNAME downsampleFourTimesDual

#undef TASK_COUNT
#define TASK_COUNT 2

# include "rationalResample/rationalResampleMultiple.inl"
