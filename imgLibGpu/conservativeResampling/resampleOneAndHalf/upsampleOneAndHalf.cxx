#if HOSTCODE
#include "upsampleOneAndHalf.h"
#endif

#include "conservativeResampling/conservativeResamplingFilters.h"

//================================================================
//
// Instance
//
//================================================================

#define FUNCNAME upsampleOneAndHalf

#define FILTER0 conservativeResampling::upsampleOneAndHalf::filter0
#define FILTER1 conservativeResampling::upsampleOneAndHalf::filter1
#define FILTER2 conservativeResampling::upsampleOneAndHalf::filter2

#define FILTER_SRC_SHIFT conservativeResampling::upsampleOneAndHalf::filterSrcShift

#define PACK_SIZE 3
#define PACK_TO_SRC_FACTOR 2

#define HORIZONTAL_FIRST 0

#define TASK_COUNT 1

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
