#if HOSTCODE
#include "upsampleFourTimes.h"
#endif

#include "gpuDevice/gpuDevice.h"

//================================================================
//
// Instance
//
//================================================================

#define FUNCNAME upsampleFourTimes

static devConstant float32 FILTER0[] = {+0.00291439f, -0.00253125f, -0.00337366f, +0.01709718f, -0.03347612f, +0.03512213f, +0.00583154f, -0.12686379f, +0.50136652f, +0.66441936f, -0.03917403f, -0.05754225f, +0.06185693f, -0.03532735f, +0.00939264f, +0.00390127f, -0.00617294f, +0.00372287f, -0.00116343f};
static devConstant float32 FILTER1[] = {+0.00139140f, +0.00094758f, -0.00769771f, +0.01736938f, -0.02111394f, +0.00302406f, +0.05197515f, -0.14858887f, +0.30323242f, +0.75632731f, +0.11126306f, -0.11761747f, +0.07046246f, -0.02279168f, -0.00471586f, +0.01190489f, -0.00839484f, +0.00315888f, -0.00013622f}; 
static devConstant float32 FILTER2[] = {-0.00013622f, +0.00315888f, -0.00839484f, +0.01190489f, -0.00471586f, -0.02279168f, +0.07046246f, -0.11761747f, +0.11126306f, +0.75632731f, +0.30323242f, -0.14858887f, +0.05197515f, +0.00302406f, -0.02111394f, +0.01736938f, -0.00769771f, +0.00094758f, +0.00139140f};
static devConstant float32 FILTER3[] = {-0.00116343f, +0.00372287f, -0.00617294f, +0.00390127f, +0.00939264f, -0.03532735f, +0.06185693f, -0.05754225f, -0.03917403f, +0.66441936f, +0.50136652f, -0.12686379f, +0.00583154f, +0.03512213f, -0.03347612f, +0.01709718f, -0.00337366f, -0.00253125f, +0.00291439f};

#define FILTER_SRC_SHIFT -9

#define PACK_SIZE 4
#define PACK_TO_SRC_FACTOR 1

#define HORIZONTAL_FIRST 0

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
#define FUNCNAME upsampleFourTimesDual

#undef TASK_COUNT
#define TASK_COUNT 2

# include "rationalResample/rationalResampleMultiple.inl"
