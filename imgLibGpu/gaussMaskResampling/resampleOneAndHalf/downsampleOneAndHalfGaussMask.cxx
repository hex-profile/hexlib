#include "downsampleOneAndHalfGaussMask.h"

#include "gpuDevice/gpuDevice.h"

//================================================================
//
// downsampleOneAndHalfGaussMaskInitial
//
//================================================================

static devConstant float32 initialFilter0[] = {+0.01948875f, +0.16907312f, +0.42677512f, +0.31344228f, +0.06698076f, +0.00416463f, +0.00007534f};
static devConstant float32 initialFilter1[] = {+0.00007534f, +0.00416463f, +0.06698076f, +0.31344228f, +0.42677512f, +0.16907312f, +0.01948875f};
static const Space initialSrcShift = -2;

//----------------------------------------------------------------

#define FUNCSPACE gaussMaskResampling
#define FUNCNAME downsampleOneAndHalfGaussMaskInitial

#define PACK_SIZE 2
#define PACK_TO_SRC_FACTOR 3

#define FILTER0 initialFilter0
#define FILTER1 initialFilter1
#define FILTER_SRC_SHIFT initialSrcShift

#define HORIZONTAL_FIRST 1

#define FOREACH_TYPE(action) \
    \
    TMP_MACRO(uint8, uint8, uint8, 1) \
    TMP_MACRO(uint16, uint16, uint16, 1) \
    TMP_MACRO(float16, float16, float16, 1)

# include "rationalResample/rationalResampleMultiple.inl"

//================================================================
//
// downsampleOneAndHalfGaussMaskSustaining
//
//================================================================

static devConstant float32 sustainingFilter0[] = {+0.00214486f, +0.10479062f, +0.55481288f, +0.31832579f, +0.01979239f, +0.00013336f, +0.00000010f};
static devConstant float32 sustainingFilter1[] = {+0.00000010f, +0.00013336f, +0.01979239f, +0.31832579f, +0.55481288f, +0.10479062f, +0.00214486f};
static const Space sustainingSrcShift = -2;

//----------------------------------------------------------------

#undef FUNCNAME
#define FUNCNAME downsampleOneAndHalfGaussMaskSustaining

#undef FILTER0
#undef FILTER1
#undef FILTER_SRC_SHIFT

#define FILTER0 sustainingFilter0
#define FILTER1 sustainingFilter1
#define FILTER_SRC_SHIFT sustainingSrcShift

# include "rationalResample/rationalResampleMultiple.inl"

