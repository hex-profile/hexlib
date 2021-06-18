#include "gpuMatrixSet.h"

#include "gpuSupport/gpuTool.h"
#include "gpuSupport/gpuTexTools.h"
#include "vectorTypes/vectorOperations.h"
#include "gpuDevice/loadstore/storeNorm.h"

//================================================================
//
// GPU_MATRIX_SET_FOREACH
//
//================================================================

#define GPU_MATRIX_SET_FOREACH(action, extra) \
    VECTOR_INT_FOREACH(action, extra) \
    VECTOR_FLOAT_FOREACH(action, extra)

//================================================================
//
// Functions
//
//================================================================

#define TMP_MACRO(Type, o) \
    \
    GPUTOOL_2D \
    ( \
        gpuMatrixSetFunc_##Type, \
        PREP_EMPTY, \
        ((Type, dst)), \
        ((Type, value)), \
        Type v = value; \
        *dst = v; \
    )

GPU_MATRIX_SET_FOREACH(TMP_MACRO, o)

#undef TMP_MACRO

//================================================================
//
// gpuMatrixSetFunc_*
//
//================================================================

#define TMP_MACRO(Type, o) \
    \
    template <> \
    stdbool gpuMatrixSet(const GpuMatrix<Type>& dst, const Type& value, stdPars(GpuProcessKit)) \
        {return gpuMatrixSetFunc_##Type(dst, value, stdPassThru);}

HOST_ONLY(GPU_MATRIX_SET_FOREACH(TMP_MACRO, o))

#undef TMP_MACRO
