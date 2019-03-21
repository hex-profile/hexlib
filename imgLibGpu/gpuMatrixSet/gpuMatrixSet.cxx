#include "gpuMatrixSet.h"

#include "gpuSupport/gpuTool.h"
#include "gpuSupport/gpuTexTools.h"
#include "vectorTypes/vectorOperations.h"
#include "gpuDevice/loadstore/storeNorm.h"

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
