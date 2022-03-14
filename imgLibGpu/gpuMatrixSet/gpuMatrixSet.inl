#include "gpuMatrixSet.h"

#include "gpuSupport/gpuTool.h"
#include "gpuSupport/gpuMixedCode.h"

//================================================================
//
// GPU_MATRIX_SET_DEFINE
//
//================================================================

#define GPU_MATRIX_SET_DEFINE(Type, o) \
    \
    GPUTOOL_2D \
    ( \
        gpuMatrixSetFunc_##Type, \
        PREP_EMPTY, \
        ((Type, dst)), \
        ((Type, value)), \
        Type v = value; \
        *dst = v; \
    ) \
    \
    HOST_ONLY(GPU_MATRIX_SET__HOST_FUNC(Type))

//----------------------------------------------------------------

#define GPU_MATRIX_SET__HOST_FUNC(Type) \
    \
    template <> \
    stdbool gpuMatrixSet(const GpuMatrix<Type>& dst, const Type& value, stdPars(GpuProcessKit)) \
        {return gpuMatrixSetFunc_##Type(dst, value, stdPassThru);}
