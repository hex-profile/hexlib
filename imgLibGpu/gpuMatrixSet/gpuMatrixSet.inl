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
    GPUTOOL_2D_AP \
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
    void gpuMatrixSetImpl(const GpuMatrixAP<Type>& dst, const Type& value, stdPars(GpuProcessKit)) \
        {gpuMatrixSetFunc_##Type(dst, value, stdPassThru);}

//================================================================
//
// GPU_ARRAY_SET_DEFINE
//
//================================================================

#define GPU_ARRAY_SET_DEFINE(Type, o) \
    \
    GPUTOOL_1D \
    ( \
        gpuArraySetFunc_##Type, \
        PREP_EMPTY, \
        ((Type, dst)), \
        ((Type, value)), \
        Type v = value; \
        *dst = v; \
    ) \
    \
    HOST_ONLY(GPU_ARRAY_SET__HOST_FUNC(Type))

//----------------------------------------------------------------

#define GPU_ARRAY_SET__HOST_FUNC(Type) \
    \
    template <> \
    void gpuArraySet(const GpuArray<Type>& dst, const Type& value, stdPars(GpuProcessKit)) \
        {gpuArraySetFunc_##Type(dst, value, stdPassThru);}
