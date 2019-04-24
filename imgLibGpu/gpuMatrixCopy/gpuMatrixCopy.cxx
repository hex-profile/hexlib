#include "gpuMatrixCopy.h"

#include "gpuSupport/gpuTool.h"
#include "gpuSupport/gpuTexTools.h"

//================================================================
//
// GPU_MATRIX_COPY_FOREACH
//
//================================================================

#define GPU_MATRIX_COPY_FOREACH(action, extra) \
    VECTOR_INT_FOREACH(action, extra) \
    VECTOR_FLOAT_FOREACH(action, extra) \

//================================================================
//
// Functions
//
//================================================================

#define TMP_MACRO(Type, o) \
    \
    GPUTOOL_2D \
    ( \
        gpuMatrixCopyFunc_##Type, \
        PREP_EMPTY, \
        ((const Type, src)) \
        ((Type, dst)), \
        PREP_EMPTY, \
        *dst = *src; \
    )

GPU_MATRIX_COPY_FOREACH(TMP_MACRO, o)

#undef TMP_MACRO

//================================================================
//
// gpuMatrixCopyFunc_*
//
//================================================================

#define TMP_MACRO(Type, o) \
    template <> \
    stdbool gpuMatrixCopy(const GpuMatrix<const Type>& src, const GpuMatrix<Type>& dst, stdPars(GpuProcessKit)) \
        {return gpuMatrixCopyFunc_##Type(src, dst, stdPassThru);}

HOST_ONLY(GPU_MATRIX_COPY_FOREACH(TMP_MACRO, o))

#undef TMP_MACRO
