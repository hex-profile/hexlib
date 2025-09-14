#include "gpuMatrixCopy.h"

#include "gpuSupport/gpuTool.h"
#include "gpuSupport/gpuTexTools.h"

//================================================================
//
// COPY_FOREACH
//
//================================================================

#define COPY_FOREACH(action, extra) \
    VECTOR_INT_FOREACH(action, extra) \
    VECTOR_FLOAT_FOREACH(action, extra) \

//================================================================
//
// gpuMatrixCopyFunc_*
//
//================================================================

#define TMP_MACRO(Type, o) \
    \
    GPUTOOL_2D_AP \
    ( \
        gpuMatrixCopyFunc_##Type, \
        PREP_EMPTY, \
        ((const Type, src)) \
        ((Type, dst)), \
        PREP_EMPTY, \
        *dst = *src; \
    )

COPY_FOREACH(TMP_MACRO, o)

#undef TMP_MACRO

//================================================================
//
// gpuMatrixCopy
//
//================================================================

#define TMP_MACRO(Type, o) \
    template <> \
    void gpuMatrixCopyImpl(const GpuMatrixAP<const Type>& src, const GpuMatrixAP<Type>& dst, stdPars(GpuProcessKit)) \
        {gpuMatrixCopyFunc_##Type(src, dst, stdPassThru);}

HOST_ONLY(COPY_FOREACH(TMP_MACRO, o))

#undef TMP_MACRO

//================================================================
//
// gpuArrayCopyFunc_*
//
//================================================================

#define TMP_MACRO(Type, o) \
    \
    GPUTOOL_1D \
    ( \
        gpuArrayCopyFunc_##Type, \
        PREP_EMPTY, \
        ((const Type, src)) \
        ((Type, dst)), \
        PREP_EMPTY, \
        *dst = *src; \
    )

COPY_FOREACH(TMP_MACRO, o)

#undef TMP_MACRO

//================================================================
//
// gpuArrayCopy
//
//================================================================

#define TMP_MACRO(Type, o) \
    template <> \
    void gpuArrayCopy(const GpuArray<const Type>& src, const GpuArray<Type>& dst, stdPars(GpuProcessKit)) \
        {gpuArrayCopyFunc_##Type(src, dst, stdPassThru);}

HOST_ONLY(COPY_FOREACH(TMP_MACRO, o))

#undef TMP_MACRO
