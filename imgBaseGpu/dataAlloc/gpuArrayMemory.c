#include "gpuArrayMemory.h"

#include "dataAlloc/arrayMemory.inl"
#include "vectorTypes/vectorType.h"

//================================================================
//
// instantiations
//
//================================================================

#define TMP_MACRO(Type, o) \
    template class ArrayMemoryEx<GpuPtr(Type)>; \
    template class ArrayMemoryEx<GpuPtr(Point<Type>)>;

VECTOR_INT_FOREACH(TMP_MACRO, o)
VECTOR_FLOAT_FOREACH(TMP_MACRO, o)

#undef TMP_MACRO
