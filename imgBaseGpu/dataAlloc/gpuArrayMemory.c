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

TMP_MACRO(int64, o)
TMP_MACRO(uint64, o)
TMP_MACRO(float64, o)

#undef TMP_MACRO
