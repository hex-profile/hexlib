#include "arrayObjectMemory.h"

#include "dataAlloc/arrayMemory.inl"

//================================================================
//
// ArrayObjectMemory<Type>::realloc
//
//================================================================

template <typename Type>
stdbool ArrayObjectMemory<Type>::realloc(Space size, Space byteAlignment, AllocatorObject<AddrU>& allocator, bool dataProcessing, stdPars(ErrorLogKit))
{
    dealloc(); // Call destructors and dealloc memory

    ////

    require(ArrayMemoryEx<Type*>::realloc(size, byteAlignment, allocator, stdPassThru));

    ////

    Type* allocPtr = this->allocPtr();
    Space allocSize = this->maxSize();

    if (dataProcessing)
    {
        for (Space i = 0; i < allocSize; ++i)
            constructDefault(allocPtr[i]); // (exceptions are forbidden!)

        constructed = true;
    }

    ////

    returnTrue;
}

//================================================================
//
// ArrayObjectMemory<Type>::dealloc
//
//================================================================

template <typename Type>
void ArrayObjectMemory<Type>::dealloc()
{
    Type* allocPtr = this->allocPtr();
    Space allocSize = this->maxSize();

    if (constructed)
    {
        for (Space i = allocSize-1; i >= 0; --i)
            destruct(allocPtr[i]);
    }

    ArrayMemoryEx<Type*>::dealloc();
    constructed = false;
}