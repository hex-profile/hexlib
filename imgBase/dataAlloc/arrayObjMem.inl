#include "arrayObjMem.h"

#include "dataAlloc/arrayMemory.inl"

//================================================================
//
// ArrayObjMem<Type>::realloc
//
//================================================================

template <typename Type>
stdbool ArrayObjMem<Type>::realloc(Space size, Space byteAlignment, AllocatorObject<AddrU>& allocator, bool dataProcessing, stdPars(ErrorLogKit))
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

    return true;
}

//================================================================
//
// ArrayObjMem<Type>::dealloc
//
//================================================================

template <typename Type>
void ArrayObjMem<Type>::dealloc()
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
