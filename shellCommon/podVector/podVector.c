#include "podVector.h"

#include <exception>

//================================================================
//
// PodVectorBase::throwError
//
//================================================================

[[noreturn]]
void PodVectorBase::throwError()
{
    throw std::bad_alloc();
}

//================================================================
//
// PodVectorBase::deallocBytes
//
//================================================================

void PodVectorBase::deallocBytes()
{
    free(allocPtr);

    allocPtr = nullptr;
}

//================================================================
//
// PodVectorBase::tryReallocToBytes
//
//================================================================

bool PodVectorBase::tryReallocToBytes(size_t newSizeInBytes)
{
    if (newSizeInBytes)
    {
        auto newPtr = realloc(allocPtr, newSizeInBytes);
        ensure(newPtr);
        allocPtr = newPtr;
    }
    else
    {
        free(allocPtr);
        allocPtr = nullptr;
    }

    return true;
}

//================================================================
//
// PodVectorBase::reallocToBytes
//
//================================================================

void PodVectorBase::reallocToBytes(size_t newSizeInBytes)
{
    if_not (tryReallocToBytes(newSizeInBytes))
        throwError();
}
