#pragma once

#include "data/space.h"
#include "compileTools/compileTools.h"
#include "storage/addrSpace.h"

//================================================================
//
// helpRead
// helpModify
//
//================================================================

template <typename Element>
sysinline Element& helpModify(Element& ref)
    {return ref;}

template <typename Element>
sysinline Element& helpRead(Element& ref)
    {return ref;}

//================================================================
//
// Uncast a protected pointer.
//
//================================================================

template <typename Pointer>
sysinline Pointer unsafePtr(Pointer ptr, Space sizeX)
    {return ptr;}

template <typename Pointer>
sysinline Pointer unsafePtr(Pointer ptr, Space sizeX, Space sizeY)
    {return ptr;}

//================================================================
//
// isPtrAligned
//
//================================================================

template <Space alignment, typename Type>
sysinline bool isPtrAligned(Type* ptr)
{
    COMPILE_ASSERT(alignment >= 1 && COMPILE_IS_POWER2(alignment));
    return (CpuAddrU(ptr) & CpuAddrU(alignment - 1)) == 0;
}
