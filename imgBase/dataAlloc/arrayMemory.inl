#include "arrayMemory.h"

#include <limits.h>

#include "errorLog/errorLog.h"
#include "point/point.h"
#include "numbers/int/intType.h"

//================================================================
//
// ArrayMemoryEx<Pointer>::realloc
//
//================================================================

template <typename Pointer>
stdbool ArrayMemoryEx<Pointer>::realloc(Space size, Space byteAlignment, AllocatorInterface<AddrU>& allocator, stdPars(ErrorLogKit))
{
    //
    // check size
    //

    REQUIRE(size >= 0);
    REQUIRE(byteAlignment >= 0);

    ////

    constexpr Space elemSize = Space(sizeof(typename PtrElemType<Pointer>::T));
    constexpr Space maxAllocCount = TYPE_MAX(Space) / elemSize;

    REQUIRE(size <= maxAllocCount);
    Space byteAllocSize = size * elemSize;

    //
    // Allocate; if successful, update array layout.
    //

    COMPILE_ASSERT(sizeof(SpaceU) <= sizeof(AddrU));

    AddrU newAddr = 0;
    require(allocator.alloc(SpaceU(byteAllocSize), SpaceU(byteAlignment), memoryDealloc, newAddr, stdPass));

    COMPILE_ASSERT(sizeof(Pointer) == sizeof(newAddr));
    Pointer newPtr = Pointer(newAddr);

    ////

    theAllocPtr = newPtr;
    theAllocSize = size;

    ////

    BaseArray::assignUnsafe(newPtr, size);

    returnTrue;
}
