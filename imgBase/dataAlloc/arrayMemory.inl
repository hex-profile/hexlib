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
stdbool ArrayMemoryEx<Pointer>::realloc(Space size, Space byteAlignment, AllocatorObject<AddrU>& allocator, stdPars(ErrorLogKit))
{
    stdBegin;

    //
    // check size
    //

    REQUIRE(size >= 0);
    REQUIRE(byteAlignment >= 0);

    ////

    const Space elemSize = (Space) sizeof(typename PtrElemType<Pointer>::T);

    const Space maxAllocCount = TYPE_MAX(Space) / elemSize;

    REQUIRE(size <= maxAllocCount);
    Space byteAllocSize = size * elemSize;

    //
    // Allocate; if successful, update array layout.
    //

    AddrU newAddr = 0;
    require(allocator.func.alloc(allocator.state, byteAllocSize, byteAlignment, memoryDealloc, newAddr, stdPass));

    COMPILE_ASSERT(sizeof(Pointer) == sizeof(newAddr));
    Pointer newPtr = Pointer(newAddr);

    ////

    theAllocPtr = newPtr;
    theAllocSize = size;

    ////

    BaseArray::assign(newPtr, size, arrayPreconditionsAreVerified());

    stdEnd;
}
