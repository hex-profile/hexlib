#include "matrixMemory.h"

#include <limits.h>

#include "errorLog/errorLog.h"
#include "data/spacex.h"

//================================================================
//
// MatrixMemoryEx<Pointer>::realloc
//
// (~80 instructions x86)
//
//================================================================

template <typename Pointer>
stdbool MatrixMemoryEx<Pointer>::realloc(const Point<Space>& size, Space baseByteAlignment, Space rowByteAlignment, AllocatorObject<AddrU>& allocator, stdPars(ErrorLogKit))
{
    Space sizeX = size.X;
    Space sizeY = size.Y;

    const Space elemSize = (Space) sizeof(typename PtrElemType<Pointer>::T);

    //
    // row alignment is less or equal to base aligment.
    //

    REQUIRE(isPower2(baseByteAlignment) && isPower2(rowByteAlignment));
    REQUIRE(0 <= rowByteAlignment && rowByteAlignment <= baseByteAlignment);

    //
    // compute alignment in elements
    //

    REQUIRE(rowByteAlignment >= 1);

    Space alignment = 1;

    if (rowByteAlignment != 1)
    {
        alignment = SpaceU(rowByteAlignment) / SpaceU(elemSize);
        REQUIRE(alignment * elemSize == rowByteAlignment); // divides evenly
    }

    //
    // check the alignment is power of 2
    //

    REQUIRE(isPower2(alignment));
    Space alignmentMask = alignment - 1;

    //
    // check size
    //

    REQUIRE(sizeX >= 0 && sizeY >= 0);

    //
    // align image size X
    // 

    Space sizeXplusMask = 0;
    REQUIRE(safeAdd(sizeX, alignmentMask, sizeXplusMask));

    Space alignedSizeX = sizeXplusMask & (~alignmentMask);
    REQUIRE(alignedSizeX >= sizeX); // self-check

    //
    // allocation byte size
    //

    Space allocCount = 0;
    REQUIRE(safeMul(alignedSizeX, sizeY, allocCount)); // can multiply?

    ////

    const Space maxAllocCount = TYPE_MAX(Space) / elemSize;

    REQUIRE(allocCount <= maxAllocCount);
    Space byteAllocSize = allocCount * elemSize;

    //
    // Allocate; if successful, update matrix layout.
    //

    AddrU newAddr = 0;
    require(allocator.func.alloc(allocator.state, byteAllocSize, baseByteAlignment, memoryOwner, newAddr, stdPass));

    COMPILE_ASSERT(sizeof(Pointer) == sizeof(AddrU));
    Pointer newPtr = Pointer(newAddr);

    ////

    allocPtr = newPtr;
    allocSize = point(sizeX, sizeY);
    allocAlignMask = alignmentMask;

    ////

    BaseMatrix::assign(newPtr, alignedSizeX, sizeX, sizeY, MatrixValidityAssertion{});

    returnTrue;
}

//================================================================
//
// MatrixMemoryEx<Pointer>::dealloc
//
//================================================================

template <typename Pointer>
void MatrixMemoryEx<Pointer>::dealloc()
{
    memoryOwner.clear();

    ////

    allocPtr = Pointer(0);
    allocSize = point(0);
    allocAlignMask = 0;

    ////

    BaseMatrix::assignNull(); // clear base matrix
}

//================================================================
//
// MatrixMemoryEx::resize
//
//================================================================

template <typename Pointer>
bool MatrixMemoryEx<Pointer>::resize(Space sizeX, Space sizeY)
{
    ensure(SpaceU(sizeX) <= SpaceU(allocSize.X));
    ensure(SpaceU(sizeY) <= SpaceU(allocSize.Y));

    Space alignedSizeX = (sizeX + allocAlignMask) & (~allocAlignMask); // overflow impossible
    BaseMatrix::assign(allocPtr, alignedSizeX, sizeX, sizeY, MatrixValidityAssertion{});

    return true;
}
