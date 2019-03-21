#include "gpuMatrixMemory.h"

#include "vectorTypes/vectorType.h"
#include "errorLog/errorLog.h"
#include "data/spacex.h"

//================================================================
//
// GpuMatrixMemory<Type>::reallocEx
//
// ~100 instructions on x86, and ~40 instructions allocator
//
//================================================================

template <typename Type>
bool GpuMatrixMemory<Type>::reallocEx(const Point<Space>& size, Space baseByteAlignment, Space rowByteAlignment, AllocatorObject<AddrU>& allocator, stdPars(ErrorLogKit))
{
    stdBegin;

    Space sizeX = size.X;
    Space sizeY = size.Y;

    const Space elemSize = (Space) sizeof(Type);

    //
    // Row alignment is less or equal to base aligment.
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
    // allocation height
    //

    Space allocTotalSize = 0;
    REQUIRE(safeMul(alignedSizeX, sizeY, allocTotalSize));

    ////

    const Space maxAllocSize = TYPE_MAX(Space) / elemSize;
    REQUIRE(allocTotalSize <= maxAllocSize);
    Space byteAllocSize = allocTotalSize * elemSize;

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

    BaseMatrix::assign(newPtr, alignedSizeX, sizeX, sizeY, matrixPreconditionsAreVerified());

    stdEnd;
}

//================================================================
//
// GpuMatrixMemory<Type>::dealloc
//
//================================================================

template <typename Type>
void GpuMatrixMemory<Type>::dealloc()
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
// GpuMatrixMemory::resize
//
//================================================================

template <typename Type>
bool GpuMatrixMemory<Type>::resize(Space sizeX, Space sizeY)
{
    require(SpaceU(sizeX) <= SpaceU(allocSize.X));
    require(SpaceU(sizeY) <= SpaceU(allocSize.Y));

    Space alignedSizeX = (sizeX + allocAlignMask) & (~allocAlignMask); // overflow impossible
    BaseMatrix::assign(allocPtr, alignedSizeX, sizeX, sizeY, matrixPreconditionsAreVerified());

    return true;
}
