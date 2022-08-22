#include "gpuLayeredMatrixMemory.h"

#include "vectorTypes/vectorType.h"
#include "errorLog/errorLog.h"
#include "data/spacex.h"

//================================================================
//
// computeAlignedSize
//
//================================================================

inline bool computeAlignedSize(Space size, Space alignmentMask, Space& result)
{
    Space sizePlusMask = 0;
    ensure(safeAdd(size, alignmentMask, sizePlusMask));

    result = sizePlusMask & (~alignmentMask);
    return true;
}

//================================================================
//
// GpuLayeredMatrixMemory<Type>::reallocEx
//
// ~115 instructions x86, plus allocator
//
//================================================================

template <typename Type>
stdbool GpuLayeredMatrixMemory<Type>::reallocEx(Space layers, const Point<Space>& size, Space baseByteAlignment, Space rowByteAlignment, AllocatorInterface<AddrU>& allocator, stdPars(ErrorLogKit))
{
    REQUIRE(layers >= 0);

    Space sizeX = size.X;
    Space sizeY = size.Y;

    const Space elemSize = (Space) sizeof(Type);

    //
    // Row alignment is less or equal to base aligment.
    //

    REQUIRE(isPower2(baseByteAlignment) && isPower2(rowByteAlignment));
    COMPILE_ASSERT(COMPILE_IS_POWER2(sizeof(Type)));

    baseByteAlignment = clampMin(baseByteAlignment, elemSize);
    rowByteAlignment = clampMin(rowByteAlignment, elemSize);

    REQUIRE(rowByteAlignment <= baseByteAlignment);

    //
    // compute alignment in elements
    //

    Space rowAlignment = SpaceU(rowByteAlignment) / SpaceU(elemSize);
    Space baseAlignment = SpaceU(baseByteAlignment) / SpaceU(elemSize);

    Space rowAlignMask = rowAlignment - 1;
    Space baseAlignMask = baseAlignment - 1;

    //
    // check size
    //

    REQUIRE(sizeX >= 0 && sizeY >= 0);

    //
    // align image size X
    // 

    Space alignedSizeX = 0;
    REQUIRE(computeAlignedSize(sizeX, rowAlignMask, alignedSizeX));

    //
    // allocation image area (in elements)
    //

    Space imageArea = 0;
    REQUIRE(safeMul(alignedSizeX, sizeY, imageArea));

    Space alignedImageArea = 0;
    REQUIRE(computeAlignedSize(imageArea, baseAlignMask, alignedImageArea));

    //
    // allocation height
    //

    Space allocTotalSize = 0;
    REQUIRE(safeMul(layers, alignedImageArea, allocTotalSize));

    ////

    constexpr Space maxAllocSize = TYPE_MAX(Space) / elemSize;
    REQUIRE(allocTotalSize <= maxAllocSize);
    Space byteAllocSize = allocTotalSize * elemSize;

    //
    // Allocate; if successful, update matrix layout.
    //

    AddrU newAddr = 0;
    require(allocator.alloc(SpaceU(byteAllocSize), SpaceU(baseByteAlignment), memoryOwner, newAddr, stdPass));

    COMPILE_ASSERT(sizeof(Pointer) == sizeof(AddrU));
    Pointer newPtr = Pointer(newAddr);

    ////

    allocPtr = newPtr;
    allocSize = point(sizeX, sizeY);
    allocAlignMask = rowAlignMask;
    allocLayers = layers;
    allocLayerPitch = alignedImageArea;

    ////

    currentImageSize = point(sizeX, sizeY);
    currentImagePitch = alignedSizeX;
    currentLayers = layers;
    currentLayerPitch = alignedImageArea;

    returnTrue;
}

//================================================================
//
// GpuLayeredMatrixMemory<Type>::dealloc
//
//================================================================

template <typename Type>
void GpuLayeredMatrixMemory<Type>::dealloc()
{
    memoryOwner.clear();

    ////

    allocPtr = Pointer(0);
    allocSize = point(0);
    allocAlignMask = 0;
    allocLayers = 0;
    allocLayerPitch = 0;

    ////

    resizeNull();
}

//================================================================
//
// GpuLayeredMatrixMemory::resize
//
//================================================================

template <typename Type>
bool GpuLayeredMatrixMemory<Type>::resize(Space layers, Space sizeX, Space sizeY)
{
    ensure(SpaceU(layers) <= SpaceU(allocLayers));
    ensure(SpaceU(sizeX) <= SpaceU(allocSize.X));
    ensure(SpaceU(sizeY) <= SpaceU(allocSize.Y));

    //
    // Pitch is compressed for better DRAM locality
    //

    Space alignedSizeX = (sizeX + allocAlignMask) & (~allocAlignMask); // overflow impossible

    ////

    currentImageSize = point(sizeX, sizeY);
    currentImagePitch = alignedSizeX;
    currentLayers = layers;
    currentLayerPitch = alignedSizeX * sizeY;

    return true;
}

//================================================================
//
// instantiations
//
//================================================================

#define TMP_MACRO(Type, o) \
    template class GpuLayeredMatrixMemory<Type>;
  
VECTOR_INT_FOREACH(TMP_MACRO, o)
VECTOR_FLOAT_FOREACH(TMP_MACRO, o)

#undef TMP_MACRO
