#include "gpuPyramidMemory.h"

#include "numbers/int/intBase.h"
#include "numbers/float/floatBase.h"
#include "numbers/divRound.h"
#include "errorLog/errorLog.h"
#include "pyramid/pyramidScale.h"
#include "prepTools/prepIncDec.h"
#include "prepTools/prepFor.h"
#include "storage/rememberCleanup.h"
#include "dataAlloc/arrayMemory.inl"

//================================================================
//
// GpuPyramidMemory::~GpuPyramidMemory
//
//================================================================

template <typename Type>
void GpuPyramidMemory<Type>::dealloc()
{
    pyramidArray.dealloc();
    currentLayers = 0;
}

//================================================================
//
// GpuPyramidMemory::realloc
//
//================================================================

template <typename Type>
stdbool GpuPyramidMemory<Type>::reallocEx
(
    const PyramidScale& scale,
    const Point<float32>& baseScaleFactor,
    const Point<Space>& baseScaleLevels,
    Space newLevels,
    Space newLayers,
    const Point<Space>& newBaseSize,
    Rounding sizeRounding,
    const Point<Space>& extraEdge,
    Space baseByteAlignment,
    Space rowByteAlignment,
    AllocatorObject<GpuAddrU>& allocator,
    stdPars(ErrorLogKit)
)
{
    dealloc();

    ////

    using Self = GpuPyramidMemory<Type>;
    Self& self = *this;
    REMEMBER_CLEANUP1_EX(deallocCleanup, self.dealloc(), Self&, self);

    //
    // allocate
    //

    REQUIRE(newLevels >= 0);
    REQUIRE(newLayers >= 0);
    require(pyramidArray.realloc(newLevels, stdPass));

    ////

    Point<Space> (*scaleFunc)(const Point<Space>& size, const Point<float32>& factor) = 0;
    if (sizeRounding == RoundDown) scaleFunc = scaleImageSize<RoundDown>;
    if (sizeRounding == RoundUp) scaleFunc = scaleImageSize<RoundUp>;
    if (sizeRounding == RoundNearest) scaleFunc = scaleImageSize<RoundNearest>;
    REQUIRE(scaleFunc != 0);

    ////

    ARRAY_EXPOSE(pyramidArray);

    for_count (i, newLevels)
    {
        Point<Space> size = scaleFunc(newBaseSize, baseScaleFactor * scale(-i + baseScaleLevels)) + extraEdge;
        require(pyramidArrayPtr[i].reallocEx(newLayers, size, baseByteAlignment, rowByteAlignment, allocator, stdPass));
    }

    ////

    currentLayers = newLayers;
    deallocCleanup.cancel();

    returnTrue;
}

//================================================================
//
// GpuPyramidMemory::levelSize
//
//================================================================

template <typename Type>
Point<Space> GpuPyramidMemory<Type>::levelSize(Space level) const
{
    ARRAY_EXPOSE(pyramidArray);

    Point<Space> result = point(0);

    if (SpaceU(level) < SpaceU(pyramidArraySize))
        result = pyramidArrayPtr[level].getLayerInline(0).size();

    return result;
}

//================================================================
//
// GpuPyramidMemory::operator[]
//
//================================================================

template <typename Type>
GpuMatrix<Type> GpuPyramidMemory<Type>::operator[] (Space level) const
{
    ARRAY_EXPOSE(pyramidArray);

    GpuMatrix<Type> result;

    if (SpaceU(level) < SpaceU(pyramidArraySize))
        result = pyramidArrayPtr[level].getLayerInline(0);

    return result;
}

//================================================================
//
// GpuPyramidMemory::getLayer
//
//================================================================

template <typename Type>
GpuMatrix<Type> GpuPyramidMemory<Type>::getLayer(Space level, Space layer) const
{
    GpuMatrix<Type> result;

    ARRAY_EXPOSE(pyramidArray);

    if (SpaceU(level) < SpaceU(pyramidArraySize))
        result = pyramidArrayPtr[level].getLayerInline(layer);

    return result;
}

//================================================================
//
// GpuPyramidMemory<Type>::getLevel
//
//================================================================

template <typename Type>
const GpuLayeredMatrix<Type>& GpuPyramidMemory<Type>::getLevel(Space level) const
{
    const GpuLayeredMatrix<Type>* result = &emptyLayeredMatrix;

    ARRAY_EXPOSE(pyramidArray);

    if (SpaceU(level) < SpaceU(pyramidArraySize))
        result = &pyramidArrayPtr[level];

    return *result;
}

//================================================================
//
// GpuPyramidMemory::getGpuLayout
//
//================================================================

template <typename Type>
bool GpuPyramidMemory<Type>::getGpuLayout(GpuPtr(Type)& basePointer, GpuPyramidLayout& layout) const
{
    ARRAY_EXPOSE(pyramidArray);

    Space levels = pyramidArraySize;
    ensure(levels <= GpuPyramidLayout::maxLevels);

    Space layers = currentLayers;

    ////

    layout.levels = levels;
    layout.layers = layers;

    ////

    basePointer = pyramidArrayPtr[0].getBaseImagePtr();
    GpuAddrU baseAddr = GpuAddrU(basePointer);

    ////

    for_count (l, levels)
    {
        ImageStorage& storage = pyramidArrayPtr[l];

        Point<Space> imageSize = storage.getImageSize();

        GpuPyramidLevelLayout& p = layout.levelData[l];

        p.size = imageSize;
        p.pitch = storage.getImagePitch();
        p.layerBytePitch = storage.getLayerPitch() * sizeof(Type);
        p.memOffset = GpuAddrU(storage.getBaseImagePtr()) - baseAddr;
    }

    ////

    return true;
}

//================================================================
//
// instantiations
//
//================================================================

#define TMP_MACRO(Type, o) \
    template class GpuPyramidMemory<Type>; \

VECTOR_INT_FOREACH(TMP_MACRO, o)
VECTOR_FLOAT_FOREACH(TMP_MACRO, o)

#undef TMP_MACRO
