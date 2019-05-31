#include "matrixBase.h"

//================================================================
//
// ensure
//
//================================================================

#define ensure(condition) \
    if (condition) ; else return false

//================================================================
//
// Checks preconditions:
//
// (1) sizeX >= 0 && sizeY >= 0
// (2) sizeX <= |pitch|
// (3) (sizeY * pitch * elemSize) fits into Space type.
//
//================================================================

template <Space elemSize>
bool matrixBaseIsValid(Space sizeX, Space sizeY, Space pitch)
{
    ensure(sizeX >= 0);
    ensure(sizeY >= 0);

    ////

    static_assert(elemSize >= 1, "");
    constexpr Space maxArea = spaceMax / elemSize;

    ////

    Space absPitch = pitch < 0 ? -pitch : pitch;
    ensure(sizeX <= absPitch);

    if (sizeY >= 1)
    {
        Space maxWidth = maxArea / sizeY;
        ensure(absPitch <= maxWidth);
    }

    return true;
}

//----------------------------------------------------------------

template bool matrixBaseIsValid<0x01>(Space sizeX, Space sizeY, Space pitch);
template bool matrixBaseIsValid<0x02>(Space sizeX, Space sizeY, Space pitch);
template bool matrixBaseIsValid<0x04>(Space sizeX, Space sizeY, Space pitch);
template bool matrixBaseIsValid<0x08>(Space sizeX, Space sizeY, Space pitch);
template bool matrixBaseIsValid<0x10>(Space sizeX, Space sizeY, Space pitch);

//================================================================
//
// matrixBaseCompileTest
//
//================================================================

bool matrixBaseCompileTest(uint8_t* srcMemPtr, Space srcMemPitch, Space srcSizeX, Space srcSizeY)
{
    // Create an empty matrix.
    MatrixBase<int> intMatrix;

    // Convert a matrix to a read-only matrix.
    MatrixBase<const int> constIntMatrix = intMatrix;
    MatrixBase<const int> anotherConstMatrix = makeConst(intMatrix);

    // Construct matrix from details: ptr, pitch and size.
    MatrixBase<const uint8_t> example(srcMemPtr, srcMemPitch, srcSizeX, srcSizeY);

    // Setup matrix from details: ptr, pitch and size.
    example.assign(srcMemPtr, srcMemPitch, srcSizeX, srcSizeY);

    // Make the matrix empty.
    example.assignNull();

    // Make the matrix empty.
    ArrayBase<uint8_t> someArray(srcMemPtr, srcSizeX);
    MatrixBase<const uint8_t> arrayFromMatrix = someArray;

    // Access matrix details (decomposing matrix is better way):
    ensure(example.memPtr() != 0);
    ensure(example.memPitch() != 0);
    ensure(example.sizeX() != 0);
    ensure(example.sizeY() != 0);

    // Check that a matrix has non-zero size
    ensure(hasData(example));

    return true;
}
