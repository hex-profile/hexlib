#include "dbgptrMatrixPointer.h"

#include <stdlib.h>
#include <string.h>

#include "dbgptr/dbgptrCommonImpl.h"
#include "numbers/int/intBase.h"

//================================================================
//
// DebugMatrixPointerByteEngine::initEmpty
//
//================================================================

void DebugMatrixPointerByteEngine::initEmpty()
{
    currentPtr = 0;

    matrixStart = 0;
    matrixPitch = 0;
    matrixSizeX = 0;
    matrixSizeY = 0;

    cachedRow = 0;
}

//================================================================
//
// DebugMatrixPointerByteEngine::initByMatrix
//
//================================================================

void DebugMatrixPointerByteEngine::initByMatrix(DbgptrAddrU matrMemPtr, Space matrMemPitch, Space matrSizeX, Space matrSizeY, const PreconditionsAreValid&)
{
    bool ok = true;

    DbgptrAddrU originalPtr = matrMemPtr;

    //
    // Handle negative pitch:
    // Flip matrix vertically; for access control, this is equivalent.
    // Remember the negative pitch flag.
    //
    // (*) If matrSizeY == 0, we don't care about the pointer as it's inaccessible anyway.
    //

    if (matrMemPitch < 0)
        matrMemPtr += matrMemPitch * (matrSizeY - 1); // (*)

    //
    // Save results
    //

    this->currentPtr = originalPtr;

    this->matrixStart = DbgptrAddrU(matrMemPtr);
    this->matrixPitch = matrMemPitch;
    this->matrixSizeX = matrSizeX;
    this->matrixSizeY = matrSizeY;

    this->cachedRow = originalPtr;

}

//================================================================
//
// Copy
//
//================================================================

sysinline void DebugMatrixPointerByteEngine::copyFrom(const DebugMatrixPointerByteEngine& that)
{
    this->currentPtr = that.currentPtr;

    this->matrixStart = that.matrixStart;
    this->matrixPitch = that.matrixPitch;
    this->matrixSizeX = that.matrixSizeX;
    this->matrixSizeY = that.matrixSizeY;

    this->cachedRow = that.cachedRow;
}

DebugMatrixPointerByteEngine::DebugMatrixPointerByteEngine(const DebugMatrixPointerByteEngine& that)
{
    that.rowUpdate(); // cache row before copy (large performance impact)
    copyFrom(that);
}

DebugMatrixPointerByteEngine& DebugMatrixPointerByteEngine::operator =(const DebugMatrixPointerByteEngine& that)
{
    that.rowUpdate(); // cache row before copy (large performance impact)
    copyFrom(that);
    return *this;
}

//================================================================
//
// DebugMatrixPointerByteEngine::rowUpdate
//
//================================================================

sysinline bool DebugMatrixPointerByteEngine::rowUpdate() const
{
    bool ok = true;

    if_not (currentPtr - cachedRow < matrixSizeX) // is the pointer inside the cached row?
        ok = rowUpdateSlow();

    return ok;
}

//================================================================
//
// DebugMatrixPointerByteEngine::rowUpdateSlow
//
// Uses division.
//
//================================================================

bool DebugMatrixPointerByteEngine::rowUpdateSlow() const
{
    DbgptrAddrU absPitch = abs(matrixPitch);

    DbgptrAddrU currentOfs = currentPtr - matrixStart;

    DbgptrAddrU Y = 0;

    if (absPitch != 0)
        Y = currentOfs / absPitch;

    ////

    bool ok = (Y < matrixSizeY);

    if (ok)
        cachedRow = matrixStart + Y * absPitch;

    return ok;
}

//================================================================
//
// DebugMatrixPointerByteEngine::validateElementSlow
//
//================================================================

void DebugMatrixPointerByteEngine::validateSingleByteSlow() const
{
    rowUpdate();

    if_not ((currentPtr - cachedRow) < matrixSizeX)
        DBGPTR_ERROR_BREAK();
}

//================================================================
//
// DebugMatrixPointerByteEngine::validateArraySlow
//
//================================================================

void DebugMatrixPointerByteEngine::validateArraySlow(bool ok, DbgptrAddrU testSizeX) const
{
    rowUpdate();

    check_flag(testSizeX <= matrixSizeX, ok);
    check_flag(currentPtr - cachedRow <= matrixSizeX - testSizeX, ok);

    if_not (ok)
        DBGPTR_ERROR_BREAK();
}

//================================================================
//
// DebugMatrixPointerByteEngine::validateMatrix
//
//================================================================

void DebugMatrixPointerByteEngine::validateMatrix(bool ok, DbgptrAddrU testSizeX, DbgptrAddrU testSizeY) const
{

    //
    // Compute current (X, Y)
    //

    DbgptrAddrU absPitch = abs(matrixPitch);

    DbgptrAddrU currentOfs = currentPtr - matrixStart;

    DbgptrAddrU Y = 0;

    if (absPitch != 0)
        Y = currentOfs / absPitch;

    DbgptrAddrU X = currentOfs - Y * absPitch;

    //
    // Check X
    //

    check_flag(testSizeX <= matrixSizeX, ok);
    check_flag(X <= matrixSizeX - testSizeX, ok);

    //
    // Check Y
    //

    check_flag(Y < matrixSizeY, ok);

    if (matrixPitch < 0)
        Y = (matrixSizeY - 1) - Y; // if matrixSizeY == 0, it fails on previous check

    check_flag(testSizeY <= matrixSizeY, ok);
    check_flag(Y <= matrixSizeY - testSizeY, ok);

    //
    // Error
    //

    if_not (ok)
        DBGPTR_ERROR_BREAK();
}

//================================================================
//
// DebugMatrixPointerByteEngine::initByArrayPointer
//
//================================================================

void DebugMatrixPointerByteEngine::initByArrayPointer(const DebugArrayPointerByteEngine& that)
{
    this->currentPtr = that.currentPtr;

    this->matrixStart = that.memoryStart;

    this->matrixPitch = Space(that.memorySize);
    this->matrixSizeX = that.memorySize;
    this->matrixSizeY = 1;

    this->cachedRow = that.memoryStart;
}

//================================================================
//
// DebugMatrixPointerByteEngine::exportArrayPointer
//
//================================================================

void DebugMatrixPointerByteEngine::exportArrayPointer(DebugArrayPointerByteEngine& result) const
{
    DbgptrAddrU resultSize = matrixSizeX;

    if_not (rowUpdate())
        resultSize = 0;

    result.setup(cachedRow, (Space) resultSize, currentPtr);
}

//================================================================
//
// Full instantiation
//
//================================================================

template class DebugMatrixPointer<int32>;

//================================================================
//
// dbgptrMatrixPointerTest
//
//================================================================

void dbgptrMatrixPointerTest(int32 nX, int32 nY)
{
    static const Space matrSizeX = 3;
    static const Space matrSizeY = 7;
    static const Space matrMemPitch = matrSizeX + 1;

    int16 myMatrix[matrMemPitch * matrSizeY];
    DebugMatrixPointer<int16> testPtr(myMatrix, matrMemPitch, nX, nY, DbgptrMatrixPreconditions());

    // Simple constructors
    DebugMatrixPointer<int16> ptr0;
    DebugMatrixPointer<int16> copyPtr = testPtr;
    ptr0 = testPtr;

    // Advanced constructors
    DebugMatrixPointer<const int16> copyConstPtr0 = testPtr;
    DebugMatrixPointer<const int16> copyConstPtr1;
    copyConstPtr1 = testPtr;

    // Base funcs

    testPtr += matrMemPitch;
    testPtr.read();
    testPtr.write(0);
    testPtr.modify();
    testPtr -= matrMemPitch;

    //
    testPtr++;
    testPtr--;
    ++testPtr;
    --testPtr;

    //
    testPtr += nX;
    testPtr -= nX;

    //
    *testPtr = 0;
    int32 value = testPtr[0];
    // value = testPtr[-1];
    // value = testPtr[matrSizeX];

    testPtr += nX + 1;
    value = *testPtr;
    testPtr -= nX + 1;

    //
    *(testPtr + 1) = 1;
    value = *(testPtr + 1 - 1);
    int32 diff = (testPtr + 1) - testPtr;

    //
    if (testPtr >= copyPtr)
        {}

    testPtr.validateRange1D(nX);
    // (testPtr-1).validateRange1D(n);
    // testPtr.validateRange1D(n+1);

    testPtr += 1 * matrMemPitch + 2;
    testPtr.validateRange2D(nX-2, nY-1);
    testPtr -= 1 * matrMemPitch + 2;

    int16* unsafe0 = unsafePtr(testPtr, nX);
    int16* unsafe1 = unsafePtr(testPtr, nX, nY);
}
