#include "dbgptrArrayPointer.h"

#include "dbgptr/dbgptrCommonImpl.h"
#include "numbers/int/intBase.h"

//================================================================
//
// DebugArrayPointerByteEngine::errorBreak
//
//================================================================

void DebugArrayPointerByteEngine::errorBreak() const
{
    DBGPTR_ERROR_BREAK();
}

//================================================================
//
// Full instantiation for some type.
//
//================================================================

template class DebugArrayPointer<int32>;

//================================================================
//
// dbgptrArrayPointerTest
//
//================================================================

void dbgptrArrayPointerTest(int32 n)
{
    const Space N = 10;
    int16 myArray[N];
    DebugArrayPointer<int16> testPtr(myArray, N, DbgptrArrayPreconditions());

    // Simple constructors
    DebugArrayPointer<int16> ptr0;
    DebugArrayPointer<int16> copyPtr = testPtr;
    ptr0 = testPtr;

    // Advanced constructors
    DebugArrayPointer<const int16> copyConstPtr0 = testPtr;
    DebugArrayPointer<const int16> copyConstPtr1;
    copyConstPtr1 = testPtr;

    // Base funcs
    testPtr.read();
    testPtr.write(0);
    testPtr.modify();

    //
    testPtr++;
    testPtr--;
    ++testPtr;
    --testPtr;

    //
    testPtr += n;
    testPtr -= n;

    //
    *testPtr = 0;
    int32 value = testPtr[0];
    //value = testPtr[-1];
    //value = testPtr[N];

    //
    *(testPtr + 1) = 1;
    value = *(testPtr - 0);
    int32 diff = (testPtr + 1) - testPtr;

    //
    if (testPtr >= copyPtr)
        {}

    //(testPtr-1).validateRange(N);
    testPtr.validateRange(n);
    // testPtr.validateRange(N+1);

    int16* p = unsafePtr(testPtr, n);

}
