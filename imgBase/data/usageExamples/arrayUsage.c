#include "errorLog/errorLog.h"
#include "stdFunc/stdFunc.h"
#include "data/array.h"

//================================================================
//
// arrayUsage
//
//================================================================

stdbool arrayUsage(stdPars(ErrorLogKit))
{
    stdBegin;

    const int srcSize = 10;
    uint8 srcArray[srcSize];
    uint8* srcPtr = srcArray;

    // Construct empty array.
    Array<int> intArray;

    // Convert an array to a read-only array.
    Array<const int> constArray = intArray;
    Array<const int> anotherConstArray = makeConst(intArray);

    // Construct array from details: ptr and size.
    Array<const uint8> example(srcPtr, srcSize);

    // Setup array from details: ptr and size.
    example.assign(srcPtr, srcSize);

    // Make the array empty:
    example.assignNull();

    // Access array details (decomposing array is better way):
    REQUIRE(example.ptr() != 0);
    REQUIRE(example.size() != 0);

    // Decompose a array to detail variables:
    ARRAY_EXPOSE(example);
    REQUIRE(examplePtr != 0);
    REQUIRE(exampleSize != 0);

    // Example element loop for a decomposed array:
    uint32 sum = 0;

    for (Space i = 0; i < exampleSize; ++i)
        sum += examplePtr[i];

    // Save element range [10, 30) as a new array using
    // "subs" (subarray by size) function. Check that no clipping occured.
    Array<const uint8> tmp1;
    REQUIRE(example.subs(10, 20, tmp1));

    // Save element range [10, 30) as a new array using
    // "subr" (subarray by rect) function. Check that no clipping occured.
    Array<const uint8> tmp2;
    REQUIRE(example.subr(10, 30, tmp2));

    // Removing const qualifier from elements (avoid this):
    Array<uint8> tmp3 = recastToNonConst(tmp2);

    // Check that arrays have equal size.
    REQUIRE(equalSize(example, tmp1, tmp2));
    REQUIRE(equalSize(tmp1, tmp2, 20));

    // Check that array has non-zero size
    REQUIRE(hasData(example));
    REQUIRE(hasData(example.size()));

    stdEnd;
}
