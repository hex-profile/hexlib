#include "arrayBase.h"

#include "types/intTypes.h"

//================================================================
//
// require
//
//================================================================

#define require(condition) \
    if (condition) ; else return false

//================================================================
//
// arrayBaseCompileTest
//
//================================================================

bool arrayBaseCompileTest(const uint8_t* srcPtr, Space srcSize)
{
    // Construct empty array.
    ArrayBase<int> intArray;

    // Convert an array to a read-only array.
    ArrayBase<const int> constArray = intArray;
    ArrayBase<const int> anotherConstArray = makeConst(intArray);

    // Construct array from details: ptr and size.
    ArrayBase<const uint8_t> example(srcPtr, srcSize);

    // Setup array from details: ptr and size.
    example.assign(srcPtr, srcSize);

    // Make the array empty:
    example.assignNull();

    // Access array details (decomposing array is better way):
    require(example.ptr() != 0);
    require(example.size() != 0);

    // Check that array has non-zero size
    require(hasData(example));

    return true;
}
