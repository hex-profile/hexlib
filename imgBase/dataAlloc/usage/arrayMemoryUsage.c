#include "dataAlloc/arrayMemory.h"
#include "errorLog/errorLog.h"

//================================================================
//
// arrayMemoryUsage
//
//================================================================

KIT_COMBINE2(TestKit, CpuFastAllocKit, ErrorLogKit);

//----------------------------------------------------------------

stdbool arrayMemoryUsage(stdPars(TestKit))
{
    // Construct empty array; no memory allocation performed.
    ArrayMemory<int> m0;

    // Allocate array; check allocation error.
    // If reallocation fails, array will have zero size.
    require(m0.realloc(33, cpuBaseByteAlignment, stdPass));

    // Deallocate memory. Destructor deallocates memory automatically.
    m0.dealloc();

    // Change array size without reallocation; check error.
    // New size should be <= allocated size, otherwise the call fails and size is not changed.
    REQUIRE(m0.resize(13));

    // Get current allocated size.
    REQUIRE(m0.maxSize() == 33);

    // Convert to Array<> implicitly and explicitly (for template arguments).
    Array<int> tmp0 = m0;
    Array<int> tmp1 = m0();

    returnTrue;
}
