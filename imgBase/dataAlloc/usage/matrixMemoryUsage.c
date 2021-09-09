#include "dataAlloc/matrixMemory.h"
#include "stdFunc/stdFunc.h"
#include "errorLog/errorLog.h"
#include "dataAlloc/memoryAllocator.h"

//================================================================
//
// matrixMemoryUsage
//
//================================================================

using TestKit = KitCombine<CpuFastAllocKit, ErrorLogKit>;

//----------------------------------------------------------------

stdbool matrixMemoryUsage(stdPars(TestKit))
{
    // Construct empty matrix; no memory allocation performed.
    MatrixMemory<int> m0;

    // Allocate matrix; check allocation error.
    // If reallocation fails, matrix will have zero size.
    // Destructor deallocates memory automatically.
    require(m0.realloc(point(33, 17), cpuBaseByteAlignment, cpuRowByteAlignment, stdPass));

    // Change matrix layout without reallocation; check error.
    // New size should be <= allocated size, otherwise the call fails and the layout is not changed.
    REQUIRE(m0.resize(point(13, 15)));

    // Get current allocated size.
    REQUIRE(m0.maxSizeX() == 13);
    REQUIRE(m0.maxSizeY() == 15);
    REQUIRE(m0.maxSize() == point(13, 15));

    // Reallocate matrix base aligned to 512 bytes and pitch aligned to 32 bytes;
    // The pitch alignment will be used across "resize" calls until the next "realloc";
    require(m0.realloc(point(333, 111), 512, 32, stdPass));
    REQUIRE(m0.resize(129, 15));

    // Convert to Matrix<> implicitly and explicitly (for template arguments).
    Matrix<int> tmp0 = m0;
    Matrix<const int> tmp1 = m0;
    Matrix<const int> tmp2 = m0();

    returnTrue;
}
