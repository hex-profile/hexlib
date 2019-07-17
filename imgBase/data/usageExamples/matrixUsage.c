#include "stdFunc/stdFunc.h"
#include "errorLog/errorLog.h"
#include "data/matrix.h"

//================================================================
//
// matrixUsage
//
//================================================================

stdbool matrixUsage(const Matrix<const uint8>& src, stdPars(ErrorLogKit))
{
    MATRIX_EXPOSE(src);
    const uint8* srcMemPtrUnsafe = unsafePtr(srcMemPtr, srcSizeX, srcSizeY);

    // Create an empty matrix.
    Matrix<int> intMatrix;
    Matrix<int> anotherEmptyMatrix = 0;

    // Convert a matrix to a read-only matrix.
    Matrix<const int> constIntMatrix = intMatrix;
    Matrix<const int> anotherConstMatrix = makeConst(intMatrix);

    // Construct matrix from details: ptr, pitch and size.
    Matrix<const uint8> example(srcMemPtrUnsafe, srcMemPitch, srcSizeX, srcSizeY);

    // Setup matrix from details: ptr, pitch and size.
    example.assign(srcMemPtrUnsafe, srcMemPitch, srcSizeX, srcSizeY);

    // Make the matrix empty.
    example.assignNull();

    // Access matrix details (decomposing matrix is better way):
    REQUIRE(example.memPtr() != 0);
    REQUIRE(example.memPitch() != 0);
    REQUIRE(example.sizeX() != 0);
    REQUIRE(example.sizeY() != 0);

    // Decompose a matrix to detail variables:
    MATRIX_EXPOSE(example);
    REQUIRE(exampleMemPtr != 0);
    REQUIRE(exampleMemPitch != 0);
    REQUIRE(exampleSizeX != 0);
    REQUIRE(exampleSizeY != 0);

    // Access some element in a decomposed matrix.
    // The macro uses multiplication. No X/Y range checking performed!
    int value = MATRIX_ELEMENT(example, 0, 0);

    // Example element loop (not optimized):
    uint32 sum = 0;

    for (Space Y = 0; Y < exampleSizeY; ++Y)
        for (Space X = 0; X < exampleSizeX; ++X)
            sum += exampleMemPtr[X + Y * exampleMemPitch];

    // Save rectangular area [10, 30) as a new matrix using
    // "subs" (submatrix by size) function. Check that no clipping occured.
    Matrix<const uint8> tmp1;
    REQUIRE(example.subs(point(10), point(20), tmp1));

    // Save rectangular area [10, 30) as a new matrix using
    // "subr" (submatrix by rect) function. Check that no clipping occured.
    Matrix<const uint8> tmp2;
    REQUIRE(example.subr(point(10), point(30), tmp2));

    // Remove const qualifier from element (avoid using it!)
    Matrix<uint8> tmp3 = recastElement<uint8>(tmp2);

    // Check that matrices have equal size.
    REQUIRE(equalSize(example, tmp1, tmp2));
    REQUIRE(equalSize(tmp1, tmp2, point(20)));

    // Check that a matrix has non-zero size
    REQUIRE(hasData(example));
    REQUIRE(hasData(example.size()));

    returnTrue;
}
