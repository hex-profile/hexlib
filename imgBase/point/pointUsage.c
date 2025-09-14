#include "point/point.h"

#include "numbers/int/intType.h"
#include "numbers/float/floatType.h"
#include "stdFunc/stdFunc.h"

//================================================================
//
// pointUsage
//
//================================================================

void pointUsage()
{
    stdTraceRoot;

    // Type traits: signed/unsigned.
    COMPILE_ASSERT(TYPE_IS_SIGNED(const Point<int>));
    COMPILE_ASSERT(TYPE_EQUAL(TYPE_MAKE_UNSIGNED_(Point<int>), Point<unsigned>));

    // Type traits: controlled support.
    COMPILE_ASSERT(TYPE_IS_CONTROLLED(const Point<float>));

    // Type traits: min and max.
    Point<int> minPoint = typeMin<Point<int>>();

    // Default constructor: uninitialized
    Point<int> uninitializedValue;
    uninitializedValue.X = 0;

    // Create point by components. The scalar version creates a point
    // with both components having the same value.
    Point<int> A = point(16, 32);
    Point<int> B = point(2); // B = {2, 2}

    // Convert point->point, scalar-guided.
    Point<float> c0 = convertNearest<float>(A);

    // Convert point->point, vector-guided.
    Point<float> c1 = convertNearest<Point<float>>(B);

    // Convert scalar->point.
    Point<float> c2 = convertNearest<Point<float>>(1);

    // Convert with success flag.
    Point<int> intResult;
    Point<bool> intSuccess = convertNearest(c2, intResult);
    intSuccess = convertNearest(25, intResult);

    // Make zero
    Point<float32> makeZero = zeroOf<Point<float32>>();

    // Generate NAN
    Point<float> pointNan = nanOf<const Point<float>&>();

    // Is definite?
    Point<bool> pointDef = def(pointNan);

    // Unary operations.
    Point<int> testUnary = -A;

    // Arithmetic binary operations.
    Point<int> C = A + B;
    Point<int> D = 2 * A;

    // Assignment arithmetic operations.
    A += 1;
    C &= D;

    // Vector comparisons.
    Point<bool> Q = (A == B);
    Point<bool> R = (1 <= B);

    // Vector bool operations.
    Point<bool> Z = Q && R;
    Point<bool> W = Q || R;
    Point<bool> V = !Z;

    // Reduction of Point<bool> to bool.
    require(allv(A >= 0));
    if (anyv(A < 0)) returnFalse;

    // Min, max, clampRange functions.
    Point<int> t1 = minv(A, B);
    Point<int> t2 = maxv(A, B);
    Point<int> t3 = clampMin(A, 0);
    Point<int> t4 = clampMax(A, 10);
    Point<int> t5 = clampRange(A, 0, 10);
}
