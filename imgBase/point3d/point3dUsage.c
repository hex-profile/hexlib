#include "point3d/point3d.h"

#include "numbers/int/intType.h"
#include "numbers/float/floatType.h"
#include "stdFunc/stdFunc.h"

//================================================================
//
// point3dUsage
//
//================================================================

stdbool point3dUsage()
{
    // Type traits: signed/unsigned.
    COMPILE_ASSERT(TYPE_IS_SIGNED(const Point3D<int>));
    COMPILE_ASSERT(TYPE_EQUAL(TYPE_MAKE_UNSIGNED_(Point3D<int>), Point3D<unsigned>));

    // Type traits: controlled support.
    COMPILE_ASSERT(TYPE_IS_CONTROLLED(const Point3D<float>));

    // Type traits: min and max.
    Point3D<int> minPoint = typeMin<Point3D<int>>();

    // Default constructor: uninitialized
    Point3D<int> uninitializedValue;
    uninitializedValue.X = 0;

    // Create point by components.
    Point3D<int> A = point3D(16, 32, 2);
    Point3D<int> B = point3D(16, 32, 2);

    // Convert point->point, scalar-guided.
    Point3D<float> c0 = convertNearest<float>(A);

    // Convert point->point, vector-guided.
    Point3D<float> c1 = convertNearest<Point3D<float>>(A);

    // Convert scalar->point.
    Point3D<float> c2 = convertNearest<Point3D<float>>(1);

    // Convert with success flag.
    Point3D<int> intResult;
    Point3D<bool> intSuccess = convertNearest(c2, intResult);
    intSuccess = convertNearest(25, intResult);

    // Make zero
    Point3D<float32> makeZero = zeroOf<Point3D<float32>>();

    // Generate NAN
    Point3D<float> pointNan = nanOf<const Point3D<float>&>();

    // Is definite?
    Point3D<bool> pointDef = def(pointNan);

    // Unary operations.
    Point3D<int> testUnary = -A;

    // Arithmetic binary operations.
    Point3D<int> C = A + B;
    Point3D<int> D = 2 * A;

    // Assignment arithmetic operations.
    A += 1;
    C &= D;

    // Vector comparisons.
    Point3D<bool> Q = (A == B);
    Point3D<bool> R = (1 <= B);

    // Vector bool operations.
    Point3D<bool> Z = Q && R;
    Point3D<bool> W = Q || R;
    Point3D<bool> V = !Z;

    // Reduction of Point3D<bool> to bool.
    require(allv(A >= 0));
    if (anyv(A < 0)) returnFalse;

    // Min, max, clampRange functions.
    Point3D<int> t1 = minv(A, B);
    Point3D<int> t2 = maxv(A, B);
    Point3D<int> t3 = clampMin(A, 0);
    Point3D<int> t4 = clampMax(A, 10);
    // Point3D<int> t5 = clampRange(A, 0, 10);

    returnTrue;
}
