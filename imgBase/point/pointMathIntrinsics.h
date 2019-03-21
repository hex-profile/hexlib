#include "point/point.h"
#include "numbers/mathIntrinsics.h"

//================================================================
//
// Funcs
//
//================================================================

POINT_DEFINE_FUNC1(saturate)

POINT_DEFINE_FUNC3(linerp)
POINT_DEFINE_FUNC3(linearIf)

//================================================================
//
// Division
//
//================================================================

POINT_DEFINE_FUNC1(nativeRecip)
POINT_DEFINE_FUNC1(nativeRecipZero)

POINT_DEFINE_FUNC2(nativeDivide)

//================================================================
//
// sqrt
//
//================================================================

POINT_DEFINE_FUNC1(fastSqrt)
POINT_DEFINE_FUNC1(recipSqrt)

//================================================================
//
// pow2/log2
//
//================================================================

POINT_DEFINE_FUNC1(nativePow2)
POINT_DEFINE_FUNC1(nativeLog2)
