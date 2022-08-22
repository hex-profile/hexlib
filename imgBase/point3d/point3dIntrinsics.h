#pragma once

#include "point3d/point3d.h"
#include "numbers/mathIntrinsics.h"

//================================================================
//
// Funcs
//
//================================================================

POINT3D_DEFINE_FUNC1(saturate)

POINT3D_DEFINE_FUNC3(linerp)
POINT3D_DEFINE_FUNC3(linearIf)

//================================================================
//
// Division
//
//================================================================

POINT3D_DEFINE_FUNC1(fastRecip)
POINT3D_DEFINE_FUNC1(fastRecipZero)
POINT3D_DEFINE_FUNC2(fastDivide)

//================================================================
//
// sqrt
//
//================================================================

POINT3D_DEFINE_FUNC1(fastSqrt)
POINT3D_DEFINE_FUNC1(recipSqrt)

//================================================================
//
// pow2/log2
//
//================================================================

POINT3D_DEFINE_FUNC1(fastPow2)
POINT3D_DEFINE_FUNC1(fastLog2)
