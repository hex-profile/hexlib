#include "movement3d.h"

#include "numbers/float/floatType.h"

//================================================================
//
//
//
//================================================================

void test(const Movement3D<float32>& movement, const Point3D<float32>& vec)
{
    auto res = apply(combine(movement, inverse(movement)), vec);

}
