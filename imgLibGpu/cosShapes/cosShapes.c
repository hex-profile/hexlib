#include "cosShapes.h"

#include "mathFuncs/rotationMath.h"

//================================================================
//
// cosShape
//
//================================================================

sysinline float32 cosShape(float32 x)
{
    auto c = circleCCW(0.25f * x).X;

    auto result = c * c;

    if_not (absv(x) < 1)
        result = 0;

    return result;
}

//================================================================
//
// genCosShape
//
//================================================================

void genCosShape(float32* arrPtr, int arrSize)
{
    auto radius = 0.5f * arrSize;
    auto divRadius = nativeRecip(radius);

    auto sum = float32{0};

    for_count(i, arrSize)
    {
        arrPtr[i] = cosShape(((i + 0.5f) - radius) * divRadius);
        sum += arrPtr[i];
    }

    auto divSum = nativeRecip(sum);

    for_count(i, arrSize)
        arrPtr[i] *= divSum;
}
