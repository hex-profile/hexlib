#include "pyramidScale.h"

//================================================================
//
// PyramidScaleArray::configure
//
//================================================================

void PyramidScaleArray::configure(float32 baseFactor)
{
    uint32 hash = 0;

    for (Space i = -maxScale; i <= +maxScale; ++i)
    {
        float32 factor = powf(baseFactor, float32(i));
        scaleArray[maxScale + i] = factor;

        hash ^= recastEqualLayout<uint32>(factor);
    }

    currentHash = hash;
    scaleIsUniform = true;
}

//================================================================
//
// PyramidScaleArray::operator =
//
//================================================================

PyramidScaleArray& PyramidScaleArray::operator =(const PyramidScale& that)
{
    uint32 hash = 0;

    for (Space i = -maxScale; i <= +maxScale; ++i)
    {
        float32 factor = that(i);
        scaleArray[maxScale + i] = factor;
        hash ^= recastEqualLayout<uint32>(factor);
    }

    currentHash = hash;
    scaleIsUniform = that.isUniform();

    return *this;
}
