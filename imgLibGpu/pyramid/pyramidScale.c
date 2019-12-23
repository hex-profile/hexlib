#include "pyramidScale.h"

//================================================================
//
// PyramidScaleArray::configure
//
//================================================================

void PyramidScaleArray::configure(float32 levelFactor)
{
    uint32 hash = 0;

    for (Space i = -maxLevel; i <= +maxLevel; ++i)
    {
        float32 factor = powf(levelFactor, float32(i));
        scaleArray[maxLevel + i] = factor;

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

    for (Space i = -maxLevel; i <= +maxLevel; ++i)
    {
        float32 factor = that(i);
        scaleArray[maxLevel + i] = factor;
        hash ^= recastEqualLayout<uint32>(factor);
    }

    currentHash = hash;
    scaleIsUniform = that.isUniform();

    return *this;
}
