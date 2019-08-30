#pragma once

#include "data/space.h"
#include "errorLog/debugBreak.h"
#include "numbers/float/floatType.h"
#include "point/point.h"

//================================================================
//
// scaleImageSize
//
//================================================================

template <Rounding sizeRounding>
inline Point<Space> scaleImageSize(const Point<Space>& baseSize, const Point<float32>& factor)
{
    Point<float32> resultf = convertFloat32(baseSize) * factor;
    return convertFlex<Point<Space>, ConvertUnchecked, sizeRounding, ConvertNonneg>(resultf);
}

//================================================================
//
// PyramidScale
//
//================================================================

struct PyramidScale
{
    using Hash = uint32;

    virtual float32 operator () (Space level) const =0;

    inline Point<float32> operator () (const Point<Space>& level) const
        {return point(operator()(level.X), operator()(level.Y));}

    virtual Hash hash() const =0;
};

//================================================================
//
// PyramidScaleArray
//
//================================================================

class PyramidScaleArray : public PyramidScale
{

public:

    sysinline float32 operator () (Space level) const
    {
        Space clampedLevel = clampRange(level, -maxLevel, +maxLevel);
        if_not (clampedLevel == level) debugBreak();
        return theScaleArray[maxLevel + clampedLevel];
    }

    sysinline Hash hash() const 
    {
        return theHash;
    }

public:

    PyramidScaleArray() =default;
    PyramidScaleArray(float32 levelFactor) {configure(levelFactor);}
    void configure(float32 levelFactor);
    PyramidScaleArray& operator =(const PyramidScaleArray& that) =default;
    PyramidScaleArray& operator =(const PyramidScale& that);

private:

    static constexpr Space maxLevel = 32;

    uint32 theHash = 0;
    float32 theScaleArray[2 * maxLevel + 1] = {0};

};
