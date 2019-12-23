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

    virtual bool isUniform() const =0;

    virtual float32 operator () (Space level) const =0;

    inline Point<float32> operator () (const Point<Space>& level) const
        {return point(operator()(level.X), operator()(level.Y));}

    virtual Hash hash() const =0;
};

//================================================================
//
// PyramidScaleNull
//
//================================================================

struct PyramidScaleNull : public PyramidScale
{
    bool isUniform() const
        {return true;}

    float32 operator () (Space level) const
        {return 1;}

    Hash hash() const
        {return 0;}
};

//================================================================
//
// PyramidScaleArray
//
//================================================================

class PyramidScaleArray : public PyramidScale
{

public:

    sysinline bool isUniform() const
    {
        return scaleIsUniform;
    }

    sysinline float32 operator () (Space level) const
    {
        Space clampedLevel = clampRange(level, -maxLevel, +maxLevel);
        if_not (clampedLevel == level) debugBreak();
        return scaleArray[maxLevel + clampedLevel];
    }

    sysinline Hash hash() const 
    {
        return currentHash;
    }

public:

    PyramidScaleArray() =default;

    PyramidScaleArray(float32 levelFactor) 
        {configure(levelFactor);}

    void configure(float32 levelFactor);

    PyramidScaleArray& operator =(const PyramidScaleArray& that) =default;

    PyramidScaleArray& operator =(const PyramidScale& that);

private:

    static constexpr Space maxLevel = 32;

    uint32 currentHash = 0;

    bool scaleIsUniform = false;

    float32 scaleArray[2 * maxLevel + 1] = {0};

};
