#pragma once

#include "data/space.h"
#include "errorLog/debugBreak.h"
#include "numbers/float/floatType.h"
#include "point/point.h"

//================================================================
//
// PyramidScale
//
// Defines power table like (baseFactor ^ scale),
// where baseFactor >= 1.
//
//================================================================

struct PyramidScale
{
    using Hash = uint32;

    virtual bool isUniform() const =0;

    virtual float32 operator () (Space scale) const =0;

    inline Point<float32> operator () (const Point<Space>& scale) const
        {return point(operator()(scale.X), operator()(scale.Y));}

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

    float32 operator () (Space scale) const
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

    sysinline float32 operator () (Space scale) const
    {
        Space clampedScale = clampRange(scale, -maxScale, +maxScale);
        if_not (clampedScale == scale) debugBreak();
        return scaleArray[maxScale + clampedScale];
    }

    sysinline Hash hash() const
    {
        return currentHash;
    }

public:

    PyramidScaleArray() =default;

    PyramidScaleArray(float32 baseFactor)
        {configure(baseFactor);}

    void configure(float32 baseFactor);

    PyramidScaleArray& operator =(const PyramidScaleArray& that) =default;

    PyramidScaleArray& operator =(const PyramidScale& that);

private:

    static constexpr Space maxScale = 32;

    uint32 currentHash = 0;

    bool scaleIsUniform = false;

    float32 scaleArray[2 * maxScale + 1] = {0};

};

//================================================================
//
// PyramidConfigOptions
//
//----------------------------------------------------------------
//
// Main pyramid parameters:
//
// * Base image size in pixels.
// * Pyramid levels and layers.
// * Pyramid scale (power table).
// * Image size rounding mode {DOWN, UP or NEAREST}.
//
// If no advanced parameters are used, the 0th pyramid level size
// will have size exactly equal to the base image size.
//
// Advanced parameters:
//
// * Resolution factor: Subsampling factor, if we subsample 4X, the factor is 0.25f.
// Each level's image size is multiplied by the factor,
// the factor is applied equally to all pyramid levels.
//
// * Base level: Sometimes we want to have the base image size
// not on 0th level, but on some other level, for example, doing upsampling.
// The specified level will have the base image size (before resolution factor application).
//
// * Extra border: An integer number of pixels to be added to each level's image size after rounding.
// For example, if we want 5 extra pixels on all sides, the extra border is 10.
//
//
//================================================================

class PyramidConfigOptions
{

public:

    auto& resolutionFactor(const Point<float32>& value)
        {theResolutionFactor = value; return *this;}

    auto& resolutionFactor(float32 value)
        {theResolutionFactor = point(value); return *this;}

    auto resolutionFactor() const
        {return theResolutionFactor;}

public:

    auto& baseLevel(const Point<Space>& value)
        {theBaseLevel = value; return *this;}

    auto& baseLevel(Space value)
        {theBaseLevel = point(value); return *this;}

    auto baseLevel() const
        {return theBaseLevel;}

public:

    auto& extraBorder(const Point<Space>& value)
        {theExtraBorder = value; return *this;}

    auto extraBorder() const
        {return theExtraBorder;}

private:

    Point<float32> theResolutionFactor = point<float32>(1);
    Point<Space> theBaseLevel = point<Space>(0);
    Point<Space> theExtraBorder = point<Space>(0);

};

//================================================================
//
// PyramidLevelSize
//
// Computes pyramid level image size in pixels.
//
//================================================================

class PyramidLevelSize
{

public:

    inline PyramidLevelSize()
    {
        convertFunc[RoundDown] = convertFlex<Point<Space>, ConvertUnchecked, RoundDown, ConvertNonneg>;
        convertFunc[RoundUp] = convertFlex<Point<Space>, ConvertUnchecked, RoundUp, ConvertNonneg>;
        convertFunc[RoundNearest] = convertFlex<Point<Space>, ConvertUnchecked, RoundNearest, ConvertNonneg>;
    }

public:

    inline Point<Space> operator ()
    (
        const Point<Space>& baseImageSize,
        int level,
        const PyramidScale& pyramidScale,
        Rounding sizeRounding,
        const PyramidConfigOptions& options = {}
    )
    {
        auto resultf = convertFloat32(baseImageSize) * options.resolutionFactor() * pyramidScale(-level + options.baseLevel()) + options.extraBorder();
        return convertFunc[sizeRounding](resultf);
    }

private:

    typedef Point<Space> ConvertFunc(const Point<float32>& value);
    ConvertFunc* convertFunc[Round_COUNT] = {nullptr};

};
