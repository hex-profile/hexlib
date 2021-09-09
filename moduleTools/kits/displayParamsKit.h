#pragma once

#include "kit/kit.h"
#include "data/space.h"
#include "charType/charArray.h"
#include "point/point.h"
#include "numbers/float/floatBase.h"
#include "numbers/int/intBase.h"

//================================================================
//
// DisplayedCircularIndex
//
//================================================================

class DisplayedCircularIndex
{

public:

    inline DisplayedCircularIndex(int32 value)
    {
        nonnegValue = value;
        inverted = false;

        if (value < 0)
        {
            nonnegValue = -(value + 1); // >= 0
            inverted = true;
        }

        this->nonnegValue = nonnegValue;
    }

public:

    inline int32 operator () (int32 period) const
    {
        if (period < 1) period = 1;

        int32 result = nonnegValue % period;

        if (inverted)
            result = (period - 1 - result);

        return result;
    }

private:

    int32 nonnegValue;
    bool inverted;

};

//================================================================
//
// DisplayedRangeIndex
//
//================================================================

class DisplayedRangeIndex
{

public:

    inline DisplayedRangeIndex(int32 index)
        : index(index) {}

    inline operator int32 () const
        {return index;}

    inline auto operator () (int32 minVal, int32 maxVal)
    {
        if (index < minVal) index = minVal;
        if (index > maxVal) index = maxVal;
        return index;
    }

private:

    int32 index;

};

//================================================================
//
// DisplayedScaleIndex
//
//================================================================

class DisplayedScaleIndex
{

public:

    inline DisplayedScaleIndex(int32 index)
        : index(index) {}

    inline operator int32 () const
        {return index;}

    inline auto operator () (int32 minVal, int32 maxVal)
    {
        if (index < minVal) index = minVal;
        if (index > maxVal) index = maxVal;
        return index;
    }

private:

    int32 index;

};

//================================================================
//
// DisplayedIndexIterator
//
//================================================================

class DisplayedIndexIterator
{

public:

    explicit DisplayedIndexIterator(int32 value)
        : value{value} {}

    int32 operator *() const
        {return value;}

    DisplayedIndexIterator& operator ++()
        {++value; return *this;}

    bool operator ==(const DisplayedIndexIterator& that) const
        {return this->value == that.value;}

    bool operator !=(const DisplayedIndexIterator& that) const
        {return this->value != that.value;}

private:

    int32 value;

};

//================================================================
//
// DisplayedRange
//
//================================================================

class DisplayedRange
{

public:

    explicit DisplayedRange(int32 mainValue, int32 rangeOrg, int32 rangeEnd)
        : mainValue(mainValue), rangeOrg(rangeOrg), rangeEnd(rangeEnd) {}

public:

    int32 singleValue() const {return mainValue;}

public:

    friend bool operator ==(const DisplayedRange& that, int32 index)
        {return that.rangeOrg <= index && index < that.rangeEnd;}

    friend bool operator ==(int32 index, const DisplayedRange& that)
        {return that.rangeOrg <= index && index < that.rangeEnd;}

public:

    auto begin() const {return DisplayedIndexIterator{rangeOrg};}
    auto end() const {return DisplayedIndexIterator{rangeEnd};}

private:

    int32 mainValue;
    int32 rangeOrg;
    int32 rangeEnd;

};

//================================================================
//
// DisplayedRangeIndexEx
//
//================================================================

class DisplayedRangeIndexEx
{

public:

    inline DisplayedRangeIndexEx(int32 index, bool displayAll)
        : index(index), displayAll(displayAll) {}

    inline auto singleValue() const
        {return index;}

    inline DisplayedRange operator ()(int32 minVal, int32 maxVal)
    {
        if (index < minVal) index = minVal;
        if (index > maxVal) index = maxVal;

        return displayAll ?
            DisplayedRange{index, minVal, maxVal + 1} :
            DisplayedRange{index, index, index + 1};
    }

    inline auto singleValue(int32 minVal, int32 maxVal)
    {
        if (index < minVal) index = minVal;
        if (index > maxVal) index = maxVal;
        return index;
    }

private:

    int32 index;
    bool displayAll;

};

//================================================================
//
// DisplaySide
//
//================================================================

enum DisplaySide {DisplayOld, DisplayNew, DisplaySide_Count};

//----------------------------------------------------------------

template <typename Type, typename Kit>
inline const Type& displaySide(const Kit& kit, const Type& oldValue, const Type& newValue)
    {return kit.display.viewIndex.singleValue(0, DisplaySide_Count-1) == DisplayOld ? oldValue : newValue;}

template <typename Kit>
inline DisplaySide displaySide(const Kit& kit)
    {return DisplaySide(kit.display.viewIndex.singleValue(0, DisplaySide_Count-1));}

//================================================================
//
// DisplayParams
//
//================================================================

struct DisplayParams
{
    // User request to display upsampled full-screen image.
    bool const fullscreen;

    // Displayed data range multiplier.
    float32 const factor;

    // Input frame size used to as full-screen target resolution.
    Point<Space> screenSize;

    // User desire to see interpolated data.
    bool const interpolation;

    // User desire to see modulated data.
    bool const modulation;

    // Abstract view index or image side.
    DisplayedRangeIndexEx& viewIndex;

    // Abstract temporal index.
    DisplayedRangeIndex& temporalIndex;

    // Abstract scale index.
    DisplayedScaleIndex& scaleIndex;

    // Abstract circular index.
    const DisplayedCircularIndex& circularIndex;

    // Abstract stage index.
    DisplayedRangeIndex& stageIndex;

    // Abstract channel index.
    const DisplayedCircularIndex& channelIndex;

};

//================================================================
//
// DisplayParamsKit
//
//================================================================

KIT_CREATE(DisplayParamsKit, const DisplayParams&, display);
