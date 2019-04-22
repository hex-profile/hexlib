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

    inline operator int32 ()
        {return index;}

    inline int32 operator () (int32 minVal, int32 maxVal)
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
// DisplaySide
//
//================================================================

enum DisplaySide {DisplayOld, DisplayNew, DisplaySide_Count};

//----------------------------------------------------------------

template <typename Type, typename Kit>
inline const Type& displaySide(const Kit& kit, const Type& oldValue, const Type& newValue)
    {return kit.displayedViewIndex(0, DisplaySide_Count-1) == DisplayOld ? oldValue : newValue;}

template <typename Kit>
inline DisplaySide displaySide(const Kit& kit)
    {return DisplaySide(kit.displayedViewIndex(0, DisplaySide_Count-1));}

//================================================================
//
// DisplayMethod
//
//================================================================

enum DisplayMethod {DISPLAY_FULLSCREEN, DISPLAY_CENTERED, DISPLAY_ORIGINAL, DISPLAY_METHOD_COUNT};

//================================================================
//
// DisplayParamsKit
//
//================================================================

KIT_CREATE6(
    DisplayParamsKit,
    Point<Space>, displayFrameSize,
    DisplayedRangeIndex&, displayedViewIndex,
    DisplayedRangeIndex&, displayedTemporalIndex,
    DisplayedRangeIndex&, displayedScaleIndex,
    const DisplayedCircularIndex&, displayedCircularIndex,
    DisplayMethod, displayMethod
);
