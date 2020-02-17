#pragma once

#include "cfgTools/boolSwitch.h"
#include "imageConsole/imageConsoleModes.h"
#include "cfgTools/rangeValueControl.h"
#include "cfgTools/multiSwitch.h"
#include "kits/displayParamsKit.h"
#include "kits/alternativeVersionKit.h"

//================================================================
//
// DisplayParamsImpl
//
//================================================================

class DisplayParamsImpl
{

public:

    void serialize(const CfgSerializeKit& kit, bool& prepParamsSteady);

public:

    DisplayMode displayMode() const 
        {return theDisplayMode;}

    VectorMode vectorMode() const 
        {return theVectorMode;}

    bool alternativeVersion() const
        {return theAlternativeVersion;}

private:

    friend class DisplayParamsThunk;

private:

    BoolSwitch<false> theAlternativeVersion;

private:

    RingSwitch<DisplayMode, DisplayMode::COUNT, DisplayMode::Fullscreen> theDisplayMode;
    MultiSwitch<VectorMode, VectorMode::COUNT, VectorMode::Color> theVectorMode;

private:

    RangeValueControl<float32> displayFactor{1.f/65536.f, 65536.f, 1.f, sqrtf(sqrtf(sqrtf(2))), RangeValueLogscale};
    BoolSwitch<true> displayInterpolation;

    RangeValueControl<int32> viewIndex{-0x7FFFFFFF-1, +0x7FFFFFFF, 0, 1, RangeValueLinear};
    RangeValueControl<int32> temporalIndex{-0x7FFFFFFF-1, +0x7FFFFFFF, 0, 1, RangeValueLinear};
    RangeValueControl<int32> circularIndex{-0x7FFFFFFF-1, +0x7FFFFFFF, 0, 1, RangeValueLinear};
    RangeValueControl<int32> scaleIndex{0, 0x7F, 0, 1, RangeValueLinear};
    RangeValueControl<int32> stageIndex{-0x7FFFFFFF-1, +0x7FFFFFFF, 0, 1, RangeValueLinear};
    RangeValueControl<int32> channelIndex{-0x7FFFFFFF-1, +0x7FFFFFFF, 0, 1, RangeValueLinear};

};

//================================================================
//
// DisplayParamsThunk
//
//================================================================

class DisplayParamsThunk
{

public:

    DisplayParamsThunk(const Point<Space>& screenSize, DisplayParamsImpl& o)
        :
        o(o),

        viewIndexThunk(o.viewIndex),
        scaleIndexThunk(o.scaleIndex),
        circularIndexThunk(o.circularIndex),
        temporalIndexThunk(o.temporalIndex),
        stageIndexThunk(o.stageIndex),
        channelIndexThunk(o.channelIndex),

        displayParams
        {
            o.theDisplayMode == DisplayMode::Fullscreen,
            o.displayFactor,
            screenSize, 
            o.displayInterpolation,
            viewIndexThunk, 
            temporalIndexThunk, 
            scaleIndexThunk,
            circularIndexThunk,
            stageIndexThunk,
            channelIndexThunk
        }
    {
    }

    ~DisplayParamsThunk()
    {
        o.viewIndex = viewIndexThunk;
        o.scaleIndex = scaleIndexThunk;
        o.temporalIndex = temporalIndexThunk;
        o.stageIndex = stageIndexThunk;
    }

public:

    inline auto kit()
    {
        return kitCombine
        (
            DisplayParamsKit{displayParams},
            AlternativeVersionKit{o.alternativeVersion()}
        );
    }

private:

    DisplayParamsImpl& o;

    DisplayedRangeIndex viewIndexThunk;
    DisplayedRangeIndex scaleIndexThunk;
    DisplayedCircularIndex circularIndexThunk;
    DisplayedRangeIndex temporalIndexThunk;
    DisplayedRangeIndex stageIndexThunk;
    DisplayedCircularIndex channelIndexThunk;

    DisplayParams const displayParams;

};
