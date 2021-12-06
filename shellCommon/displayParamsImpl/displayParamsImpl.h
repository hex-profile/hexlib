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

    void serialize(const CfgSerializeKit& kit, bool& altVersionSteady);

public:

    DisplayMode displayMode() const 
        {return theDisplayMode;}

    VectorMode vectorMode() const 
        {return theVectorMode;}

    bool alternative() const
        {return theAlternativeVersion;}

private:

    friend class DisplayParamsThunk;

private:

    BoolSwitch theAlternativeVersion{false};

private:

    RingSwitch<DisplayMode, DisplayMode::COUNT, DisplayMode::Fullscreen> theDisplayMode;
    MultiSwitch<VectorMode, VectorMode::COUNT, VectorMode::Color> theVectorMode;

private:

    RangeValueControl<float32> displayFactor{1.f/65536.f, 65536.f, 1.f, sqrtf(sqrtf(sqrtf(2))), RangeValueLogscale};
    StandardSignal displayFactorDec;
    StandardSignal displayFactorInc;
    StandardSignal displayFactorReset;

    BoolSwitch displayInterpolation{true};
    BoolSwitch displayModulation{false};

    RangeValueControl<int32> viewIndex{0, 32, 0, 1, RangeValueLinear};
    BoolSwitch viewIndexDisplayAll{false};

    RangeValueControl<int32> temporalIndex{-0x7FFFFFFF-1, +0x7FFFFFFF, 0, 1, RangeValueLinear};
    RangeValueControl<int32> circularIndex{-0x7FFFFFFF-1, +0x7FFFFFFF, 0, 1, RangeValueLinear};
    RangeValueControl<int32> scaleIndex{-32, +128, 0, 1, RangeValueLinear};
    RangeValueControl<int32> stageIndex{-32, +32, 0, 1, RangeValueLinear};
    RangeValueControl<int32> channelIndex{0, 128, 0, 1, RangeValueLinear};

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

        viewIndexThunk(o.viewIndex, o.viewIndexDisplayAll),
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
            o.displayModulation,
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
        o.viewIndex = viewIndexThunk.singleValue();
        o.scaleIndex = scaleIndexThunk;
        o.temporalIndex = temporalIndexThunk;
        o.stageIndex = stageIndexThunk;
    }

public:

    inline auto getKit()
    {
        return kitCombine
        (
            DisplayParamsKit{displayParams},
            AlternativeVersionKit{o.alternative()}
        );
    }

private:

    DisplayParamsImpl& o;

    DisplayedRangeIndexEx viewIndexThunk;
    DisplayedScaleIndex scaleIndexThunk;
    DisplayedCircularIndex circularIndexThunk;
    DisplayedRangeIndex temporalIndexThunk;
    DisplayedRangeIndex stageIndexThunk;
    DisplayedCircularIndex channelIndexThunk;

    DisplayParams const displayParams;

};
