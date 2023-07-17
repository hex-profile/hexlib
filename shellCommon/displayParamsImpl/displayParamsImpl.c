#include "displayParamsImpl.h"

#include "kits/alternativeVersionKit.h"

//================================================================
//
// DisplayParamsImpl::serialize
//
//================================================================

void DisplayParamsImpl::serialize(const CfgSerializeKit& kit, bool hotkeys, bool& altVersionSteady)
{
    #define HOTSTR(s) (!hotkeys ? STR("") : STR(s))

    ////

    check_flag(theAlternativeVersion.serialize(kit, STR("Alternative Version"), HOTSTR("A")), altVersionSteady);

    theDisplayMode.serialize(kit, STR("Mode"), HOTSTR("Ctrl+D"));

    theVectorMode.serialize
    (
        kit, STR("Vector Mode"),
        {STR("Color"), HOTSTR("Z")},
        {STR("Magnitude"), HOTSTR("X")},
        {STR("X-part"), HOTSTR("C")},
        {STR("Y-part"), HOTSTR("V")}
    );

    displayFactor.serialize(kit, STR("Display Factor"), HOTSTR("Num +"), HOTSTR("Num -"), HOTSTR("Num *"));

    displayFactorDec.serialize(kit, STR("Display Factor Dec"), HOTSTR("Alt+="));
    displayFactorInc.serialize(kit, STR("Display Factor Inc"), HOTSTR("Alt+-"));
    displayFactorReset.serialize(kit, STR("Display Factor Reset"), HOTSTR("Alt+0"));
    displayFactor.feedIncrements(displayFactorDec, displayFactorInc, displayFactorReset);

    displayInterpolation.serialize(kit, STR("Interpolation"), HOTSTR("Alt+I"));
    displayModulation.serialize(kit, STR("Modulation"), HOTSTR("Ctrl+Q"));

    viewIndexSteady = viewIndex.serialize(kit, STR("View Index"), HOTSTR("9"), HOTSTR("0"));
    viewIndexDisplayAll.serialize(kit, STR("View Index: Display All"));

    temporalIndex.serialize(kit, STR("Temporal Index"), HOTSTR(","), HOTSTR("."));
    circularIndex.serialize(kit, STR("Circular Index"), HOTSTR(";"), HOTSTR("'"));
    scaleIndexSteady = scaleIndex.serialize(kit, STR("Scale Index"), HOTSTR("="), HOTSTR("-"));
    stageIndex.serialize(kit, STR("Stage Index"), HOTSTR("["), HOTSTR("]"));
    channelIndex.serialize(kit, STR("Channel Index"), HOTSTR(""), HOTSTR("Alt+C"));

    ////

    #undef HOTSTR
}
