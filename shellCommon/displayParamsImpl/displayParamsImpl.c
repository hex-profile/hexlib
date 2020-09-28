#include "displayParamsImpl.h"
#include "kits/alternativeVersionKit.h"

//================================================================
//
// DisplayParamsImpl::serialize
//
//================================================================

void DisplayParamsImpl::serialize(const CfgSerializeKit& kit, bool& prepParamsSteady)
{
    check_flag(theAlternativeVersion.serialize(kit, STR("Alternative Version"), STR("a")), prepParamsSteady);
    theAlternativeVersionPrintAlways.serialize(kit, STR("Alternative Version: Print Always"));

    theDisplayMode.serialize(kit, STR("Mode"), STR("Ctrl+D"));

    theVectorMode.serialize
    (
        kit, STR("Vector Mode"),
        {STR("Color"), STR("Z")},
        {STR("Magnitude"), STR("X")},
        {STR("X"), STR("C")},
        {STR("Y"), STR("V")}
    );

    displayFactor.serialize(kit, STR("Display Factor"), STR("Num +"), STR("Num -"), STR("Num *"));
    displayInterpolation.serialize(kit, STR("Interpolation"), STR("Alt+I"));

    viewIndex.serialize(kit, STR("View Index"), STR("9"), STR("0"));
    viewIndexDisplayAll.serialize(kit, STR("View Index: Display All"));

    temporalIndex.serialize(kit, STR("Temporal Index"), STR(","), STR("."));
    circularIndex.serialize(kit, STR("Circular Index"), STR(";"), STR("'"));
    scaleIndex.serialize(kit, STR("Scale Index"), STR("="), STR("-"));
    stageIndex.serialize(kit, STR("Stage Index"), STR("["), STR("]"));
    channelIndex.serialize(kit, STR("Channel Index"), STR(""), STR("Alt+C"));
}