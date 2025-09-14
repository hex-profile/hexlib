#include "displayWaitController.h"

#include "userOutput/printMsgEx.h"

//================================================================
//
// DisplayWaitController::serialize
//
//================================================================

void DisplayWaitController::serialize(const ModuleSerializeKit& kit)
{
    waitActive.serialize(kit, STR("Active"));
    displayTime.serialize(kit, STR("Show Frame Time"));
    targetDelayMs.serialize(kit, STR("Target Delay in Milliseconds"));
}

//================================================================
//
// DisplayWaitController::waitForDisplayTime
//
//================================================================

void DisplayWaitController::waitForDisplayTime(stdPars(Kit))
{
    float32 targetDelay = targetDelayMs * 1e-3f;

    if (waitActive && displayTime)
        printMsgL(kit, STR("Render: Target delay = %0 ms"), fltf(targetDelay * 1000, 1));

    //----------------------------------------------------------------
    //
    // First time: no wait
    //
    //----------------------------------------------------------------

    if_not (lastOutputInit)
    {
        lastOutput = kit.timer.moment();
        lastOutputInit = true;
        return;
    }

    //----------------------------------------------------------------
    //
    // Wait
    //
    //----------------------------------------------------------------

    float32 workTime = kit.timer.diff(lastOutput, kit.timer.moment());

    for (; ;)
    {
        float32 elapsedTime = kit.timer.diff(lastOutput, kit.timer.moment());
        float32 additionalDelay = clampMin(targetDelay - elapsedTime, 0.f);

        if_not (additionalDelay > 0)
            break;

        //
        // Waste some CPU cycles on all cores, don't give up the priority
        //

        volatile float32 value = 3.14f;

        pragmaOmp(parallel)
        {
            for_count (i, 8192)
                value = sqrtf(value + 1);
        }
    }

    //----------------------------------------------------------------
    //
    // Output
    //
    //----------------------------------------------------------------

    TimeMoment outputMoment = kit.timer.moment();

    //----------------------------------------------------------------
    //
    // Update last output moment
    //
    //----------------------------------------------------------------

    float32 realDelay = kit.timer.diff(lastOutput, outputMoment);

    if (displayTime)
        printMsgL(kit, STR("Display: Real delay = %0 ms (work %1 ms)"), fltf(realDelay * 1000, 1), fltf(workTime * 1000, 1));

    ////

    lastOutput = outputMoment;
}
