#include "profilerStruct.h"

#include "errorLog/debugBreak.h"

//================================================================
//
// profilerUpdateDeviceTreeTime
//
//================================================================

void profilerUpdateDeviceTreeTime(ProfilerNode& node)
{
    float32 childrenTime = 0;
    float32 childrenOverhead = 0;

    for (ProfilerNode* p = node.lastChild; p != 0; p = p->prevBrother)
    {
        profilerUpdateDeviceTreeTime(*p);

        childrenTime += p->deviceTotalTime;
        childrenOverhead += p->deviceTotalOverheadTime;
    }

    node.deviceTotalTime = node.deviceNodeTime + childrenTime;

    ////

    float32 totalOverheadTime = node.deviceNodeOverheadTime + childrenOverhead;

    //
    // If it is leaf, check whether on AVERAGE the overhead
    // is likely to be hidden by the kernel own execution,
    // or potentially not.
    //

    if_not (node.lastChild)
    {
        if (node.deviceNodeOverheadTime <= 0.8f * node.deviceNodeTime)
            totalOverheadTime = 0; // assume it goes parallel
    }

    node.deviceTotalOverheadTime = totalOverheadTime;

}
