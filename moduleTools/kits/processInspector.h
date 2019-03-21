#pragma once

//================================================================
//
// ProcessInspector
//
//================================================================

struct ProcessInspector
{
    virtual void operator()(bool& steadyProcessing) =0;
};
