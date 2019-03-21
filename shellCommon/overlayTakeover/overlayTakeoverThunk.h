#pragma once

#include "cfgTools/overlayTakeover.h"

//================================================================
//
// OverlayTakeoverThunk
//
//================================================================
                                    
class OverlayTakeoverThunk : public OverlayTakeover
{

public:

    void setActiveID(uint32 id)
        {targetValue = id;}

    uint32 getActiveID()
        {return targetValue;}

public:

    inline OverlayTakeoverThunk(uint32& targetValue)
        : targetValue(targetValue) {}

private:

    uint32& targetValue;

};
