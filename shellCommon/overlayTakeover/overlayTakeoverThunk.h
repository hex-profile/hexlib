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

    void setActiveID(size_t id)
        {targetValue = id;}

    size_t getActiveID()
        {return targetValue;}

public:

    inline OverlayTakeoverThunk(uint32& targetValue)
        : targetValue(targetValue) {}

private:

    uint32& targetValue;

};
