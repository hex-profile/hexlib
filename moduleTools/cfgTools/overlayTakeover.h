#pragma once

#include "kit/kit.h"

//================================================================
//
// OverlayTakeover
//
// Only one user ID can be active.
//
//================================================================

struct OverlayTakeover
{
    using ID = size_t;
    static constexpr ID nullID = 0;

    virtual void setActiveID(const ID& id) const =0;
    virtual ID getActiveID() const =0;
};

//================================================================
//
// OverlayTakeoverKit
//
//================================================================

KIT_CREATE1(OverlayTakeoverKit, const OverlayTakeover&, overlayTakeover);
