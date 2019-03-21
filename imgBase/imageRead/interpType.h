#pragma once

//================================================================
//
// InterpType
//
// The interpolation mode for UPSAMPLING
//
//================================================================

enum InterpType
{
    // = Box kernel (upsampling)
    INTERP_NONE,
    INTERP_NEAREST = INTERP_NONE,

    // = Tent kernel (upsampling)
    INTERP_LINEAR,

    // = Bicubic kernel (upsampling)
    INTERP_CUBIC,

    //
    INTERP__COUNT
};

//----------------------------------------------------------------

#define INTERP_TYPE_FOREACH(action, extra) \
    \
    action(INTERP_NEAREST, extra) \
    action(INTERP_LINEAR, extra) \
    action(INTERP_CUBIC, extra)
