#pragma once

//================================================================
//
// InterpType
//
// The interpolation mode for UPSAMPLING only.
//
//================================================================

enum InterpType
{
    // = Box kernel.
    INTERP_NONE,
    INTERP_NEAREST = INTERP_NONE,

    // = Tent kernel.
    INTERP_LINEAR,

    // = Bicubic kernel.
    INTERP_CUBIC,

    // = B-spline cubic kernel (for use with Michael Unser prefilering).
    INTERP_CUBIC_BSPLINE,

    //
    INTERP__COUNT
};

//----------------------------------------------------------------

#define INTERP_TYPE_FOREACH(action, extra) \
    \
    action(INTERP_NEAREST, extra) \
    action(INTERP_LINEAR, extra) \
    action(INTERP_CUBIC, extra)
