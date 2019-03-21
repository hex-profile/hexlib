#pragma once

//================================================================
//
// BorderMode
//
//================================================================

enum BorderMode
{
    // Reads 0 outside image
    BORDER_ZERO,

    // Reads last pixel outside image
    BORDER_CLAMP,

    // Reads mirrored image. Only one reflection is performed, beyond this area zero is returned.
    BORDER_MIRROR,

    // Reads repeated image.
    BORDER_WRAP,

    BORDER_MODE_COUNT
};

//----------------------------------------------------------------

#define BORDER_MODE_FOREACH(action, extra) \
    action(BORDER_ZERO, extra) \
    action(BORDER_CLAMP, extra) \
    action(BORDER_MIRROR, extra) \
    action(BORDER_WRAP, extra)
