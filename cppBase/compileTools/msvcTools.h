#pragma once

//================================================================
//
// MSVC_SELECT
//
// For Microsoft bugs workarounds.
//
//================================================================

#if defined(_MSC_VER)
    #define MSVC_SELECT(yes, no) yes
#else
    #define MSVC_SELECT(yes, no) no
#endif

//----------------------------------------------------------------

#define MSVC_ONLY(code) \
    MSVC_SELECT(code, PREP_EMPTY)

#define MSVC_EXCLUDE(code) \
    MSVC_SELECT(PREP_EMPTY, code)

