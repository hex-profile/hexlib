#pragma once

//================================================================
//
// GCC_BUGFIX
//
// For GCC bugs workarounds.
//
//================================================================

#if defined(__GNUC__)
    #define GCC_BUGFIX_ONLY(code) code
#else
    #define GCC_BUGFIX_ONLY(code)
#endif

//================================================================
//
// GCC_BUGFIX_PRAGMA
//
//================================================================

#define GCC_BUGFIX_PRAGMA(text) \
    GCC_BUGFIX_ONLY(_Pragma(text))
