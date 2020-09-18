#pragma once

#include "extLib/types/charType.h"

//================================================================
//
// CHARTYPE_SELECT
//
// Selection between wchar_t and char.
//
//================================================================

#if defined(_UNICODE)

    #define CHARTYPE_SELECT(notUnicode, unicode) \
        unicode

#else

    #define CHARTYPE_SELECT(notUnicode, unicode) \
        notUnicode

#endif

//================================================================
//
// CT
//
//================================================================

#define CT(x) \
    CT_IMPL(x)

#define CT_IMPL(x) \
    CHARTYPE_SELECT(x, L ## x)
