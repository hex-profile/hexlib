#pragma once

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
// CharType
//
//================================================================

#ifndef HEXLIB_CHARTYPE
#define HEXLIB_CHARTYPE

using CharType = CHARTYPE_SELECT(char, wchar_t);

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
