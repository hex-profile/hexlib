#pragma once

//================================================================
//
// CharType
//
//================================================================

#ifndef HEXLIB_CHARTYPE
#define HEXLIB_CHARTYPE

#if defined(_UNICODE)
    using CharType = wchar_t;
#else
    using CharType = char;
#endif

#endif // HEXLIB_CHARTYPE
