#pragma once

//================================================================
//
// CharType
//
//================================================================

#if defined(_UNICODE)
    using CharType = wchar_t;
#else
    using CharType = char;
#endif
