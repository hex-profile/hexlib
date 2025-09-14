#pragma once

//================================================================
//
// PREP_PASTE
//
//================================================================

#define PREP_PASTE(X, Y) \
    PREP_PASTE_AUX(X, Y)

#define PREP_PASTE_AUX(X, Y) \
    X##Y

//================================================================
//
// PREP_PASTE_*
//
//================================================================

#define PREP_PASTE2(v0, v1) \
    PREP_PASTE2_AUX(v0, v1)

#define PREP_PASTE2_AUX(v0, v1) \
    v0##v1

//----------------------------------------------------------------

#define PREP_PASTE3(v0, v1, v2) \
    PREP_PASTE3_AUX(v0, v1, v2)

#define PREP_PASTE3_AUX(v0, v1, v2) \
    v0##v1##v2

//----------------------------------------------------------------

#define PREP_PASTE4(v0, v1, v2, v3) \
    PREP_PASTE4_AUX(v0, v1, v2, v3)

#define PREP_PASTE4_AUX(v0, v1, v2, v3) \
    v0##v1##v2##v3

//----------------------------------------------------------------

#define PREP_PASTE5(v0, v1, v2, v3, v4) \
    PREP_PASTE5_AUX(v0, v1, v2, v3, v4)

#define PREP_PASTE5_AUX(v0, v1, v2, v3, v4) \
    v0##v1##v2##v3##v4

//----------------------------------------------------------------

#define PREP_PASTE6(v0, v1, v2, v3, v4, v5) \
    PREP_PASTE6_AUX(v0, v1, v2, v3, v4, v5)

#define PREP_PASTE6_AUX(v0, v1, v2, v3, v4, v5) \
    v0##v1##v2##v3##v4##v5

//================================================================
//
// PREP_PASTE_UNDER*
//
//================================================================

#define PREP_PASTE_UNDER_HELPER(X, Y) \
    X##_##Y

#define PREP_PASTE_UNDER(X, Y) \
    PREP_PASTE_UNDER_HELPER(X, Y)

#define PREP_PASTE_UNDER3(v0, v1, v2) \
    PREP_PASTE_UNDER(v0, PREP_PASTE_UNDER(v1, v2))

#define PREP_PASTE_UNDER4(v0, v1, v2, v3) \
    PREP_PASTE_UNDER(v0, PREP_PASTE_UNDER3(v1, v2, v3))

#define PREP_PASTE_UNDER5(v0, v1, v2, v3, v4) \
    PREP_PASTE_UNDER(v0, PREP_PASTE_UNDER4(v1, v2, v3, v4))

#define PREP_PASTE_UNDER6(v0, v1, v2, v3, v4, v5) \
    PREP_PASTE_UNDER(v0, PREP_PASTE_UNDER5(v1, v2, v3, v4, v5))

//================================================================
//
// PREP_STRINGIZE
//
// Transforms to a string
//
//================================================================

#define PREP_STRINGIZE(X) \
    PREP_STRINGIZE_AUX(X)

#define PREP_STRINGIZE_AUX(X) \
    #X

//================================================================
//
// PREP_EMPTY
//
//================================================================

#define PREP_EMPTY

//================================================================
//
// PREP_COMMA
//
//================================================================

#define PREP_COMMA ,

//================================================================
//
// PREP_PASS*
//
// To use when the preprocessor cannot understand a comma in template brackets <>.
//
// For example,
//
// MY_MACRO(MyClass<T1, T2>)
//
// for the preprocessor means two arguments, use PREP_PASTE2.
//
//================================================================

#define PREP_PASS(v) \
    v

#define PREP_PASS2(v0, v1) \
    v0, v1

#define PREP_PASS3(v0, v1, v2) \
    v0, v1, v2

#define PREP_PASS4(v0, v1, v2, v3) \
    v0, v1, v2, v3
