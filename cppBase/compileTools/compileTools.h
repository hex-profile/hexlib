#pragma once

#include <cstddef>

//================================================================
//
// Basic compile-time tools:
//
// * Language extensions: in lower case and with underscores.
// * Compile-time expressions and types handling
//
//================================================================

//================================================================
//
// sysinline
//
//================================================================

#if defined(__CUDA_ARCH__)
    #define sysinline __device__ __host__ inline
#elif defined(_MSC_VER)
    #define sysinline __forceinline
#else
    #define sysinline inline
#endif

//================================================================
//
// allv
//
// Boolean "and" of all vector components.
//
//================================================================

sysinline bool allv(const bool& value)
    {return value;}

//----------------------------------------------------------------

sysinline bool allv(const void* value)
    {return value != 0;}

//================================================================
//
// anyv
//
// Boolean "or" of all vector components.
//
//================================================================

sysinline bool anyv(const bool& value)
    {return value;}

sysinline bool anyv(const void* value)
    {return value != 0;}

//================================================================
//
// Keywords:
//
// if_not
// while_not
//
//================================================================

#define if_not(expr) \
    if (!(allv(expr)))

#define while_not(expr) \
    while (!(allv(expr)))

//================================================================
//
// check_flag
//
// If the condition is not satisfied, clears the flag.
//
//================================================================

#define check_flag(condition, ok) \
    if (allv(condition)) ; else ok = false

//================================================================
//
// ensure
// ensurev
//
// Checks a condition; if the condition is not true, returns false;
//
// require is for bool functions;
// ensurev is for void functions;
//
//================================================================

#define ensure(condition) \
    if (allv(condition)) ; else return false

#define ensurev(condition) \
    if (allv(condition)) ; else return

#define ensure_ex(condition, returnValue) \
    if (allv(condition)) ; else return (returnValue)

//================================================================
//
// breakBlock
//
// Special construction for local error-checking block.
//
//================================================================

#define breakBlock(flag) \
    bool flag = false; \
    for (; !flag; flag = true)

#define breakBlock_ \
    breakBlock(PREP_PASTE(__bb_, __LINE__))

#define breakFalse \
    break

#define breakTrue \
    continue

#define breakRequire(X) \
    if (allv(X)) ; else breakFalse

//================================================================
//
// soft_cast
//
//================================================================

template <typename Dst, typename Src>
sysinline Dst soft_cast(Src src)
    {return src;}

//================================================================
//
// COMPILE_ASSERT
//
// Compile-time assert.
//
//================================================================

#define COMPILE_ASSERT(X) \
    static_assert(X, COMPILE_ASSERT_STRINGIZE(X))

#define COMPILE_ASSERT_STRINGIZE(X) \
    COMPILE_ASSERT_STRINGIZE2(X)

#define COMPILE_ASSERT_STRINGIZE2(X) \
    #X

//================================================================
//
// COMPILE_ARRAY_SIZE
//
// Compile-time array size.
//
//================================================================

#define COMPILE_ARRAY_SIZE(X) \
    (sizeof(X) / sizeof((X)[0]))

//================================================================
//
// COMPILE_ASSERT_EQUAL_LAYOUT
// COMPILE_ASSERT_FITTING_LAYOUT
//
//================================================================

#define COMPILE_ASSERT_EQUAL_LAYOUT(T1, T2) \
    COMPILE_ASSERT((sizeof(T1) == sizeof(T2) && alignof(T1) == alignof(T2)))

#define COMPILE_ASSERT_FITTING_LAYOUT(T1, T2) \
    COMPILE_ASSERT((sizeof(T1) <= sizeof(T2) && alignof(T2) % alignof(T1) == 0))

//================================================================
//
// recastEqualLayout
//
//================================================================

//================================================================
//
// TYPE_ALIGN
//
//================================================================

template <typename Dst, typename Src>
sysinline Dst& recastEqualLayout(Src& src)
{
    COMPILE_ASSERT(sizeof(Src) == sizeof(Dst));

#if !(defined(_MSC_VER) && (_MSC_VER <= 1900)) // workaround
    COMPILE_ASSERT(alignof(Src) == alignof(Dst));
#endif

    return * (Dst*) &src;
}

//----------------------------------------------------------------

template <typename Dst, typename Src>
sysinline Dst& recastFittingLayout(Src& src)
{
    COMPILE_ASSERT_FITTING_LAYOUT(Dst, Src);
    return * (Dst*) &src;
}

//================================================================
//
// MISSING_FUNCTION_BODY
//
// A body of a template function which has not yet been defined:
// used to provoke compile error at the function call.
//
// Usually, compilers do not compile template function body until instantiation,
// but some of them do.
//
//================================================================

#if defined(__GNUC__)

    #define MISSING_FUNCTION_BODY \
        ;

#else

    #define MISSING_FUNCTION_BODY \
        {static_assert(0, "missing function body");}

#endif

//================================================================
//
// TYPE_SELECT
//
// Conditional selection of a type at compile-time.
//
//================================================================

template <bool cond>
struct TypeSelectHelper_;

template <>
struct TypeSelectHelper_<true>
{
    template <typename T1, typename T2>
    struct Selector {using T = T1;};
};

template <>
struct TypeSelectHelper_<false>
{
    template <typename T1, typename T2>
    struct Selector {using T = T2;};
};

//----------------------------------------------------------------

template <bool cond, typename T1, typename T2>
struct TypeSelect
{
    using T = typename TypeSelectHelper_<cond>::template Selector<T1, T2>::T;
};

//----------------------------------------------------------------

#define TYPE_SELECT(cond, T1, T2) \
    typename TypeSelect<bool(cond), T1, T2>::T

#define TYPE_SELECT_(cond, T1, T2) \
    TypeSelect<bool(cond), T1, T2>::T

//================================================================
//
// TYPE_EQUAL
//
// Checks that two types are identical.
//
//================================================================

template <typename T1, typename T2>
struct TypeEqual
{
    static const bool val = false;
};

template <typename T>
struct TypeEqual<T, T>
{
    static const bool val = true;
};

//----------------------------------------------------------------

#define TYPE_EQUAL(T1, T2) \
    (TypeEqual<T1, T2>::val)

//================================================================
//
// TYPE_CLEANSE(T)
//
// Strips CV modifiers and reference.
//
//================================================================

template <typename Type>
struct TypeCleanse {using T = Type;};

#define TMP_MACRO(Src) \
    template <typename Type> \
    struct TypeCleanse<Src> {using T = Type;};

TMP_MACRO(const Type)
TMP_MACRO(volatile Type)
TMP_MACRO(const volatile Type)

TMP_MACRO(Type&)
TMP_MACRO(const Type&)
TMP_MACRO(volatile Type&)
TMP_MACRO(const volatile Type&)

#undef TMP_MACRO

////

#define TYPE_CLEANSE(Type) \
    typename TypeCleanse<Type>::T

#define TYPE_CLEANSE_(Type) \
    TypeCleanse<Type>::T

//================================================================
//
// ParamType
//
//----------------------------------------------------------------
//
// Makes "const Type&" from "Type".
// If the type is reference, does nothing.
//
//================================================================

template <typename Type>
struct ParamType {using T = const Type&;};

template <typename Type>
struct ParamType<Type&> {using T = Type&;};

//================================================================
//
// INSTANTIATE_FUNC
//
// Instantiates a template function.
//
//================================================================

#define INSTANTIATE_FUNC(expr) \
    INSTANTIATE_FUNC_EX(expr, PREP_EMPTY)

#define INSTANTIATE_FUNC_EX(expr, extraResolver) \
    INSTANTIATE_FUNC_EX2(__LINE__, extraResolver, expr)

#define INSTANTIATE_FUNC_EX2(line, extraResolver, expr) \
    INSTANTIATE_FUNC_EX3(PREP_PASTE3(compileInst_, line, extraResolver), expr)

#define INSTANTIATE_FUNC_EX3(name, expr) \
    namespace {void name(int n) {compileConvertAddr_(&expr);}}

template <typename Type>
static void compileConvertAddr_(Type value)
    {return (void) value;}

//================================================================
//
// MAKE_VARIABLE_USED
//
//================================================================

template <typename Type>
sysinline void compileUseVariableByRef(Type&) {}

template <typename Type>
sysinline void compileUseVariableByValue(Type) {}

//----------------------------------------------------------------

#define MAKE_VARIABLE_USED(var) \
    compileUseVariableByRef(var)

#define MAKE_VARIABLE_USED_BY_VALUE(var) \
    compileUseVariableByValue(var)

//================================================================
//
// COMPILE_MIN
// COMPILE_MAX
//
//================================================================

#define COMPILE_MIN(a, b) \
    (((a) < (b)) ? (a) : (b))

#define COMPILE_MAX(a, b) \
    (((a) > (b)) ? (a) : (b))

#define COMPILE_CLAMP_MIN(X, minValue) \
    COMPILE_MAX(X, minValue)

#define COMPILE_CLAMP_MAX(X, maxValue) \
    COMPILE_MIN(X, maxValue)

#define COMPILE_CLAMP(X, minValue, maxValue) \
    COMPILE_MIN(COMPILE_MAX(X, minValue), maxValue)

//================================================================
//
// COMPILE_IS_POWER2
//
// Checks that the argument is a power of 2.
// Can be used at compile-time.
//
//================================================================

#define COMPILE_IS_POWER2(X) \
    (((X) & ((X) - 1)) == 0)
