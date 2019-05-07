#pragma once

#include "compileTools/compileTools.h"

//================================================================
//
// Kit
//
// Kit is a structure containing several parameters.
// When you define a kit, you enumerate parameter names and types:
//
// KIT_CREATE2(MyParams, int, A, int, B);
//
//----------------------------------------------------------------
//
// Kit defines a constructor by all its parameters (in the same order as defined):
//
// KIT_CREATE2(MyParams, int, A, char*, B);
//
// void myFunc()
// {
//     MyParams params(2, "test");
// }
//
//----------------------------------------------------------------
//
// Any kit can be implicitly converted into any other kit,
// assuming that the source kit has all field names found in the destination kit
// and the fields are convertible.
//
// For this purpose, a kit defines inline template constructor taking arbitrary structure,
// constructing all its fields by fields with the same names in the source structure.
//
// To avoid full copy on kit conversions (when passing a kit to a function having different kit type),
// you can explicitly inherit a kit from an other kit, in this case compiler only passes a pointer.
//
//================================================================

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Shared tools
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// KitFieldTag
//
// Used to record the presence of field name in a kit.
//
//================================================================

template <typename Tag>
struct KitFieldTag {};

//================================================================
//
// KIT__TYPENAME_YES
// KIT__TYPENAME_NO
//
//================================================================

#define KIT__TYPENAME_YES() \
    typename

#define KIT__TYPENAME_NO()

//================================================================
//
// KitIsConvertible
//
// Is source type pointer is convertible to destination type pointer?
//
//================================================================

template <typename Src, typename Dst>
struct KitIsConvertible
{
    using FalseType = char;
    struct TrueType {char data[2];};
    COMPILE_ASSERT(sizeof(FalseType) != sizeof(TrueType));

    static FalseType check(...);
    static TrueType check(const Dst*);

    static Src src;

    static const bool val = (sizeof(check(&src)) == sizeof(TrueType));
};

//================================================================
//
// KitFieldSelector
//
// Selects from two pointers by compile-time condition.
// Used for kit replace functionality.
//
//================================================================

template <bool condition>
struct KitFieldSelector;

template <>
struct KitFieldSelector<false>
{
    template <typename T0, typename T1>
    static sysinline const T0* func(const T0* v0, const T1* v1)
        {return v0;}
};

template <>
struct KitFieldSelector<true>
{
    template <typename T0, typename T1>
    static sysinline const T1* func(const T0* v0, const T1* v1)
        {return v1;}
};

//================================================================
//
// KitReplacer
//
//================================================================

template <typename OldKit, typename NewKit, typename Tag>
struct KitReplacer : public KitFieldSelector< KitIsConvertible<NewKit, KitFieldTag<Tag> >::val> {};

//================================================================
//
// KitReplaceConstructor
// KitCombineConstructor
//
// Used to disambiguate constructors
//
//================================================================

enum KitReplaceConstructor {};
enum KitCombineConstructor {};

//================================================================
//
// kitReplace
//
//================================================================

template <typename BaseKit, typename NewKit>
sysinline BaseKit kitReplace(const BaseKit& baseKit, const NewKit& newKit)
{
    return BaseKit(baseKit, KitReplaceConstructor(), newKit);
}

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// KIT_CREATE
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

#define KIT_CREATE0(Kit) \
    \
    struct Kit \
    { \
        sysinline Kit(const Kit& that) \
        { \
        } \
        \
        sysinline Kit \
        ( \
        ) \
        { \
        } \
        \
        template <typename OtherKit> \
        sysinline Kit(const OtherKit& otherKit) \
        { \
        } \
        \
        template <typename OldKit, typename NewKit> \
        sysinline Kit(const OldKit& oldKit, KitReplaceConstructor, const NewKit& newKit) \
        { \
        } \
    }
    
//----------------------------------------------------------------

# include "kitCreate.inl"

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// KIT_COMBINE
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

template <typename... Types>
struct KitCombine
    :
    public Types...
{
    template <typename OtherKit>
    sysinline KitCombine(const OtherKit& otherKit)
        :
        Types(otherKit)...
    {
    }
   
    template <typename OldKit, typename NewKit>
    sysinline KitCombine(const OldKit& oldKit, KitReplaceConstructor, const NewKit& newKit)
        :
        Types(oldKit, KitReplaceConstructor(), newKit)...
    {
    }
   
    sysinline KitCombine
    (
        const Types&... otherTypes,
        KitCombineConstructor
    )
        :
        Types(otherTypes)...
    {
    }
};

//----------------------------------------------------------------

template <typename... Types>
sysinline auto kitCombine(const Types&... values)
    {return KitCombine<Types...>(values..., KitCombineConstructor());}

//----------------------------------------------------------------

#define KIT_COMBINE1(Kit, T0) \
    using Kit = KitCombine<T0>

#define KIT_COMBINE2(Kit, T0, T1) \
    using Kit = KitCombine<T0, T1>

#define KIT_COMBINE3(Kit, T0, T1, T2) \
    using Kit = KitCombine<T0, T1, T2>

#define KIT_COMBINE4(Kit, T0, T1, T2, T3) \
    using Kit = KitCombine<T0, T1, T2, T3>

#define KIT_COMBINE5(Kit, T0, T1, T2, T3, T4) \
    using Kit = KitCombine<T0, T1, T2, T3, T4>

#define KIT_COMBINE6(Kit, T0, T1, T2, T3, T4, T5) \
    using Kit = KitCombine<T0, T1, T2, T3, T4, T5>

#define KIT_COMBINE7(Kit, T0, T1, T2, T3, T4, T5, T6) \
    using Kit = KitCombine<T0, T1, T2, T3, T4, T5, T6>

#define KIT_COMBINE8(Kit, T0, T1, T2, T3, T4, T5, T6, T7) \
    using Kit = KitCombine<T0, T1, T2, T3, T4, T5, T6, T7>

#define KIT_COMBINE9(Kit, T0, T1, T2, T3, T4, T5, T6, T7, T8) \
    using Kit = KitCombine<T0, T1, T2, T3, T4, T5, T6, T7, T8>

#define KIT_COMBINE10(Kit, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9) \
    using Kit = KitCombine<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>

#define KIT_COMBINE11(Kit, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10) \
    using Kit = KitCombine<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>

#define KIT_COMBINE12(Kit, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11) \
    using Kit = KitCombine<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11>
