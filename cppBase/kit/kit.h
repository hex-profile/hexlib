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
// Kit_FieldTag
//
// Used to record the presence of field name in a kit.
//
//================================================================

template <typename Tag>
struct Kit_FieldTag {};

//================================================================
//
// Kit_IsConvertible
//
// Is source type pointer is convertible to destination type pointer?
//
//================================================================

template <typename Src, typename Dst>
struct Kit_IsConvertible
{
    using FalseType = char;
    struct TrueType {char data[2];};
    COMPILE_ASSERT(sizeof(FalseType) != sizeof(TrueType));

    static FalseType check(...);
    static TrueType check(const Dst*);

    static Src* makeSrc();

    static constexpr bool value = (sizeof(check(makeSrc())) == sizeof(TrueType));
};

//================================================================
//
// Kit_ReplaceSelector
//
// Selects from two pointers by compile-time condition.
// Used for kit replace functionality.
//
//================================================================

template <bool condition>
struct Kit_ReplaceSelector;

template <>
struct Kit_ReplaceSelector<false>
{
    template <typename T0, typename T1>
    static sysinline const T0* func(const T0* v0, const T1* v1)
        {return v0;}
};

template <>
struct Kit_ReplaceSelector<true>
{
    template <typename T0, typename T1>
    static sysinline const T1* func(const T0* v0, const T1* v1)
        {return v1;}
};

//================================================================
//
// Kit_Replacer
//
//================================================================

template <typename OldKit, typename NewKit, typename Tag>
struct Kit_Replacer : public Kit_ReplaceSelector<Kit_IsConvertible<NewKit, Kit_FieldTag<Tag>>::value> {};

//================================================================
//
// Kit_ReplaceConstructor
// Kit_CombineConstructor
//
// Used to disambiguate constructors
//
//================================================================

enum class Kit_ReplaceConstructor {};
enum class Kit_CombineConstructor {};

//================================================================
//
// kitReplace
//
//================================================================

template <typename BaseKit, typename NewKit>
sysinline BaseKit kitReplace(const BaseKit& baseKit, const NewKit& newKit)
{
    return BaseKit(baseKit, Kit_ReplaceConstructor(), newKit);
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

template <typename Type>
struct Kit_ValueReader
{
    static sysinline Type& func(Type& value)
        {return value;}

    static sysinline const Type& func(const Type& value)
        {return value;}
};

//----------------------------------------------------------------

#define KIT_CREATE(Kit, Field, name) \
    \
    struct Kit \
        : \
        Kit_FieldTag<struct name##_Tag> \
    { \
        \
        Field name; \
        \
        template <typename Kit> \
        struct Kit_FieldReader \
        { \
            static sysinline Field func(const Kit& kit) \
                {return kit.name;} \
        }; \
        \
        template <typename Type> \
        struct Kit_ReaderSelector \
        { \
            static constexpr bool isKit = Kit_IsConvertible<Type, Kit_FieldTag<name##_Tag>>::value; \
            using T = TYPE_SELECT(isKit, Kit_FieldReader<Type>, Kit_ValueReader<Type>); \
        }; \
        \
        template <typename Type> \
        sysinline Kit(Type& value) \
            : name(Kit_ReaderSelector<Type>::T::func(value)) {} \
        \
        template <typename Type> \
        sysinline Kit(const Type& value) \
            : name(Kit_ReaderSelector<Type>::T::func(value)) {} \
        \
        template <typename OldKit, typename NewKit> \
        sysinline Kit(const OldKit& oldKit, Kit_ReplaceConstructor, const NewKit& newKit) \
            : \
            name(Kit_Replacer<OldKit, NewKit, name##_Tag>::func(&oldKit, &newKit)->name) \
        { \
        } \
    }

//----------------------------------------------------------------

# include "kitCreate.inl"

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// KitCombine
// kitCombine
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
    sysinline KitCombine(const OldKit& oldKit, Kit_ReplaceConstructor, const NewKit& newKit)
        :
        Types(oldKit, Kit_ReplaceConstructor(), newKit)...
    {
    }

    sysinline KitCombine
    (
        Kit_CombineConstructor,
        const Types&... otherTypes
    )
        :
        Types(otherTypes)...
    {
    }
};

//----------------------------------------------------------------

template <typename... Types>
sysinline auto kitCombine(const Types&... values)
    {return KitCombine<Types...>(Kit_CombineConstructor(), values...);}
