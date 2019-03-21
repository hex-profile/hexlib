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
// one can explicitly inherit a kit from the other kit: in this case compiler passes a pointer.
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

#define KIT__CREATE0(Kit, typenameWord) \
    \
    struct Kit \
    { \
        \
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
        \
    }

//----------------------------------------------------------------

#define KIT_CREATE0(Kit) \
    KIT__CREATE0(Kit, KIT__TYPENAME_NO)

#define KIT_CREATE0_(Kit) \
    KIT__CREATE0(Kit, KIT__TYPENAME_YES)

//----------------------------------------------------------------

#define KIT__CREATE1(Kit, Type0, name0, typenameWord) \
    \
    struct Kit \
        : \
        KitFieldTag<struct name0##_Tag> \
    { \
        \
        Type0 name0; \
        \
        sysinline Kit(const Kit& that) \
            : \
            name0(that.name0) \
        { \
        } \
        \
        sysinline Kit \
        ( \
            typenameWord() ParamType< Type0 >::T name0, \
            int = 0 \
        ) \
            : \
            name0(name0) \
        { \
        } \
        \
        template <typename OtherKit> \
        sysinline Kit(const OtherKit& otherKit) \
            : \
            name0(otherKit.name0) \
        { \
        } \
        \
        template <typename OldKit, typename NewKit> \
        sysinline Kit(const OldKit& oldKit, KitReplaceConstructor, const NewKit& newKit) \
            : \
            name0(KitReplacer<OldKit, NewKit, name0##_Tag>::func(&oldKit, &newKit)->name0) \
        { \
        } \
        \
    }

//----------------------------------------------------------------

#define KIT_CREATE1(Kit, Type0, name0) \
    KIT__CREATE1(Kit, Type0, name0, KIT__TYPENAME_NO)

#define KIT_CREATE1_(Kit, Type0, name0) \
    KIT__CREATE1(Kit, Type0, name0, KIT__TYPENAME_YES)

//----------------------------------------------------------------

#define KIT__CREATE2(Kit, Type0, name0, Type1, name1, typenameWord) \
    \
    struct Kit \
        : \
        KitFieldTag<struct name0##_Tag>, \
        KitFieldTag<struct name1##_Tag> \
    { \
        \
        Type0 name0; \
        Type1 name1; \
        \
        sysinline Kit(const Kit& that) \
            : \
            name0(that.name0), \
            name1(that.name1) \
        { \
        } \
        \
        sysinline Kit \
        ( \
            typenameWord() ParamType< Type0 >::T name0, \
            typenameWord() ParamType< Type1 >::T name1 \
        ) \
            : \
            name0(name0), \
            name1(name1) \
        { \
        } \
        \
        template <typename OtherKit> \
        sysinline Kit(const OtherKit& otherKit) \
            : \
            name0(otherKit.name0), \
            name1(otherKit.name1) \
        { \
        } \
        \
        template <typename OldKit, typename NewKit> \
        sysinline Kit(const OldKit& oldKit, KitReplaceConstructor, const NewKit& newKit) \
            : \
            name0(KitReplacer<OldKit, NewKit, name0##_Tag>::func(&oldKit, &newKit)->name0), \
            name1(KitReplacer<OldKit, NewKit, name1##_Tag>::func(&oldKit, &newKit)->name1) \
        { \
        } \
        \
    }

//----------------------------------------------------------------

#define KIT_CREATE2(Kit, Type0, name0, Type1, name1) \
    KIT__CREATE2(Kit, Type0, name0, Type1, name1, KIT__TYPENAME_NO)

#define KIT_CREATE2_(Kit, Type0, name0, Type1, name1) \
    KIT__CREATE2(Kit, Type0, name0, Type1, name1, KIT__TYPENAME_YES)

//----------------------------------------------------------------

#define KIT__CREATE3(Kit, Type0, name0, Type1, name1, Type2, name2, typenameWord) \
    \
    struct Kit \
        : \
        KitFieldTag<struct name0##_Tag>, \
        KitFieldTag<struct name1##_Tag>, \
        KitFieldTag<struct name2##_Tag> \
    { \
        \
        Type0 name0; \
        Type1 name1; \
        Type2 name2; \
        \
        sysinline Kit(const Kit& that) \
            : \
            name0(that.name0), \
            name1(that.name1), \
            name2(that.name2) \
        { \
        } \
        \
        sysinline Kit \
        ( \
            typenameWord() ParamType< Type0 >::T name0, \
            typenameWord() ParamType< Type1 >::T name1, \
            typenameWord() ParamType< Type2 >::T name2 \
        ) \
            : \
            name0(name0), \
            name1(name1), \
            name2(name2) \
        { \
        } \
        \
        template <typename OtherKit> \
        sysinline Kit(const OtherKit& otherKit) \
            : \
            name0(otherKit.name0), \
            name1(otherKit.name1), \
            name2(otherKit.name2) \
        { \
        } \
        \
        template <typename OldKit, typename NewKit> \
        sysinline Kit(const OldKit& oldKit, KitReplaceConstructor, const NewKit& newKit) \
            : \
            name0(KitReplacer<OldKit, NewKit, name0##_Tag>::func(&oldKit, &newKit)->name0), \
            name1(KitReplacer<OldKit, NewKit, name1##_Tag>::func(&oldKit, &newKit)->name1), \
            name2(KitReplacer<OldKit, NewKit, name2##_Tag>::func(&oldKit, &newKit)->name2) \
        { \
        } \
        \
    }

//----------------------------------------------------------------

#define KIT_CREATE3(Kit, Type0, name0, Type1, name1, Type2, name2) \
    KIT__CREATE3(Kit, Type0, name0, Type1, name1, Type2, name2, KIT__TYPENAME_NO)

#define KIT_CREATE3_(Kit, Type0, name0, Type1, name1, Type2, name2) \
    KIT__CREATE3(Kit, Type0, name0, Type1, name1, Type2, name2, KIT__TYPENAME_YES)

//----------------------------------------------------------------

#define KIT__CREATE4(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, typenameWord) \
    \
    struct Kit \
        : \
        KitFieldTag<struct name0##_Tag>, \
        KitFieldTag<struct name1##_Tag>, \
        KitFieldTag<struct name2##_Tag>, \
        KitFieldTag<struct name3##_Tag> \
    { \
        \
        Type0 name0; \
        Type1 name1; \
        Type2 name2; \
        Type3 name3; \
        \
        sysinline Kit(const Kit& that) \
            : \
            name0(that.name0), \
            name1(that.name1), \
            name2(that.name2), \
            name3(that.name3) \
        { \
        } \
        \
        sysinline Kit \
        ( \
            typenameWord() ParamType< Type0 >::T name0, \
            typenameWord() ParamType< Type1 >::T name1, \
            typenameWord() ParamType< Type2 >::T name2, \
            typenameWord() ParamType< Type3 >::T name3 \
        ) \
            : \
            name0(name0), \
            name1(name1), \
            name2(name2), \
            name3(name3) \
        { \
        } \
        \
        template <typename OtherKit> \
        sysinline Kit(const OtherKit& otherKit) \
            : \
            name0(otherKit.name0), \
            name1(otherKit.name1), \
            name2(otherKit.name2), \
            name3(otherKit.name3) \
        { \
        } \
        \
        template <typename OldKit, typename NewKit> \
        sysinline Kit(const OldKit& oldKit, KitReplaceConstructor, const NewKit& newKit) \
            : \
            name0(KitReplacer<OldKit, NewKit, name0##_Tag>::func(&oldKit, &newKit)->name0), \
            name1(KitReplacer<OldKit, NewKit, name1##_Tag>::func(&oldKit, &newKit)->name1), \
            name2(KitReplacer<OldKit, NewKit, name2##_Tag>::func(&oldKit, &newKit)->name2), \
            name3(KitReplacer<OldKit, NewKit, name3##_Tag>::func(&oldKit, &newKit)->name3) \
        { \
        } \
        \
    }

//----------------------------------------------------------------

#define KIT_CREATE4(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3) \
    KIT__CREATE4(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, KIT__TYPENAME_NO)

#define KIT_CREATE4_(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3) \
    KIT__CREATE4(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, KIT__TYPENAME_YES)

//----------------------------------------------------------------

#define KIT__CREATE5(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, typenameWord) \
    \
    struct Kit \
        : \
        KitFieldTag<struct name0##_Tag>, \
        KitFieldTag<struct name1##_Tag>, \
        KitFieldTag<struct name2##_Tag>, \
        KitFieldTag<struct name3##_Tag>, \
        KitFieldTag<struct name4##_Tag> \
    { \
        \
        Type0 name0; \
        Type1 name1; \
        Type2 name2; \
        Type3 name3; \
        Type4 name4; \
        \
        sysinline Kit(const Kit& that) \
            : \
            name0(that.name0), \
            name1(that.name1), \
            name2(that.name2), \
            name3(that.name3), \
            name4(that.name4) \
        { \
        } \
        \
        sysinline Kit \
        ( \
            typenameWord() ParamType< Type0 >::T name0, \
            typenameWord() ParamType< Type1 >::T name1, \
            typenameWord() ParamType< Type2 >::T name2, \
            typenameWord() ParamType< Type3 >::T name3, \
            typenameWord() ParamType< Type4 >::T name4 \
        ) \
            : \
            name0(name0), \
            name1(name1), \
            name2(name2), \
            name3(name3), \
            name4(name4) \
        { \
        } \
        \
        template <typename OtherKit> \
        sysinline Kit(const OtherKit& otherKit) \
            : \
            name0(otherKit.name0), \
            name1(otherKit.name1), \
            name2(otherKit.name2), \
            name3(otherKit.name3), \
            name4(otherKit.name4) \
        { \
        } \
        \
        template <typename OldKit, typename NewKit> \
        sysinline Kit(const OldKit& oldKit, KitReplaceConstructor, const NewKit& newKit) \
            : \
            name0(KitReplacer<OldKit, NewKit, name0##_Tag>::func(&oldKit, &newKit)->name0), \
            name1(KitReplacer<OldKit, NewKit, name1##_Tag>::func(&oldKit, &newKit)->name1), \
            name2(KitReplacer<OldKit, NewKit, name2##_Tag>::func(&oldKit, &newKit)->name2), \
            name3(KitReplacer<OldKit, NewKit, name3##_Tag>::func(&oldKit, &newKit)->name3), \
            name4(KitReplacer<OldKit, NewKit, name4##_Tag>::func(&oldKit, &newKit)->name4) \
        { \
        } \
        \
    }

//----------------------------------------------------------------

#define KIT_CREATE5(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4) \
    KIT__CREATE5(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, KIT__TYPENAME_NO)

#define KIT_CREATE5_(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4) \
    KIT__CREATE5(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, KIT__TYPENAME_YES)

//----------------------------------------------------------------

#define KIT__CREATE6(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, typenameWord) \
    \
    struct Kit \
        : \
        KitFieldTag<struct name0##_Tag>, \
        KitFieldTag<struct name1##_Tag>, \
        KitFieldTag<struct name2##_Tag>, \
        KitFieldTag<struct name3##_Tag>, \
        KitFieldTag<struct name4##_Tag>, \
        KitFieldTag<struct name5##_Tag> \
    { \
        \
        Type0 name0; \
        Type1 name1; \
        Type2 name2; \
        Type3 name3; \
        Type4 name4; \
        Type5 name5; \
        \
        sysinline Kit(const Kit& that) \
            : \
            name0(that.name0), \
            name1(that.name1), \
            name2(that.name2), \
            name3(that.name3), \
            name4(that.name4), \
            name5(that.name5) \
        { \
        } \
        \
        sysinline Kit \
        ( \
            typenameWord() ParamType< Type0 >::T name0, \
            typenameWord() ParamType< Type1 >::T name1, \
            typenameWord() ParamType< Type2 >::T name2, \
            typenameWord() ParamType< Type3 >::T name3, \
            typenameWord() ParamType< Type4 >::T name4, \
            typenameWord() ParamType< Type5 >::T name5 \
        ) \
            : \
            name0(name0), \
            name1(name1), \
            name2(name2), \
            name3(name3), \
            name4(name4), \
            name5(name5) \
        { \
        } \
        \
        template <typename OtherKit> \
        sysinline Kit(const OtherKit& otherKit) \
            : \
            name0(otherKit.name0), \
            name1(otherKit.name1), \
            name2(otherKit.name2), \
            name3(otherKit.name3), \
            name4(otherKit.name4), \
            name5(otherKit.name5) \
        { \
        } \
        \
        template <typename OldKit, typename NewKit> \
        sysinline Kit(const OldKit& oldKit, KitReplaceConstructor, const NewKit& newKit) \
            : \
            name0(KitReplacer<OldKit, NewKit, name0##_Tag>::func(&oldKit, &newKit)->name0), \
            name1(KitReplacer<OldKit, NewKit, name1##_Tag>::func(&oldKit, &newKit)->name1), \
            name2(KitReplacer<OldKit, NewKit, name2##_Tag>::func(&oldKit, &newKit)->name2), \
            name3(KitReplacer<OldKit, NewKit, name3##_Tag>::func(&oldKit, &newKit)->name3), \
            name4(KitReplacer<OldKit, NewKit, name4##_Tag>::func(&oldKit, &newKit)->name4), \
            name5(KitReplacer<OldKit, NewKit, name5##_Tag>::func(&oldKit, &newKit)->name5) \
        { \
        } \
        \
    }

//----------------------------------------------------------------

#define KIT_CREATE6(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5) \
    KIT__CREATE6(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, KIT__TYPENAME_NO)

#define KIT_CREATE6_(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5) \
    KIT__CREATE6(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, KIT__TYPENAME_YES)

//----------------------------------------------------------------

#define KIT__CREATE7(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, typenameWord) \
    \
    struct Kit \
        : \
        KitFieldTag<struct name0##_Tag>, \
        KitFieldTag<struct name1##_Tag>, \
        KitFieldTag<struct name2##_Tag>, \
        KitFieldTag<struct name3##_Tag>, \
        KitFieldTag<struct name4##_Tag>, \
        KitFieldTag<struct name5##_Tag>, \
        KitFieldTag<struct name6##_Tag> \
    { \
        \
        Type0 name0; \
        Type1 name1; \
        Type2 name2; \
        Type3 name3; \
        Type4 name4; \
        Type5 name5; \
        Type6 name6; \
        \
        sysinline Kit(const Kit& that) \
            : \
            name0(that.name0), \
            name1(that.name1), \
            name2(that.name2), \
            name3(that.name3), \
            name4(that.name4), \
            name5(that.name5), \
            name6(that.name6) \
        { \
        } \
        \
        sysinline Kit \
        ( \
            typenameWord() ParamType< Type0 >::T name0, \
            typenameWord() ParamType< Type1 >::T name1, \
            typenameWord() ParamType< Type2 >::T name2, \
            typenameWord() ParamType< Type3 >::T name3, \
            typenameWord() ParamType< Type4 >::T name4, \
            typenameWord() ParamType< Type5 >::T name5, \
            typenameWord() ParamType< Type6 >::T name6 \
        ) \
            : \
            name0(name0), \
            name1(name1), \
            name2(name2), \
            name3(name3), \
            name4(name4), \
            name5(name5), \
            name6(name6) \
        { \
        } \
        \
        template <typename OtherKit> \
        sysinline Kit(const OtherKit& otherKit) \
            : \
            name0(otherKit.name0), \
            name1(otherKit.name1), \
            name2(otherKit.name2), \
            name3(otherKit.name3), \
            name4(otherKit.name4), \
            name5(otherKit.name5), \
            name6(otherKit.name6) \
        { \
        } \
        \
        template <typename OldKit, typename NewKit> \
        sysinline Kit(const OldKit& oldKit, KitReplaceConstructor, const NewKit& newKit) \
            : \
            name0(KitReplacer<OldKit, NewKit, name0##_Tag>::func(&oldKit, &newKit)->name0), \
            name1(KitReplacer<OldKit, NewKit, name1##_Tag>::func(&oldKit, &newKit)->name1), \
            name2(KitReplacer<OldKit, NewKit, name2##_Tag>::func(&oldKit, &newKit)->name2), \
            name3(KitReplacer<OldKit, NewKit, name3##_Tag>::func(&oldKit, &newKit)->name3), \
            name4(KitReplacer<OldKit, NewKit, name4##_Tag>::func(&oldKit, &newKit)->name4), \
            name5(KitReplacer<OldKit, NewKit, name5##_Tag>::func(&oldKit, &newKit)->name5), \
            name6(KitReplacer<OldKit, NewKit, name6##_Tag>::func(&oldKit, &newKit)->name6) \
        { \
        } \
        \
    }

//----------------------------------------------------------------

#define KIT_CREATE7(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6) \
    KIT__CREATE7(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, KIT__TYPENAME_NO)

#define KIT_CREATE7_(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6) \
    KIT__CREATE7(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, KIT__TYPENAME_YES)

//----------------------------------------------------------------

#define KIT__CREATE8(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, typenameWord) \
    \
    struct Kit \
        : \
        KitFieldTag<struct name0##_Tag>, \
        KitFieldTag<struct name1##_Tag>, \
        KitFieldTag<struct name2##_Tag>, \
        KitFieldTag<struct name3##_Tag>, \
        KitFieldTag<struct name4##_Tag>, \
        KitFieldTag<struct name5##_Tag>, \
        KitFieldTag<struct name6##_Tag>, \
        KitFieldTag<struct name7##_Tag> \
    { \
        \
        Type0 name0; \
        Type1 name1; \
        Type2 name2; \
        Type3 name3; \
        Type4 name4; \
        Type5 name5; \
        Type6 name6; \
        Type7 name7; \
        \
        sysinline Kit(const Kit& that) \
            : \
            name0(that.name0), \
            name1(that.name1), \
            name2(that.name2), \
            name3(that.name3), \
            name4(that.name4), \
            name5(that.name5), \
            name6(that.name6), \
            name7(that.name7) \
        { \
        } \
        \
        sysinline Kit \
        ( \
            typenameWord() ParamType< Type0 >::T name0, \
            typenameWord() ParamType< Type1 >::T name1, \
            typenameWord() ParamType< Type2 >::T name2, \
            typenameWord() ParamType< Type3 >::T name3, \
            typenameWord() ParamType< Type4 >::T name4, \
            typenameWord() ParamType< Type5 >::T name5, \
            typenameWord() ParamType< Type6 >::T name6, \
            typenameWord() ParamType< Type7 >::T name7 \
        ) \
            : \
            name0(name0), \
            name1(name1), \
            name2(name2), \
            name3(name3), \
            name4(name4), \
            name5(name5), \
            name6(name6), \
            name7(name7) \
        { \
        } \
        \
        template <typename OtherKit> \
        sysinline Kit(const OtherKit& otherKit) \
            : \
            name0(otherKit.name0), \
            name1(otherKit.name1), \
            name2(otherKit.name2), \
            name3(otherKit.name3), \
            name4(otherKit.name4), \
            name5(otherKit.name5), \
            name6(otherKit.name6), \
            name7(otherKit.name7) \
        { \
        } \
        \
        template <typename OldKit, typename NewKit> \
        sysinline Kit(const OldKit& oldKit, KitReplaceConstructor, const NewKit& newKit) \
            : \
            name0(KitReplacer<OldKit, NewKit, name0##_Tag>::func(&oldKit, &newKit)->name0), \
            name1(KitReplacer<OldKit, NewKit, name1##_Tag>::func(&oldKit, &newKit)->name1), \
            name2(KitReplacer<OldKit, NewKit, name2##_Tag>::func(&oldKit, &newKit)->name2), \
            name3(KitReplacer<OldKit, NewKit, name3##_Tag>::func(&oldKit, &newKit)->name3), \
            name4(KitReplacer<OldKit, NewKit, name4##_Tag>::func(&oldKit, &newKit)->name4), \
            name5(KitReplacer<OldKit, NewKit, name5##_Tag>::func(&oldKit, &newKit)->name5), \
            name6(KitReplacer<OldKit, NewKit, name6##_Tag>::func(&oldKit, &newKit)->name6), \
            name7(KitReplacer<OldKit, NewKit, name7##_Tag>::func(&oldKit, &newKit)->name7) \
        { \
        } \
        \
    }

//----------------------------------------------------------------

#define KIT_CREATE8(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7) \
    KIT__CREATE8(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, KIT__TYPENAME_NO)

#define KIT_CREATE8_(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7) \
    KIT__CREATE8(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, KIT__TYPENAME_YES)

//----------------------------------------------------------------

#define KIT__CREATE9(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, typenameWord) \
    \
    struct Kit \
        : \
        KitFieldTag<struct name0##_Tag>, \
        KitFieldTag<struct name1##_Tag>, \
        KitFieldTag<struct name2##_Tag>, \
        KitFieldTag<struct name3##_Tag>, \
        KitFieldTag<struct name4##_Tag>, \
        KitFieldTag<struct name5##_Tag>, \
        KitFieldTag<struct name6##_Tag>, \
        KitFieldTag<struct name7##_Tag>, \
        KitFieldTag<struct name8##_Tag> \
    { \
        \
        Type0 name0; \
        Type1 name1; \
        Type2 name2; \
        Type3 name3; \
        Type4 name4; \
        Type5 name5; \
        Type6 name6; \
        Type7 name7; \
        Type8 name8; \
        \
        sysinline Kit(const Kit& that) \
            : \
            name0(that.name0), \
            name1(that.name1), \
            name2(that.name2), \
            name3(that.name3), \
            name4(that.name4), \
            name5(that.name5), \
            name6(that.name6), \
            name7(that.name7), \
            name8(that.name8) \
        { \
        } \
        \
        sysinline Kit \
        ( \
            typenameWord() ParamType< Type0 >::T name0, \
            typenameWord() ParamType< Type1 >::T name1, \
            typenameWord() ParamType< Type2 >::T name2, \
            typenameWord() ParamType< Type3 >::T name3, \
            typenameWord() ParamType< Type4 >::T name4, \
            typenameWord() ParamType< Type5 >::T name5, \
            typenameWord() ParamType< Type6 >::T name6, \
            typenameWord() ParamType< Type7 >::T name7, \
            typenameWord() ParamType< Type8 >::T name8 \
        ) \
            : \
            name0(name0), \
            name1(name1), \
            name2(name2), \
            name3(name3), \
            name4(name4), \
            name5(name5), \
            name6(name6), \
            name7(name7), \
            name8(name8) \
        { \
        } \
        \
        template <typename OtherKit> \
        sysinline Kit(const OtherKit& otherKit) \
            : \
            name0(otherKit.name0), \
            name1(otherKit.name1), \
            name2(otherKit.name2), \
            name3(otherKit.name3), \
            name4(otherKit.name4), \
            name5(otherKit.name5), \
            name6(otherKit.name6), \
            name7(otherKit.name7), \
            name8(otherKit.name8) \
        { \
        } \
        \
        template <typename OldKit, typename NewKit> \
        sysinline Kit(const OldKit& oldKit, KitReplaceConstructor, const NewKit& newKit) \
            : \
            name0(KitReplacer<OldKit, NewKit, name0##_Tag>::func(&oldKit, &newKit)->name0), \
            name1(KitReplacer<OldKit, NewKit, name1##_Tag>::func(&oldKit, &newKit)->name1), \
            name2(KitReplacer<OldKit, NewKit, name2##_Tag>::func(&oldKit, &newKit)->name2), \
            name3(KitReplacer<OldKit, NewKit, name3##_Tag>::func(&oldKit, &newKit)->name3), \
            name4(KitReplacer<OldKit, NewKit, name4##_Tag>::func(&oldKit, &newKit)->name4), \
            name5(KitReplacer<OldKit, NewKit, name5##_Tag>::func(&oldKit, &newKit)->name5), \
            name6(KitReplacer<OldKit, NewKit, name6##_Tag>::func(&oldKit, &newKit)->name6), \
            name7(KitReplacer<OldKit, NewKit, name7##_Tag>::func(&oldKit, &newKit)->name7), \
            name8(KitReplacer<OldKit, NewKit, name8##_Tag>::func(&oldKit, &newKit)->name8) \
        { \
        } \
        \
    }

//----------------------------------------------------------------

#define KIT_CREATE9(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8) \
    KIT__CREATE9(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, KIT__TYPENAME_NO)

#define KIT_CREATE9_(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8) \
    KIT__CREATE9(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, KIT__TYPENAME_YES)

//----------------------------------------------------------------

#define KIT__CREATE10(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, typenameWord) \
    \
    struct Kit \
        : \
        KitFieldTag<struct name0##_Tag>, \
        KitFieldTag<struct name1##_Tag>, \
        KitFieldTag<struct name2##_Tag>, \
        KitFieldTag<struct name3##_Tag>, \
        KitFieldTag<struct name4##_Tag>, \
        KitFieldTag<struct name5##_Tag>, \
        KitFieldTag<struct name6##_Tag>, \
        KitFieldTag<struct name7##_Tag>, \
        KitFieldTag<struct name8##_Tag>, \
        KitFieldTag<struct name9##_Tag> \
    { \
        \
        Type0 name0; \
        Type1 name1; \
        Type2 name2; \
        Type3 name3; \
        Type4 name4; \
        Type5 name5; \
        Type6 name6; \
        Type7 name7; \
        Type8 name8; \
        Type9 name9; \
        \
        sysinline Kit(const Kit& that) \
            : \
            name0(that.name0), \
            name1(that.name1), \
            name2(that.name2), \
            name3(that.name3), \
            name4(that.name4), \
            name5(that.name5), \
            name6(that.name6), \
            name7(that.name7), \
            name8(that.name8), \
            name9(that.name9) \
        { \
        } \
        \
        sysinline Kit \
        ( \
            typenameWord() ParamType< Type0 >::T name0, \
            typenameWord() ParamType< Type1 >::T name1, \
            typenameWord() ParamType< Type2 >::T name2, \
            typenameWord() ParamType< Type3 >::T name3, \
            typenameWord() ParamType< Type4 >::T name4, \
            typenameWord() ParamType< Type5 >::T name5, \
            typenameWord() ParamType< Type6 >::T name6, \
            typenameWord() ParamType< Type7 >::T name7, \
            typenameWord() ParamType< Type8 >::T name8, \
            typenameWord() ParamType< Type9 >::T name9 \
        ) \
            : \
            name0(name0), \
            name1(name1), \
            name2(name2), \
            name3(name3), \
            name4(name4), \
            name5(name5), \
            name6(name6), \
            name7(name7), \
            name8(name8), \
            name9(name9) \
        { \
        } \
        \
        template <typename OtherKit> \
        sysinline Kit(const OtherKit& otherKit) \
            : \
            name0(otherKit.name0), \
            name1(otherKit.name1), \
            name2(otherKit.name2), \
            name3(otherKit.name3), \
            name4(otherKit.name4), \
            name5(otherKit.name5), \
            name6(otherKit.name6), \
            name7(otherKit.name7), \
            name8(otherKit.name8), \
            name9(otherKit.name9) \
        { \
        } \
        \
        template <typename OldKit, typename NewKit> \
        sysinline Kit(const OldKit& oldKit, KitReplaceConstructor, const NewKit& newKit) \
            : \
            name0(KitReplacer<OldKit, NewKit, name0##_Tag>::func(&oldKit, &newKit)->name0), \
            name1(KitReplacer<OldKit, NewKit, name1##_Tag>::func(&oldKit, &newKit)->name1), \
            name2(KitReplacer<OldKit, NewKit, name2##_Tag>::func(&oldKit, &newKit)->name2), \
            name3(KitReplacer<OldKit, NewKit, name3##_Tag>::func(&oldKit, &newKit)->name3), \
            name4(KitReplacer<OldKit, NewKit, name4##_Tag>::func(&oldKit, &newKit)->name4), \
            name5(KitReplacer<OldKit, NewKit, name5##_Tag>::func(&oldKit, &newKit)->name5), \
            name6(KitReplacer<OldKit, NewKit, name6##_Tag>::func(&oldKit, &newKit)->name6), \
            name7(KitReplacer<OldKit, NewKit, name7##_Tag>::func(&oldKit, &newKit)->name7), \
            name8(KitReplacer<OldKit, NewKit, name8##_Tag>::func(&oldKit, &newKit)->name8), \
            name9(KitReplacer<OldKit, NewKit, name9##_Tag>::func(&oldKit, &newKit)->name9) \
        { \
        } \
        \
    }

//----------------------------------------------------------------

#define KIT_CREATE10(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9) \
    KIT__CREATE10(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, KIT__TYPENAME_NO)

#define KIT_CREATE10_(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9) \
    KIT__CREATE10(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, KIT__TYPENAME_YES)

//----------------------------------------------------------------

#define KIT__CREATE11(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, typenameWord) \
    \
    struct Kit \
        : \
        KitFieldTag<struct name0##_Tag>, \
        KitFieldTag<struct name1##_Tag>, \
        KitFieldTag<struct name2##_Tag>, \
        KitFieldTag<struct name3##_Tag>, \
        KitFieldTag<struct name4##_Tag>, \
        KitFieldTag<struct name5##_Tag>, \
        KitFieldTag<struct name6##_Tag>, \
        KitFieldTag<struct name7##_Tag>, \
        KitFieldTag<struct name8##_Tag>, \
        KitFieldTag<struct name9##_Tag>, \
        KitFieldTag<struct name10##_Tag> \
    { \
        \
        Type0 name0; \
        Type1 name1; \
        Type2 name2; \
        Type3 name3; \
        Type4 name4; \
        Type5 name5; \
        Type6 name6; \
        Type7 name7; \
        Type8 name8; \
        Type9 name9; \
        Type10 name10; \
        \
        sysinline Kit(const Kit& that) \
            : \
            name0(that.name0), \
            name1(that.name1), \
            name2(that.name2), \
            name3(that.name3), \
            name4(that.name4), \
            name5(that.name5), \
            name6(that.name6), \
            name7(that.name7), \
            name8(that.name8), \
            name9(that.name9), \
            name10(that.name10) \
        { \
        } \
        \
        sysinline Kit \
        ( \
            typenameWord() ParamType< Type0 >::T name0, \
            typenameWord() ParamType< Type1 >::T name1, \
            typenameWord() ParamType< Type2 >::T name2, \
            typenameWord() ParamType< Type3 >::T name3, \
            typenameWord() ParamType< Type4 >::T name4, \
            typenameWord() ParamType< Type5 >::T name5, \
            typenameWord() ParamType< Type6 >::T name6, \
            typenameWord() ParamType< Type7 >::T name7, \
            typenameWord() ParamType< Type8 >::T name8, \
            typenameWord() ParamType< Type9 >::T name9, \
            typenameWord() ParamType< Type10 >::T name10 \
        ) \
            : \
            name0(name0), \
            name1(name1), \
            name2(name2), \
            name3(name3), \
            name4(name4), \
            name5(name5), \
            name6(name6), \
            name7(name7), \
            name8(name8), \
            name9(name9), \
            name10(name10) \
        { \
        } \
        \
        template <typename OtherKit> \
        sysinline Kit(const OtherKit& otherKit) \
            : \
            name0(otherKit.name0), \
            name1(otherKit.name1), \
            name2(otherKit.name2), \
            name3(otherKit.name3), \
            name4(otherKit.name4), \
            name5(otherKit.name5), \
            name6(otherKit.name6), \
            name7(otherKit.name7), \
            name8(otherKit.name8), \
            name9(otherKit.name9), \
            name10(otherKit.name10) \
        { \
        } \
        \
        template <typename OldKit, typename NewKit> \
        sysinline Kit(const OldKit& oldKit, KitReplaceConstructor, const NewKit& newKit) \
            : \
            name0(KitReplacer<OldKit, NewKit, name0##_Tag>::func(&oldKit, &newKit)->name0), \
            name1(KitReplacer<OldKit, NewKit, name1##_Tag>::func(&oldKit, &newKit)->name1), \
            name2(KitReplacer<OldKit, NewKit, name2##_Tag>::func(&oldKit, &newKit)->name2), \
            name3(KitReplacer<OldKit, NewKit, name3##_Tag>::func(&oldKit, &newKit)->name3), \
            name4(KitReplacer<OldKit, NewKit, name4##_Tag>::func(&oldKit, &newKit)->name4), \
            name5(KitReplacer<OldKit, NewKit, name5##_Tag>::func(&oldKit, &newKit)->name5), \
            name6(KitReplacer<OldKit, NewKit, name6##_Tag>::func(&oldKit, &newKit)->name6), \
            name7(KitReplacer<OldKit, NewKit, name7##_Tag>::func(&oldKit, &newKit)->name7), \
            name8(KitReplacer<OldKit, NewKit, name8##_Tag>::func(&oldKit, &newKit)->name8), \
            name9(KitReplacer<OldKit, NewKit, name9##_Tag>::func(&oldKit, &newKit)->name9), \
            name10(KitReplacer<OldKit, NewKit, name10##_Tag>::func(&oldKit, &newKit)->name10) \
        { \
        } \
        \
    }

//----------------------------------------------------------------

#define KIT_CREATE11(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10) \
    KIT__CREATE11(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, KIT__TYPENAME_NO)

#define KIT_CREATE11_(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10) \
    KIT__CREATE11(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, KIT__TYPENAME_YES)

//----------------------------------------------------------------

#define KIT__CREATE12(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, typenameWord) \
    \
    struct Kit \
        : \
        KitFieldTag<struct name0##_Tag>, \
        KitFieldTag<struct name1##_Tag>, \
        KitFieldTag<struct name2##_Tag>, \
        KitFieldTag<struct name3##_Tag>, \
        KitFieldTag<struct name4##_Tag>, \
        KitFieldTag<struct name5##_Tag>, \
        KitFieldTag<struct name6##_Tag>, \
        KitFieldTag<struct name7##_Tag>, \
        KitFieldTag<struct name8##_Tag>, \
        KitFieldTag<struct name9##_Tag>, \
        KitFieldTag<struct name10##_Tag>, \
        KitFieldTag<struct name11##_Tag> \
    { \
        \
        Type0 name0; \
        Type1 name1; \
        Type2 name2; \
        Type3 name3; \
        Type4 name4; \
        Type5 name5; \
        Type6 name6; \
        Type7 name7; \
        Type8 name8; \
        Type9 name9; \
        Type10 name10; \
        Type11 name11; \
        \
        sysinline Kit(const Kit& that) \
            : \
            name0(that.name0), \
            name1(that.name1), \
            name2(that.name2), \
            name3(that.name3), \
            name4(that.name4), \
            name5(that.name5), \
            name6(that.name6), \
            name7(that.name7), \
            name8(that.name8), \
            name9(that.name9), \
            name10(that.name10), \
            name11(that.name11) \
        { \
        } \
        \
        sysinline Kit \
        ( \
            typenameWord() ParamType< Type0 >::T name0, \
            typenameWord() ParamType< Type1 >::T name1, \
            typenameWord() ParamType< Type2 >::T name2, \
            typenameWord() ParamType< Type3 >::T name3, \
            typenameWord() ParamType< Type4 >::T name4, \
            typenameWord() ParamType< Type5 >::T name5, \
            typenameWord() ParamType< Type6 >::T name6, \
            typenameWord() ParamType< Type7 >::T name7, \
            typenameWord() ParamType< Type8 >::T name8, \
            typenameWord() ParamType< Type9 >::T name9, \
            typenameWord() ParamType< Type10 >::T name10, \
            typenameWord() ParamType< Type11 >::T name11 \
        ) \
            : \
            name0(name0), \
            name1(name1), \
            name2(name2), \
            name3(name3), \
            name4(name4), \
            name5(name5), \
            name6(name6), \
            name7(name7), \
            name8(name8), \
            name9(name9), \
            name10(name10), \
            name11(name11) \
        { \
        } \
        \
        template <typename OtherKit> \
        sysinline Kit(const OtherKit& otherKit) \
            : \
            name0(otherKit.name0), \
            name1(otherKit.name1), \
            name2(otherKit.name2), \
            name3(otherKit.name3), \
            name4(otherKit.name4), \
            name5(otherKit.name5), \
            name6(otherKit.name6), \
            name7(otherKit.name7), \
            name8(otherKit.name8), \
            name9(otherKit.name9), \
            name10(otherKit.name10), \
            name11(otherKit.name11) \
        { \
        } \
        \
        template <typename OldKit, typename NewKit> \
        sysinline Kit(const OldKit& oldKit, KitReplaceConstructor, const NewKit& newKit) \
            : \
            name0(KitReplacer<OldKit, NewKit, name0##_Tag>::func(&oldKit, &newKit)->name0), \
            name1(KitReplacer<OldKit, NewKit, name1##_Tag>::func(&oldKit, &newKit)->name1), \
            name2(KitReplacer<OldKit, NewKit, name2##_Tag>::func(&oldKit, &newKit)->name2), \
            name3(KitReplacer<OldKit, NewKit, name3##_Tag>::func(&oldKit, &newKit)->name3), \
            name4(KitReplacer<OldKit, NewKit, name4##_Tag>::func(&oldKit, &newKit)->name4), \
            name5(KitReplacer<OldKit, NewKit, name5##_Tag>::func(&oldKit, &newKit)->name5), \
            name6(KitReplacer<OldKit, NewKit, name6##_Tag>::func(&oldKit, &newKit)->name6), \
            name7(KitReplacer<OldKit, NewKit, name7##_Tag>::func(&oldKit, &newKit)->name7), \
            name8(KitReplacer<OldKit, NewKit, name8##_Tag>::func(&oldKit, &newKit)->name8), \
            name9(KitReplacer<OldKit, NewKit, name9##_Tag>::func(&oldKit, &newKit)->name9), \
            name10(KitReplacer<OldKit, NewKit, name10##_Tag>::func(&oldKit, &newKit)->name10), \
            name11(KitReplacer<OldKit, NewKit, name11##_Tag>::func(&oldKit, &newKit)->name11) \
        { \
        } \
        \
    }

//----------------------------------------------------------------

#define KIT_CREATE12(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11) \
    KIT__CREATE12(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, KIT__TYPENAME_NO)

#define KIT_CREATE12_(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11) \
    KIT__CREATE12(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, KIT__TYPENAME_YES)

//----------------------------------------------------------------

#define KIT__CREATE13(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, Type12, name12, typenameWord) \
    \
    struct Kit \
        : \
        KitFieldTag<struct name0##_Tag>, \
        KitFieldTag<struct name1##_Tag>, \
        KitFieldTag<struct name2##_Tag>, \
        KitFieldTag<struct name3##_Tag>, \
        KitFieldTag<struct name4##_Tag>, \
        KitFieldTag<struct name5##_Tag>, \
        KitFieldTag<struct name6##_Tag>, \
        KitFieldTag<struct name7##_Tag>, \
        KitFieldTag<struct name8##_Tag>, \
        KitFieldTag<struct name9##_Tag>, \
        KitFieldTag<struct name10##_Tag>, \
        KitFieldTag<struct name11##_Tag>, \
        KitFieldTag<struct name12##_Tag> \
    { \
        \
        Type0 name0; \
        Type1 name1; \
        Type2 name2; \
        Type3 name3; \
        Type4 name4; \
        Type5 name5; \
        Type6 name6; \
        Type7 name7; \
        Type8 name8; \
        Type9 name9; \
        Type10 name10; \
        Type11 name11; \
        Type12 name12; \
        \
        sysinline Kit(const Kit& that) \
            : \
            name0(that.name0), \
            name1(that.name1), \
            name2(that.name2), \
            name3(that.name3), \
            name4(that.name4), \
            name5(that.name5), \
            name6(that.name6), \
            name7(that.name7), \
            name8(that.name8), \
            name9(that.name9), \
            name10(that.name10), \
            name11(that.name11), \
            name12(that.name12) \
        { \
        } \
        \
        sysinline Kit \
        ( \
            typenameWord() ParamType< Type0 >::T name0, \
            typenameWord() ParamType< Type1 >::T name1, \
            typenameWord() ParamType< Type2 >::T name2, \
            typenameWord() ParamType< Type3 >::T name3, \
            typenameWord() ParamType< Type4 >::T name4, \
            typenameWord() ParamType< Type5 >::T name5, \
            typenameWord() ParamType< Type6 >::T name6, \
            typenameWord() ParamType< Type7 >::T name7, \
            typenameWord() ParamType< Type8 >::T name8, \
            typenameWord() ParamType< Type9 >::T name9, \
            typenameWord() ParamType< Type10 >::T name10, \
            typenameWord() ParamType< Type11 >::T name11, \
            typenameWord() ParamType< Type12 >::T name12 \
        ) \
            : \
            name0(name0), \
            name1(name1), \
            name2(name2), \
            name3(name3), \
            name4(name4), \
            name5(name5), \
            name6(name6), \
            name7(name7), \
            name8(name8), \
            name9(name9), \
            name10(name10), \
            name11(name11), \
            name12(name12) \
        { \
        } \
        \
        template <typename OtherKit> \
        sysinline Kit(const OtherKit& otherKit) \
            : \
            name0(otherKit.name0), \
            name1(otherKit.name1), \
            name2(otherKit.name2), \
            name3(otherKit.name3), \
            name4(otherKit.name4), \
            name5(otherKit.name5), \
            name6(otherKit.name6), \
            name7(otherKit.name7), \
            name8(otherKit.name8), \
            name9(otherKit.name9), \
            name10(otherKit.name10), \
            name11(otherKit.name11), \
            name12(otherKit.name12) \
        { \
        } \
        \
        template <typename OldKit, typename NewKit> \
        sysinline Kit(const OldKit& oldKit, KitReplaceConstructor, const NewKit& newKit) \
            : \
            name0(KitReplacer<OldKit, NewKit, name0##_Tag>::func(&oldKit, &newKit)->name0), \
            name1(KitReplacer<OldKit, NewKit, name1##_Tag>::func(&oldKit, &newKit)->name1), \
            name2(KitReplacer<OldKit, NewKit, name2##_Tag>::func(&oldKit, &newKit)->name2), \
            name3(KitReplacer<OldKit, NewKit, name3##_Tag>::func(&oldKit, &newKit)->name3), \
            name4(KitReplacer<OldKit, NewKit, name4##_Tag>::func(&oldKit, &newKit)->name4), \
            name5(KitReplacer<OldKit, NewKit, name5##_Tag>::func(&oldKit, &newKit)->name5), \
            name6(KitReplacer<OldKit, NewKit, name6##_Tag>::func(&oldKit, &newKit)->name6), \
            name7(KitReplacer<OldKit, NewKit, name7##_Tag>::func(&oldKit, &newKit)->name7), \
            name8(KitReplacer<OldKit, NewKit, name8##_Tag>::func(&oldKit, &newKit)->name8), \
            name9(KitReplacer<OldKit, NewKit, name9##_Tag>::func(&oldKit, &newKit)->name9), \
            name10(KitReplacer<OldKit, NewKit, name10##_Tag>::func(&oldKit, &newKit)->name10), \
            name11(KitReplacer<OldKit, NewKit, name11##_Tag>::func(&oldKit, &newKit)->name11), \
            name12(KitReplacer<OldKit, NewKit, name12##_Tag>::func(&oldKit, &newKit)->name12) \
        { \
        } \
        \
    }

//----------------------------------------------------------------

#define KIT_CREATE13(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, Type12, name12) \
    KIT__CREATE13(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, Type12, name12, KIT__TYPENAME_NO)

#define KIT_CREATE13_(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, Type12, name12) \
    KIT__CREATE13(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, Type12, name12, KIT__TYPENAME_YES)

//----------------------------------------------------------------

#define KIT__CREATE14(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, Type12, name12, Type13, name13, typenameWord) \
    \
    struct Kit \
        : \
        KitFieldTag<struct name0##_Tag>, \
        KitFieldTag<struct name1##_Tag>, \
        KitFieldTag<struct name2##_Tag>, \
        KitFieldTag<struct name3##_Tag>, \
        KitFieldTag<struct name4##_Tag>, \
        KitFieldTag<struct name5##_Tag>, \
        KitFieldTag<struct name6##_Tag>, \
        KitFieldTag<struct name7##_Tag>, \
        KitFieldTag<struct name8##_Tag>, \
        KitFieldTag<struct name9##_Tag>, \
        KitFieldTag<struct name10##_Tag>, \
        KitFieldTag<struct name11##_Tag>, \
        KitFieldTag<struct name12##_Tag>, \
        KitFieldTag<struct name13##_Tag> \
    { \
        \
        Type0 name0; \
        Type1 name1; \
        Type2 name2; \
        Type3 name3; \
        Type4 name4; \
        Type5 name5; \
        Type6 name6; \
        Type7 name7; \
        Type8 name8; \
        Type9 name9; \
        Type10 name10; \
        Type11 name11; \
        Type12 name12; \
        Type13 name13; \
        \
        sysinline Kit(const Kit& that) \
            : \
            name0(that.name0), \
            name1(that.name1), \
            name2(that.name2), \
            name3(that.name3), \
            name4(that.name4), \
            name5(that.name5), \
            name6(that.name6), \
            name7(that.name7), \
            name8(that.name8), \
            name9(that.name9), \
            name10(that.name10), \
            name11(that.name11), \
            name12(that.name12), \
            name13(that.name13) \
        { \
        } \
        \
        sysinline Kit \
        ( \
            typenameWord() ParamType< Type0 >::T name0, \
            typenameWord() ParamType< Type1 >::T name1, \
            typenameWord() ParamType< Type2 >::T name2, \
            typenameWord() ParamType< Type3 >::T name3, \
            typenameWord() ParamType< Type4 >::T name4, \
            typenameWord() ParamType< Type5 >::T name5, \
            typenameWord() ParamType< Type6 >::T name6, \
            typenameWord() ParamType< Type7 >::T name7, \
            typenameWord() ParamType< Type8 >::T name8, \
            typenameWord() ParamType< Type9 >::T name9, \
            typenameWord() ParamType< Type10 >::T name10, \
            typenameWord() ParamType< Type11 >::T name11, \
            typenameWord() ParamType< Type12 >::T name12, \
            typenameWord() ParamType< Type13 >::T name13 \
        ) \
            : \
            name0(name0), \
            name1(name1), \
            name2(name2), \
            name3(name3), \
            name4(name4), \
            name5(name5), \
            name6(name6), \
            name7(name7), \
            name8(name8), \
            name9(name9), \
            name10(name10), \
            name11(name11), \
            name12(name12), \
            name13(name13) \
        { \
        } \
        \
        template <typename OtherKit> \
        sysinline Kit(const OtherKit& otherKit) \
            : \
            name0(otherKit.name0), \
            name1(otherKit.name1), \
            name2(otherKit.name2), \
            name3(otherKit.name3), \
            name4(otherKit.name4), \
            name5(otherKit.name5), \
            name6(otherKit.name6), \
            name7(otherKit.name7), \
            name8(otherKit.name8), \
            name9(otherKit.name9), \
            name10(otherKit.name10), \
            name11(otherKit.name11), \
            name12(otherKit.name12), \
            name13(otherKit.name13) \
        { \
        } \
        \
        template <typename OldKit, typename NewKit> \
        sysinline Kit(const OldKit& oldKit, KitReplaceConstructor, const NewKit& newKit) \
            : \
            name0(KitReplacer<OldKit, NewKit, name0##_Tag>::func(&oldKit, &newKit)->name0), \
            name1(KitReplacer<OldKit, NewKit, name1##_Tag>::func(&oldKit, &newKit)->name1), \
            name2(KitReplacer<OldKit, NewKit, name2##_Tag>::func(&oldKit, &newKit)->name2), \
            name3(KitReplacer<OldKit, NewKit, name3##_Tag>::func(&oldKit, &newKit)->name3), \
            name4(KitReplacer<OldKit, NewKit, name4##_Tag>::func(&oldKit, &newKit)->name4), \
            name5(KitReplacer<OldKit, NewKit, name5##_Tag>::func(&oldKit, &newKit)->name5), \
            name6(KitReplacer<OldKit, NewKit, name6##_Tag>::func(&oldKit, &newKit)->name6), \
            name7(KitReplacer<OldKit, NewKit, name7##_Tag>::func(&oldKit, &newKit)->name7), \
            name8(KitReplacer<OldKit, NewKit, name8##_Tag>::func(&oldKit, &newKit)->name8), \
            name9(KitReplacer<OldKit, NewKit, name9##_Tag>::func(&oldKit, &newKit)->name9), \
            name10(KitReplacer<OldKit, NewKit, name10##_Tag>::func(&oldKit, &newKit)->name10), \
            name11(KitReplacer<OldKit, NewKit, name11##_Tag>::func(&oldKit, &newKit)->name11), \
            name12(KitReplacer<OldKit, NewKit, name12##_Tag>::func(&oldKit, &newKit)->name12), \
            name13(KitReplacer<OldKit, NewKit, name13##_Tag>::func(&oldKit, &newKit)->name13) \
        { \
        } \
        \
    }

//----------------------------------------------------------------

#define KIT_CREATE14(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, Type12, name12, Type13, name13) \
    KIT__CREATE14(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, Type12, name12, Type13, name13, KIT__TYPENAME_NO)

#define KIT_CREATE14_(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, Type12, name12, Type13, name13) \
    KIT__CREATE14(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, Type12, name12, Type13, name13, KIT__TYPENAME_YES)

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// KIT_COMBINE
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

#define KIT_COMBINE0(Kit) \
    \
    struct Kit \
    { \
        template <typename OtherKit> \
        sysinline Kit(const OtherKit& otherKit) \
        { \
        } \
        \
        template <typename OldKit, typename NewKit> \
        sysinline Kit(const OldKit& oldKit, KitReplaceConstructor, const NewKit& newKit) \
        { \
        } \
        \
        sysinline Kit \
        ( \
            KitCombineConstructor \
        ) \
        { \
        } \
    }

//----------------------------------------------------------------

KIT_COMBINE0(KitCombine0);

//----------------------------------------------------------------

sysinline KitCombine0 kitCombine()
    {return KitCombine0(KitCombineConstructor());}

//----------------------------------------------------------------

#define KIT_COMBINE1(Kit, T0) \
    \
    struct Kit \
        : \
        public T0 \
    { \
        template <typename OtherKit> \
        sysinline Kit(const OtherKit& otherKit) \
            : \
            T0(otherKit) \
        { \
        } \
        \
        template <typename OldKit, typename NewKit> \
        sysinline Kit(const OldKit& oldKit, KitReplaceConstructor, const NewKit& newKit) \
            : \
            T0(oldKit, KitReplaceConstructor(), newKit) \
        { \
        } \
        \
        template <typename Q0> \
        sysinline Kit \
        ( \
            const Q0& q0, \
            KitCombineConstructor \
        ) \
            : \
            T0(q0) \
        { \
        } \
    }

//----------------------------------------------------------------

template <typename T0>
KIT_COMBINE1(KitCombine1, T0);

//----------------------------------------------------------------

template <typename T0>
sysinline KitCombine1<T0> kitCombine(const T0& v0)
    {return KitCombine1<T0>(v0, KitCombineConstructor());}

//----------------------------------------------------------------

#define KIT_COMBINE2(Kit, T0, T1) \
    \
    struct Kit \
        : \
        public T0, \
        public T1 \
    { \
        template <typename OtherKit> \
        sysinline Kit(const OtherKit& otherKit) \
            : \
            T0(otherKit), \
            T1(otherKit) \
        { \
        } \
        \
        template <typename OldKit, typename NewKit> \
        sysinline Kit(const OldKit& oldKit, KitReplaceConstructor, const NewKit& newKit) \
            : \
            T0(oldKit, KitReplaceConstructor(), newKit), \
            T1(oldKit, KitReplaceConstructor(), newKit) \
        { \
        } \
        \
        template <typename Q0, typename Q1> \
        sysinline Kit \
        ( \
            const Q0& q0, \
            const Q1& q1, \
            KitCombineConstructor \
        ) \
            : \
            T0(q0), \
            T1(q1) \
        { \
        } \
    }

//----------------------------------------------------------------

template <typename T0, typename T1>
KIT_COMBINE2(KitCombine2, T0, T1);

//----------------------------------------------------------------

template <typename T0, typename T1>
sysinline KitCombine2<T0, T1> kitCombine(const T0& v0, const T1& v1)
    {return KitCombine2<T0, T1>(v0, v1, KitCombineConstructor());}

//----------------------------------------------------------------

#define KIT_COMBINE3(Kit, T0, T1, T2) \
    \
    struct Kit \
        : \
        public T0, \
        public T1, \
        public T2 \
    { \
        template <typename OtherKit> \
        sysinline Kit(const OtherKit& otherKit) \
            : \
            T0(otherKit), \
            T1(otherKit), \
            T2(otherKit) \
        { \
        } \
        \
        template <typename OldKit, typename NewKit> \
        sysinline Kit(const OldKit& oldKit, KitReplaceConstructor, const NewKit& newKit) \
            : \
            T0(oldKit, KitReplaceConstructor(), newKit), \
            T1(oldKit, KitReplaceConstructor(), newKit), \
            T2(oldKit, KitReplaceConstructor(), newKit) \
        { \
        } \
        \
        template <typename Q0, typename Q1, typename Q2> \
        sysinline Kit \
        ( \
            const Q0& q0, \
            const Q1& q1, \
            const Q2& q2, \
            KitCombineConstructor \
        ) \
            : \
            T0(q0), \
            T1(q1), \
            T2(q2) \
        { \
        } \
    }

//----------------------------------------------------------------

template <typename T0, typename T1, typename T2>
KIT_COMBINE3(KitCombine3, T0, T1, T2);

//----------------------------------------------------------------

template <typename T0, typename T1, typename T2>
sysinline KitCombine3<T0, T1, T2> kitCombine(const T0& v0, const T1& v1, const T2& v2)
    {return KitCombine3<T0, T1, T2>(v0, v1, v2, KitCombineConstructor());}

//----------------------------------------------------------------

#define KIT_COMBINE4(Kit, T0, T1, T2, T3) \
    \
    struct Kit \
        : \
        public T0, \
        public T1, \
        public T2, \
        public T3 \
    { \
        template <typename OtherKit> \
        sysinline Kit(const OtherKit& otherKit) \
            : \
            T0(otherKit), \
            T1(otherKit), \
            T2(otherKit), \
            T3(otherKit) \
        { \
        } \
        \
        template <typename OldKit, typename NewKit> \
        sysinline Kit(const OldKit& oldKit, KitReplaceConstructor, const NewKit& newKit) \
            : \
            T0(oldKit, KitReplaceConstructor(), newKit), \
            T1(oldKit, KitReplaceConstructor(), newKit), \
            T2(oldKit, KitReplaceConstructor(), newKit), \
            T3(oldKit, KitReplaceConstructor(), newKit) \
        { \
        } \
        \
        template <typename Q0, typename Q1, typename Q2, typename Q3> \
        sysinline Kit \
        ( \
            const Q0& q0, \
            const Q1& q1, \
            const Q2& q2, \
            const Q3& q3, \
            KitCombineConstructor \
        ) \
            : \
            T0(q0), \
            T1(q1), \
            T2(q2), \
            T3(q3) \
        { \
        } \
    }

//----------------------------------------------------------------

template <typename T0, typename T1, typename T2, typename T3>
KIT_COMBINE4(KitCombine4, T0, T1, T2, T3);

//----------------------------------------------------------------

template <typename T0, typename T1, typename T2, typename T3>
sysinline KitCombine4<T0, T1, T2, T3> kitCombine(const T0& v0, const T1& v1, const T2& v2, const T3& v3)
    {return KitCombine4<T0, T1, T2, T3>(v0, v1, v2, v3, KitCombineConstructor());}

//----------------------------------------------------------------

#define KIT_COMBINE5(Kit, T0, T1, T2, T3, T4) \
    \
    struct Kit \
        : \
        public T0, \
        public T1, \
        public T2, \
        public T3, \
        public T4 \
    { \
        template <typename OtherKit> \
        sysinline Kit(const OtherKit& otherKit) \
            : \
            T0(otherKit), \
            T1(otherKit), \
            T2(otherKit), \
            T3(otherKit), \
            T4(otherKit) \
        { \
        } \
        \
        template <typename OldKit, typename NewKit> \
        sysinline Kit(const OldKit& oldKit, KitReplaceConstructor, const NewKit& newKit) \
            : \
            T0(oldKit, KitReplaceConstructor(), newKit), \
            T1(oldKit, KitReplaceConstructor(), newKit), \
            T2(oldKit, KitReplaceConstructor(), newKit), \
            T3(oldKit, KitReplaceConstructor(), newKit), \
            T4(oldKit, KitReplaceConstructor(), newKit) \
        { \
        } \
        \
        template <typename Q0, typename Q1, typename Q2, typename Q3, typename Q4> \
        sysinline Kit \
        ( \
            const Q0& q0, \
            const Q1& q1, \
            const Q2& q2, \
            const Q3& q3, \
            const Q4& q4, \
            KitCombineConstructor \
        ) \
            : \
            T0(q0), \
            T1(q1), \
            T2(q2), \
            T3(q3), \
            T4(q4) \
        { \
        } \
    }

//----------------------------------------------------------------

template <typename T0, typename T1, typename T2, typename T3, typename T4>
KIT_COMBINE5(KitCombine5, T0, T1, T2, T3, T4);

//----------------------------------------------------------------

template <typename T0, typename T1, typename T2, typename T3, typename T4>
sysinline KitCombine5<T0, T1, T2, T3, T4> kitCombine(const T0& v0, const T1& v1, const T2& v2, const T3& v3, const T4& v4)
    {return KitCombine5<T0, T1, T2, T3, T4>(v0, v1, v2, v3, v4, KitCombineConstructor());}

//----------------------------------------------------------------

#define KIT_COMBINE6(Kit, T0, T1, T2, T3, T4, T5) \
    \
    struct Kit \
        : \
        public T0, \
        public T1, \
        public T2, \
        public T3, \
        public T4, \
        public T5 \
    { \
        template <typename OtherKit> \
        sysinline Kit(const OtherKit& otherKit) \
            : \
            T0(otherKit), \
            T1(otherKit), \
            T2(otherKit), \
            T3(otherKit), \
            T4(otherKit), \
            T5(otherKit) \
        { \
        } \
        \
        template <typename OldKit, typename NewKit> \
        sysinline Kit(const OldKit& oldKit, KitReplaceConstructor, const NewKit& newKit) \
            : \
            T0(oldKit, KitReplaceConstructor(), newKit), \
            T1(oldKit, KitReplaceConstructor(), newKit), \
            T2(oldKit, KitReplaceConstructor(), newKit), \
            T3(oldKit, KitReplaceConstructor(), newKit), \
            T4(oldKit, KitReplaceConstructor(), newKit), \
            T5(oldKit, KitReplaceConstructor(), newKit) \
        { \
        } \
        \
        template <typename Q0, typename Q1, typename Q2, typename Q3, typename Q4, typename Q5> \
        sysinline Kit \
        ( \
            const Q0& q0, \
            const Q1& q1, \
            const Q2& q2, \
            const Q3& q3, \
            const Q4& q4, \
            const Q5& q5, \
            KitCombineConstructor \
        ) \
            : \
            T0(q0), \
            T1(q1), \
            T2(q2), \
            T3(q3), \
            T4(q4), \
            T5(q5) \
        { \
        } \
    }

//----------------------------------------------------------------

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5>
KIT_COMBINE6(KitCombine6, T0, T1, T2, T3, T4, T5);

//----------------------------------------------------------------

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5>
sysinline KitCombine6<T0, T1, T2, T3, T4, T5> kitCombine(const T0& v0, const T1& v1, const T2& v2, const T3& v3, const T4& v4, const T5& v5)
    {return KitCombine6<T0, T1, T2, T3, T4, T5>(v0, v1, v2, v3, v4, v5, KitCombineConstructor());}

//----------------------------------------------------------------

#define KIT_COMBINE7(Kit, T0, T1, T2, T3, T4, T5, T6) \
    \
    struct Kit \
        : \
        public T0, \
        public T1, \
        public T2, \
        public T3, \
        public T4, \
        public T5, \
        public T6 \
    { \
        template <typename OtherKit> \
        sysinline Kit(const OtherKit& otherKit) \
            : \
            T0(otherKit), \
            T1(otherKit), \
            T2(otherKit), \
            T3(otherKit), \
            T4(otherKit), \
            T5(otherKit), \
            T6(otherKit) \
        { \
        } \
        \
        template <typename OldKit, typename NewKit> \
        sysinline Kit(const OldKit& oldKit, KitReplaceConstructor, const NewKit& newKit) \
            : \
            T0(oldKit, KitReplaceConstructor(), newKit), \
            T1(oldKit, KitReplaceConstructor(), newKit), \
            T2(oldKit, KitReplaceConstructor(), newKit), \
            T3(oldKit, KitReplaceConstructor(), newKit), \
            T4(oldKit, KitReplaceConstructor(), newKit), \
            T5(oldKit, KitReplaceConstructor(), newKit), \
            T6(oldKit, KitReplaceConstructor(), newKit) \
        { \
        } \
        \
        template <typename Q0, typename Q1, typename Q2, typename Q3, typename Q4, typename Q5, typename Q6> \
        sysinline Kit \
        ( \
            const Q0& q0, \
            const Q1& q1, \
            const Q2& q2, \
            const Q3& q3, \
            const Q4& q4, \
            const Q5& q5, \
            const Q6& q6, \
            KitCombineConstructor \
        ) \
            : \
            T0(q0), \
            T1(q1), \
            T2(q2), \
            T3(q3), \
            T4(q4), \
            T5(q5), \
            T6(q6) \
        { \
        } \
    }

//----------------------------------------------------------------

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
KIT_COMBINE7(KitCombine7, T0, T1, T2, T3, T4, T5, T6);

//----------------------------------------------------------------

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
sysinline KitCombine7<T0, T1, T2, T3, T4, T5, T6> kitCombine(const T0& v0, const T1& v1, const T2& v2, const T3& v3, const T4& v4, const T5& v5, const T6& v6)
    {return KitCombine7<T0, T1, T2, T3, T4, T5, T6>(v0, v1, v2, v3, v4, v5, v6, KitCombineConstructor());}

//----------------------------------------------------------------

#define KIT_COMBINE8(Kit, T0, T1, T2, T3, T4, T5, T6, T7) \
    \
    struct Kit \
        : \
        public T0, \
        public T1, \
        public T2, \
        public T3, \
        public T4, \
        public T5, \
        public T6, \
        public T7 \
    { \
        template <typename OtherKit> \
        sysinline Kit(const OtherKit& otherKit) \
            : \
            T0(otherKit), \
            T1(otherKit), \
            T2(otherKit), \
            T3(otherKit), \
            T4(otherKit), \
            T5(otherKit), \
            T6(otherKit), \
            T7(otherKit) \
        { \
        } \
        \
        template <typename OldKit, typename NewKit> \
        sysinline Kit(const OldKit& oldKit, KitReplaceConstructor, const NewKit& newKit) \
            : \
            T0(oldKit, KitReplaceConstructor(), newKit), \
            T1(oldKit, KitReplaceConstructor(), newKit), \
            T2(oldKit, KitReplaceConstructor(), newKit), \
            T3(oldKit, KitReplaceConstructor(), newKit), \
            T4(oldKit, KitReplaceConstructor(), newKit), \
            T5(oldKit, KitReplaceConstructor(), newKit), \
            T6(oldKit, KitReplaceConstructor(), newKit), \
            T7(oldKit, KitReplaceConstructor(), newKit) \
        { \
        } \
        \
        template <typename Q0, typename Q1, typename Q2, typename Q3, typename Q4, typename Q5, typename Q6, typename Q7> \
        sysinline Kit \
        ( \
            const Q0& q0, \
            const Q1& q1, \
            const Q2& q2, \
            const Q3& q3, \
            const Q4& q4, \
            const Q5& q5, \
            const Q6& q6, \
            const Q7& q7, \
            KitCombineConstructor \
        ) \
            : \
            T0(q0), \
            T1(q1), \
            T2(q2), \
            T3(q3), \
            T4(q4), \
            T5(q5), \
            T6(q6), \
            T7(q7) \
        { \
        } \
    }

//----------------------------------------------------------------

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
KIT_COMBINE8(KitCombine8, T0, T1, T2, T3, T4, T5, T6, T7);

//----------------------------------------------------------------

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
sysinline KitCombine8<T0, T1, T2, T3, T4, T5, T6, T7> kitCombine(const T0& v0, const T1& v1, const T2& v2, const T3& v3, const T4& v4, const T5& v5, const T6& v6, const T7& v7)
    {return KitCombine8<T0, T1, T2, T3, T4, T5, T6, T7>(v0, v1, v2, v3, v4, v5, v6, v7, KitCombineConstructor());}

//----------------------------------------------------------------

#define KIT_COMBINE9(Kit, T0, T1, T2, T3, T4, T5, T6, T7, T8) \
    \
    struct Kit \
        : \
        public T0, \
        public T1, \
        public T2, \
        public T3, \
        public T4, \
        public T5, \
        public T6, \
        public T7, \
        public T8 \
    { \
        template <typename OtherKit> \
        sysinline Kit(const OtherKit& otherKit) \
            : \
            T0(otherKit), \
            T1(otherKit), \
            T2(otherKit), \
            T3(otherKit), \
            T4(otherKit), \
            T5(otherKit), \
            T6(otherKit), \
            T7(otherKit), \
            T8(otherKit) \
        { \
        } \
        \
        template <typename OldKit, typename NewKit> \
        sysinline Kit(const OldKit& oldKit, KitReplaceConstructor, const NewKit& newKit) \
            : \
            T0(oldKit, KitReplaceConstructor(), newKit), \
            T1(oldKit, KitReplaceConstructor(), newKit), \
            T2(oldKit, KitReplaceConstructor(), newKit), \
            T3(oldKit, KitReplaceConstructor(), newKit), \
            T4(oldKit, KitReplaceConstructor(), newKit), \
            T5(oldKit, KitReplaceConstructor(), newKit), \
            T6(oldKit, KitReplaceConstructor(), newKit), \
            T7(oldKit, KitReplaceConstructor(), newKit), \
            T8(oldKit, KitReplaceConstructor(), newKit) \
        { \
        } \
        \
        template <typename Q0, typename Q1, typename Q2, typename Q3, typename Q4, typename Q5, typename Q6, typename Q7, typename Q8> \
        sysinline Kit \
        ( \
            const Q0& q0, \
            const Q1& q1, \
            const Q2& q2, \
            const Q3& q3, \
            const Q4& q4, \
            const Q5& q5, \
            const Q6& q6, \
            const Q7& q7, \
            const Q8& q8, \
            KitCombineConstructor \
        ) \
            : \
            T0(q0), \
            T1(q1), \
            T2(q2), \
            T3(q3), \
            T4(q4), \
            T5(q5), \
            T6(q6), \
            T7(q7), \
            T8(q8) \
        { \
        } \
    }

//----------------------------------------------------------------

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8>
KIT_COMBINE9(KitCombine9, T0, T1, T2, T3, T4, T5, T6, T7, T8);

//----------------------------------------------------------------

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8>
sysinline KitCombine9<T0, T1, T2, T3, T4, T5, T6, T7, T8> kitCombine(const T0& v0, const T1& v1, const T2& v2, const T3& v3, const T4& v4, const T5& v5, const T6& v6, const T7& v7, const T8& v8)
    {return KitCombine9<T0, T1, T2, T3, T4, T5, T6, T7, T8>(v0, v1, v2, v3, v4, v5, v6, v7, v8, KitCombineConstructor());}

//----------------------------------------------------------------

#define KIT_COMBINE10(Kit, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9) \
    \
    struct Kit \
        : \
        public T0, \
        public T1, \
        public T2, \
        public T3, \
        public T4, \
        public T5, \
        public T6, \
        public T7, \
        public T8, \
        public T9 \
    { \
        template <typename OtherKit> \
        sysinline Kit(const OtherKit& otherKit) \
            : \
            T0(otherKit), \
            T1(otherKit), \
            T2(otherKit), \
            T3(otherKit), \
            T4(otherKit), \
            T5(otherKit), \
            T6(otherKit), \
            T7(otherKit), \
            T8(otherKit), \
            T9(otherKit) \
        { \
        } \
        \
        template <typename OldKit, typename NewKit> \
        sysinline Kit(const OldKit& oldKit, KitReplaceConstructor, const NewKit& newKit) \
            : \
            T0(oldKit, KitReplaceConstructor(), newKit), \
            T1(oldKit, KitReplaceConstructor(), newKit), \
            T2(oldKit, KitReplaceConstructor(), newKit), \
            T3(oldKit, KitReplaceConstructor(), newKit), \
            T4(oldKit, KitReplaceConstructor(), newKit), \
            T5(oldKit, KitReplaceConstructor(), newKit), \
            T6(oldKit, KitReplaceConstructor(), newKit), \
            T7(oldKit, KitReplaceConstructor(), newKit), \
            T8(oldKit, KitReplaceConstructor(), newKit), \
            T9(oldKit, KitReplaceConstructor(), newKit) \
        { \
        } \
        \
        template <typename Q0, typename Q1, typename Q2, typename Q3, typename Q4, typename Q5, typename Q6, typename Q7, typename Q8, typename Q9> \
        sysinline Kit \
        ( \
            const Q0& q0, \
            const Q1& q1, \
            const Q2& q2, \
            const Q3& q3, \
            const Q4& q4, \
            const Q5& q5, \
            const Q6& q6, \
            const Q7& q7, \
            const Q8& q8, \
            const Q9& q9, \
            KitCombineConstructor \
        ) \
            : \
            T0(q0), \
            T1(q1), \
            T2(q2), \
            T3(q3), \
            T4(q4), \
            T5(q5), \
            T6(q6), \
            T7(q7), \
            T8(q8), \
            T9(q9) \
        { \
        } \
    }

//----------------------------------------------------------------

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9>
KIT_COMBINE10(KitCombine10, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9);

//----------------------------------------------------------------

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9>
sysinline KitCombine10<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> kitCombine(const T0& v0, const T1& v1, const T2& v2, const T3& v3, const T4& v4, const T5& v5, const T6& v6, const T7& v7, const T8& v8, const T9& v9)
    {return KitCombine10<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, KitCombineConstructor());}

//----------------------------------------------------------------

#define KIT_COMBINE11(Kit, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10) \
    \
    struct Kit \
        : \
        public T0, \
        public T1, \
        public T2, \
        public T3, \
        public T4, \
        public T5, \
        public T6, \
        public T7, \
        public T8, \
        public T9, \
        public T10 \
    { \
        template <typename OtherKit> \
        sysinline Kit(const OtherKit& otherKit) \
            : \
            T0(otherKit), \
            T1(otherKit), \
            T2(otherKit), \
            T3(otherKit), \
            T4(otherKit), \
            T5(otherKit), \
            T6(otherKit), \
            T7(otherKit), \
            T8(otherKit), \
            T9(otherKit), \
            T10(otherKit) \
        { \
        } \
        \
        template <typename OldKit, typename NewKit> \
        sysinline Kit(const OldKit& oldKit, KitReplaceConstructor, const NewKit& newKit) \
            : \
            T0(oldKit, KitReplaceConstructor(), newKit), \
            T1(oldKit, KitReplaceConstructor(), newKit), \
            T2(oldKit, KitReplaceConstructor(), newKit), \
            T3(oldKit, KitReplaceConstructor(), newKit), \
            T4(oldKit, KitReplaceConstructor(), newKit), \
            T5(oldKit, KitReplaceConstructor(), newKit), \
            T6(oldKit, KitReplaceConstructor(), newKit), \
            T7(oldKit, KitReplaceConstructor(), newKit), \
            T8(oldKit, KitReplaceConstructor(), newKit), \
            T9(oldKit, KitReplaceConstructor(), newKit), \
            T10(oldKit, KitReplaceConstructor(), newKit) \
        { \
        } \
        \
        template <typename Q0, typename Q1, typename Q2, typename Q3, typename Q4, typename Q5, typename Q6, typename Q7, typename Q8, typename Q9, typename Q10> \
        sysinline Kit \
        ( \
            const Q0& q0, \
            const Q1& q1, \
            const Q2& q2, \
            const Q3& q3, \
            const Q4& q4, \
            const Q5& q5, \
            const Q6& q6, \
            const Q7& q7, \
            const Q8& q8, \
            const Q9& q9, \
            const Q10& q10, \
            KitCombineConstructor \
        ) \
            : \
            T0(q0), \
            T1(q1), \
            T2(q2), \
            T3(q3), \
            T4(q4), \
            T5(q5), \
            T6(q6), \
            T7(q7), \
            T8(q8), \
            T9(q9), \
            T10(q10) \
        { \
        } \
    }

//----------------------------------------------------------------

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10>
KIT_COMBINE11(KitCombine11, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10);

//----------------------------------------------------------------

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10>
sysinline KitCombine11<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10> kitCombine(const T0& v0, const T1& v1, const T2& v2, const T3& v3, const T4& v4, const T5& v5, const T6& v6, const T7& v7, const T8& v8, const T9& v9, const T10& v10)
    {return KitCombine11<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, KitCombineConstructor());}

//----------------------------------------------------------------

#define KIT_COMBINE12(Kit, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11) \
    \
    struct Kit \
        : \
        public T0, \
        public T1, \
        public T2, \
        public T3, \
        public T4, \
        public T5, \
        public T6, \
        public T7, \
        public T8, \
        public T9, \
        public T10, \
        public T11 \
    { \
        template <typename OtherKit> \
        sysinline Kit(const OtherKit& otherKit) \
            : \
            T0(otherKit), \
            T1(otherKit), \
            T2(otherKit), \
            T3(otherKit), \
            T4(otherKit), \
            T5(otherKit), \
            T6(otherKit), \
            T7(otherKit), \
            T8(otherKit), \
            T9(otherKit), \
            T10(otherKit), \
            T11(otherKit) \
        { \
        } \
        \
        template <typename OldKit, typename NewKit> \
        sysinline Kit(const OldKit& oldKit, KitReplaceConstructor, const NewKit& newKit) \
            : \
            T0(oldKit, KitReplaceConstructor(), newKit), \
            T1(oldKit, KitReplaceConstructor(), newKit), \
            T2(oldKit, KitReplaceConstructor(), newKit), \
            T3(oldKit, KitReplaceConstructor(), newKit), \
            T4(oldKit, KitReplaceConstructor(), newKit), \
            T5(oldKit, KitReplaceConstructor(), newKit), \
            T6(oldKit, KitReplaceConstructor(), newKit), \
            T7(oldKit, KitReplaceConstructor(), newKit), \
            T8(oldKit, KitReplaceConstructor(), newKit), \
            T9(oldKit, KitReplaceConstructor(), newKit), \
            T10(oldKit, KitReplaceConstructor(), newKit), \
            T11(oldKit, KitReplaceConstructor(), newKit) \
        { \
        } \
        \
        template <typename Q0, typename Q1, typename Q2, typename Q3, typename Q4, typename Q5, typename Q6, typename Q7, typename Q8, typename Q9, typename Q10, typename Q11> \
        sysinline Kit \
        ( \
            const Q0& q0, \
            const Q1& q1, \
            const Q2& q2, \
            const Q3& q3, \
            const Q4& q4, \
            const Q5& q5, \
            const Q6& q6, \
            const Q7& q7, \
            const Q8& q8, \
            const Q9& q9, \
            const Q10& q10, \
            const Q11& q11, \
            KitCombineConstructor \
        ) \
            : \
            T0(q0), \
            T1(q1), \
            T2(q2), \
            T3(q3), \
            T4(q4), \
            T5(q5), \
            T6(q6), \
            T7(q7), \
            T8(q8), \
            T9(q9), \
            T10(q10), \
            T11(q11) \
        { \
        } \
    }

//----------------------------------------------------------------

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11>
KIT_COMBINE12(KitCombine12, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11);

//----------------------------------------------------------------

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11>
sysinline KitCombine12<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11> kitCombine(const T0& v0, const T1& v1, const T2& v2, const T3& v3, const T4& v4, const T5& v5, const T6& v6, const T7& v7, const T8& v8, const T9& v9, const T10& v10, const T11& v11)
    {return KitCombine12<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11>(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, KitCombineConstructor());}
