//----------------------------------------------------------------

#define KIT__CREATE2(Kit, Type0, name0, Type1, name1) \
    \
    struct Kit \
        : \
        Kit_FieldTag<struct name0##_Tag>, \
        Kit_FieldTag<struct name1##_Tag> \
    { \
        \
        Type0 name0; \
        Type1 name1; \
        \
        sysinline Kit \
        ( \
            ParamType<Type0>::T name0, \
            ParamType<Type1>::T name1 \
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
        sysinline Kit(const OldKit& oldKit, Kit_ReplaceConstructor, const NewKit& newKit) \
            : \
            name0(Kit_Replacer<OldKit, NewKit, name0##_Tag>::func(&oldKit, &newKit)->name0), \
            name1(Kit_Replacer<OldKit, NewKit, name1##_Tag>::func(&oldKit, &newKit)->name1) \
        { \
        } \
    }

#define KIT_CREATE2(Kit, Type0, name0, Type1, name1) \
    KIT__CREATE2(Kit, Type0, name0, Type1, name1)


//----------------------------------------------------------------

#define KIT__CREATE3(Kit, Type0, name0, Type1, name1, Type2, name2) \
    \
    struct Kit \
        : \
        Kit_FieldTag<struct name0##_Tag>, \
        Kit_FieldTag<struct name1##_Tag>, \
        Kit_FieldTag<struct name2##_Tag> \
    { \
        \
        Type0 name0; \
        Type1 name1; \
        Type2 name2; \
        \
        sysinline Kit \
        ( \
            ParamType<Type0>::T name0, \
            ParamType<Type1>::T name1, \
            ParamType<Type2>::T name2 \
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
        sysinline Kit(const OldKit& oldKit, Kit_ReplaceConstructor, const NewKit& newKit) \
            : \
            name0(Kit_Replacer<OldKit, NewKit, name0##_Tag>::func(&oldKit, &newKit)->name0), \
            name1(Kit_Replacer<OldKit, NewKit, name1##_Tag>::func(&oldKit, &newKit)->name1), \
            name2(Kit_Replacer<OldKit, NewKit, name2##_Tag>::func(&oldKit, &newKit)->name2) \
        { \
        } \
    }

#define KIT_CREATE3(Kit, Type0, name0, Type1, name1, Type2, name2) \
    KIT__CREATE3(Kit, Type0, name0, Type1, name1, Type2, name2)


//----------------------------------------------------------------

#define KIT__CREATE4(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3) \
    \
    struct Kit \
        : \
        Kit_FieldTag<struct name0##_Tag>, \
        Kit_FieldTag<struct name1##_Tag>, \
        Kit_FieldTag<struct name2##_Tag>, \
        Kit_FieldTag<struct name3##_Tag> \
    { \
        \
        Type0 name0; \
        Type1 name1; \
        Type2 name2; \
        Type3 name3; \
        \
        sysinline Kit \
        ( \
            ParamType<Type0>::T name0, \
            ParamType<Type1>::T name1, \
            ParamType<Type2>::T name2, \
            ParamType<Type3>::T name3 \
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
        sysinline Kit(const OldKit& oldKit, Kit_ReplaceConstructor, const NewKit& newKit) \
            : \
            name0(Kit_Replacer<OldKit, NewKit, name0##_Tag>::func(&oldKit, &newKit)->name0), \
            name1(Kit_Replacer<OldKit, NewKit, name1##_Tag>::func(&oldKit, &newKit)->name1), \
            name2(Kit_Replacer<OldKit, NewKit, name2##_Tag>::func(&oldKit, &newKit)->name2), \
            name3(Kit_Replacer<OldKit, NewKit, name3##_Tag>::func(&oldKit, &newKit)->name3) \
        { \
        } \
    }

#define KIT_CREATE4(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3) \
    KIT__CREATE4(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3)


//----------------------------------------------------------------

#define KIT__CREATE5(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4) \
    \
    struct Kit \
        : \
        Kit_FieldTag<struct name0##_Tag>, \
        Kit_FieldTag<struct name1##_Tag>, \
        Kit_FieldTag<struct name2##_Tag>, \
        Kit_FieldTag<struct name3##_Tag>, \
        Kit_FieldTag<struct name4##_Tag> \
    { \
        \
        Type0 name0; \
        Type1 name1; \
        Type2 name2; \
        Type3 name3; \
        Type4 name4; \
        \
        sysinline Kit \
        ( \
            ParamType<Type0>::T name0, \
            ParamType<Type1>::T name1, \
            ParamType<Type2>::T name2, \
            ParamType<Type3>::T name3, \
            ParamType<Type4>::T name4 \
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
        sysinline Kit(const OldKit& oldKit, Kit_ReplaceConstructor, const NewKit& newKit) \
            : \
            name0(Kit_Replacer<OldKit, NewKit, name0##_Tag>::func(&oldKit, &newKit)->name0), \
            name1(Kit_Replacer<OldKit, NewKit, name1##_Tag>::func(&oldKit, &newKit)->name1), \
            name2(Kit_Replacer<OldKit, NewKit, name2##_Tag>::func(&oldKit, &newKit)->name2), \
            name3(Kit_Replacer<OldKit, NewKit, name3##_Tag>::func(&oldKit, &newKit)->name3), \
            name4(Kit_Replacer<OldKit, NewKit, name4##_Tag>::func(&oldKit, &newKit)->name4) \
        { \
        } \
    }

#define KIT_CREATE5(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4) \
    KIT__CREATE5(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4)


//----------------------------------------------------------------

#define KIT__CREATE6(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5) \
    \
    struct Kit \
        : \
        Kit_FieldTag<struct name0##_Tag>, \
        Kit_FieldTag<struct name1##_Tag>, \
        Kit_FieldTag<struct name2##_Tag>, \
        Kit_FieldTag<struct name3##_Tag>, \
        Kit_FieldTag<struct name4##_Tag>, \
        Kit_FieldTag<struct name5##_Tag> \
    { \
        \
        Type0 name0; \
        Type1 name1; \
        Type2 name2; \
        Type3 name3; \
        Type4 name4; \
        Type5 name5; \
        \
        sysinline Kit \
        ( \
            ParamType<Type0>::T name0, \
            ParamType<Type1>::T name1, \
            ParamType<Type2>::T name2, \
            ParamType<Type3>::T name3, \
            ParamType<Type4>::T name4, \
            ParamType<Type5>::T name5 \
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
        sysinline Kit(const OldKit& oldKit, Kit_ReplaceConstructor, const NewKit& newKit) \
            : \
            name0(Kit_Replacer<OldKit, NewKit, name0##_Tag>::func(&oldKit, &newKit)->name0), \
            name1(Kit_Replacer<OldKit, NewKit, name1##_Tag>::func(&oldKit, &newKit)->name1), \
            name2(Kit_Replacer<OldKit, NewKit, name2##_Tag>::func(&oldKit, &newKit)->name2), \
            name3(Kit_Replacer<OldKit, NewKit, name3##_Tag>::func(&oldKit, &newKit)->name3), \
            name4(Kit_Replacer<OldKit, NewKit, name4##_Tag>::func(&oldKit, &newKit)->name4), \
            name5(Kit_Replacer<OldKit, NewKit, name5##_Tag>::func(&oldKit, &newKit)->name5) \
        { \
        } \
    }

#define KIT_CREATE6(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5) \
    KIT__CREATE6(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5)


//----------------------------------------------------------------

#define KIT__CREATE7(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6) \
    \
    struct Kit \
        : \
        Kit_FieldTag<struct name0##_Tag>, \
        Kit_FieldTag<struct name1##_Tag>, \
        Kit_FieldTag<struct name2##_Tag>, \
        Kit_FieldTag<struct name3##_Tag>, \
        Kit_FieldTag<struct name4##_Tag>, \
        Kit_FieldTag<struct name5##_Tag>, \
        Kit_FieldTag<struct name6##_Tag> \
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
        sysinline Kit \
        ( \
            ParamType<Type0>::T name0, \
            ParamType<Type1>::T name1, \
            ParamType<Type2>::T name2, \
            ParamType<Type3>::T name3, \
            ParamType<Type4>::T name4, \
            ParamType<Type5>::T name5, \
            ParamType<Type6>::T name6 \
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
        sysinline Kit(const OldKit& oldKit, Kit_ReplaceConstructor, const NewKit& newKit) \
            : \
            name0(Kit_Replacer<OldKit, NewKit, name0##_Tag>::func(&oldKit, &newKit)->name0), \
            name1(Kit_Replacer<OldKit, NewKit, name1##_Tag>::func(&oldKit, &newKit)->name1), \
            name2(Kit_Replacer<OldKit, NewKit, name2##_Tag>::func(&oldKit, &newKit)->name2), \
            name3(Kit_Replacer<OldKit, NewKit, name3##_Tag>::func(&oldKit, &newKit)->name3), \
            name4(Kit_Replacer<OldKit, NewKit, name4##_Tag>::func(&oldKit, &newKit)->name4), \
            name5(Kit_Replacer<OldKit, NewKit, name5##_Tag>::func(&oldKit, &newKit)->name5), \
            name6(Kit_Replacer<OldKit, NewKit, name6##_Tag>::func(&oldKit, &newKit)->name6) \
        { \
        } \
    }

#define KIT_CREATE7(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6) \
    KIT__CREATE7(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6)


//----------------------------------------------------------------

#define KIT__CREATE8(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7) \
    \
    struct Kit \
        : \
        Kit_FieldTag<struct name0##_Tag>, \
        Kit_FieldTag<struct name1##_Tag>, \
        Kit_FieldTag<struct name2##_Tag>, \
        Kit_FieldTag<struct name3##_Tag>, \
        Kit_FieldTag<struct name4##_Tag>, \
        Kit_FieldTag<struct name5##_Tag>, \
        Kit_FieldTag<struct name6##_Tag>, \
        Kit_FieldTag<struct name7##_Tag> \
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
        sysinline Kit \
        ( \
            ParamType<Type0>::T name0, \
            ParamType<Type1>::T name1, \
            ParamType<Type2>::T name2, \
            ParamType<Type3>::T name3, \
            ParamType<Type4>::T name4, \
            ParamType<Type5>::T name5, \
            ParamType<Type6>::T name6, \
            ParamType<Type7>::T name7 \
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
        sysinline Kit(const OldKit& oldKit, Kit_ReplaceConstructor, const NewKit& newKit) \
            : \
            name0(Kit_Replacer<OldKit, NewKit, name0##_Tag>::func(&oldKit, &newKit)->name0), \
            name1(Kit_Replacer<OldKit, NewKit, name1##_Tag>::func(&oldKit, &newKit)->name1), \
            name2(Kit_Replacer<OldKit, NewKit, name2##_Tag>::func(&oldKit, &newKit)->name2), \
            name3(Kit_Replacer<OldKit, NewKit, name3##_Tag>::func(&oldKit, &newKit)->name3), \
            name4(Kit_Replacer<OldKit, NewKit, name4##_Tag>::func(&oldKit, &newKit)->name4), \
            name5(Kit_Replacer<OldKit, NewKit, name5##_Tag>::func(&oldKit, &newKit)->name5), \
            name6(Kit_Replacer<OldKit, NewKit, name6##_Tag>::func(&oldKit, &newKit)->name6), \
            name7(Kit_Replacer<OldKit, NewKit, name7##_Tag>::func(&oldKit, &newKit)->name7) \
        { \
        } \
    }

#define KIT_CREATE8(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7) \
    KIT__CREATE8(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7)


//----------------------------------------------------------------

#define KIT__CREATE9(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8) \
    \
    struct Kit \
        : \
        Kit_FieldTag<struct name0##_Tag>, \
        Kit_FieldTag<struct name1##_Tag>, \
        Kit_FieldTag<struct name2##_Tag>, \
        Kit_FieldTag<struct name3##_Tag>, \
        Kit_FieldTag<struct name4##_Tag>, \
        Kit_FieldTag<struct name5##_Tag>, \
        Kit_FieldTag<struct name6##_Tag>, \
        Kit_FieldTag<struct name7##_Tag>, \
        Kit_FieldTag<struct name8##_Tag> \
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
        sysinline Kit \
        ( \
            ParamType<Type0>::T name0, \
            ParamType<Type1>::T name1, \
            ParamType<Type2>::T name2, \
            ParamType<Type3>::T name3, \
            ParamType<Type4>::T name4, \
            ParamType<Type5>::T name5, \
            ParamType<Type6>::T name6, \
            ParamType<Type7>::T name7, \
            ParamType<Type8>::T name8 \
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
        sysinline Kit(const OldKit& oldKit, Kit_ReplaceConstructor, const NewKit& newKit) \
            : \
            name0(Kit_Replacer<OldKit, NewKit, name0##_Tag>::func(&oldKit, &newKit)->name0), \
            name1(Kit_Replacer<OldKit, NewKit, name1##_Tag>::func(&oldKit, &newKit)->name1), \
            name2(Kit_Replacer<OldKit, NewKit, name2##_Tag>::func(&oldKit, &newKit)->name2), \
            name3(Kit_Replacer<OldKit, NewKit, name3##_Tag>::func(&oldKit, &newKit)->name3), \
            name4(Kit_Replacer<OldKit, NewKit, name4##_Tag>::func(&oldKit, &newKit)->name4), \
            name5(Kit_Replacer<OldKit, NewKit, name5##_Tag>::func(&oldKit, &newKit)->name5), \
            name6(Kit_Replacer<OldKit, NewKit, name6##_Tag>::func(&oldKit, &newKit)->name6), \
            name7(Kit_Replacer<OldKit, NewKit, name7##_Tag>::func(&oldKit, &newKit)->name7), \
            name8(Kit_Replacer<OldKit, NewKit, name8##_Tag>::func(&oldKit, &newKit)->name8) \
        { \
        } \
    }

#define KIT_CREATE9(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8) \
    KIT__CREATE9(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8)


//----------------------------------------------------------------

#define KIT__CREATE10(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9) \
    \
    struct Kit \
        : \
        Kit_FieldTag<struct name0##_Tag>, \
        Kit_FieldTag<struct name1##_Tag>, \
        Kit_FieldTag<struct name2##_Tag>, \
        Kit_FieldTag<struct name3##_Tag>, \
        Kit_FieldTag<struct name4##_Tag>, \
        Kit_FieldTag<struct name5##_Tag>, \
        Kit_FieldTag<struct name6##_Tag>, \
        Kit_FieldTag<struct name7##_Tag>, \
        Kit_FieldTag<struct name8##_Tag>, \
        Kit_FieldTag<struct name9##_Tag> \
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
        sysinline Kit \
        ( \
            ParamType<Type0>::T name0, \
            ParamType<Type1>::T name1, \
            ParamType<Type2>::T name2, \
            ParamType<Type3>::T name3, \
            ParamType<Type4>::T name4, \
            ParamType<Type5>::T name5, \
            ParamType<Type6>::T name6, \
            ParamType<Type7>::T name7, \
            ParamType<Type8>::T name8, \
            ParamType<Type9>::T name9 \
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
        sysinline Kit(const OldKit& oldKit, Kit_ReplaceConstructor, const NewKit& newKit) \
            : \
            name0(Kit_Replacer<OldKit, NewKit, name0##_Tag>::func(&oldKit, &newKit)->name0), \
            name1(Kit_Replacer<OldKit, NewKit, name1##_Tag>::func(&oldKit, &newKit)->name1), \
            name2(Kit_Replacer<OldKit, NewKit, name2##_Tag>::func(&oldKit, &newKit)->name2), \
            name3(Kit_Replacer<OldKit, NewKit, name3##_Tag>::func(&oldKit, &newKit)->name3), \
            name4(Kit_Replacer<OldKit, NewKit, name4##_Tag>::func(&oldKit, &newKit)->name4), \
            name5(Kit_Replacer<OldKit, NewKit, name5##_Tag>::func(&oldKit, &newKit)->name5), \
            name6(Kit_Replacer<OldKit, NewKit, name6##_Tag>::func(&oldKit, &newKit)->name6), \
            name7(Kit_Replacer<OldKit, NewKit, name7##_Tag>::func(&oldKit, &newKit)->name7), \
            name8(Kit_Replacer<OldKit, NewKit, name8##_Tag>::func(&oldKit, &newKit)->name8), \
            name9(Kit_Replacer<OldKit, NewKit, name9##_Tag>::func(&oldKit, &newKit)->name9) \
        { \
        } \
    }

#define KIT_CREATE10(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9) \
    KIT__CREATE10(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9)


//----------------------------------------------------------------

#define KIT__CREATE11(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10) \
    \
    struct Kit \
        : \
        Kit_FieldTag<struct name0##_Tag>, \
        Kit_FieldTag<struct name1##_Tag>, \
        Kit_FieldTag<struct name2##_Tag>, \
        Kit_FieldTag<struct name3##_Tag>, \
        Kit_FieldTag<struct name4##_Tag>, \
        Kit_FieldTag<struct name5##_Tag>, \
        Kit_FieldTag<struct name6##_Tag>, \
        Kit_FieldTag<struct name7##_Tag>, \
        Kit_FieldTag<struct name8##_Tag>, \
        Kit_FieldTag<struct name9##_Tag>, \
        Kit_FieldTag<struct name10##_Tag> \
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
        sysinline Kit \
        ( \
            ParamType<Type0>::T name0, \
            ParamType<Type1>::T name1, \
            ParamType<Type2>::T name2, \
            ParamType<Type3>::T name3, \
            ParamType<Type4>::T name4, \
            ParamType<Type5>::T name5, \
            ParamType<Type6>::T name6, \
            ParamType<Type7>::T name7, \
            ParamType<Type8>::T name8, \
            ParamType<Type9>::T name9, \
            ParamType<Type10>::T name10 \
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
        sysinline Kit(const OldKit& oldKit, Kit_ReplaceConstructor, const NewKit& newKit) \
            : \
            name0(Kit_Replacer<OldKit, NewKit, name0##_Tag>::func(&oldKit, &newKit)->name0), \
            name1(Kit_Replacer<OldKit, NewKit, name1##_Tag>::func(&oldKit, &newKit)->name1), \
            name2(Kit_Replacer<OldKit, NewKit, name2##_Tag>::func(&oldKit, &newKit)->name2), \
            name3(Kit_Replacer<OldKit, NewKit, name3##_Tag>::func(&oldKit, &newKit)->name3), \
            name4(Kit_Replacer<OldKit, NewKit, name4##_Tag>::func(&oldKit, &newKit)->name4), \
            name5(Kit_Replacer<OldKit, NewKit, name5##_Tag>::func(&oldKit, &newKit)->name5), \
            name6(Kit_Replacer<OldKit, NewKit, name6##_Tag>::func(&oldKit, &newKit)->name6), \
            name7(Kit_Replacer<OldKit, NewKit, name7##_Tag>::func(&oldKit, &newKit)->name7), \
            name8(Kit_Replacer<OldKit, NewKit, name8##_Tag>::func(&oldKit, &newKit)->name8), \
            name9(Kit_Replacer<OldKit, NewKit, name9##_Tag>::func(&oldKit, &newKit)->name9), \
            name10(Kit_Replacer<OldKit, NewKit, name10##_Tag>::func(&oldKit, &newKit)->name10) \
        { \
        } \
    }

#define KIT_CREATE11(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10) \
    KIT__CREATE11(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10)


//----------------------------------------------------------------

#define KIT__CREATE12(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11) \
    \
    struct Kit \
        : \
        Kit_FieldTag<struct name0##_Tag>, \
        Kit_FieldTag<struct name1##_Tag>, \
        Kit_FieldTag<struct name2##_Tag>, \
        Kit_FieldTag<struct name3##_Tag>, \
        Kit_FieldTag<struct name4##_Tag>, \
        Kit_FieldTag<struct name5##_Tag>, \
        Kit_FieldTag<struct name6##_Tag>, \
        Kit_FieldTag<struct name7##_Tag>, \
        Kit_FieldTag<struct name8##_Tag>, \
        Kit_FieldTag<struct name9##_Tag>, \
        Kit_FieldTag<struct name10##_Tag>, \
        Kit_FieldTag<struct name11##_Tag> \
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
        sysinline Kit \
        ( \
            ParamType<Type0>::T name0, \
            ParamType<Type1>::T name1, \
            ParamType<Type2>::T name2, \
            ParamType<Type3>::T name3, \
            ParamType<Type4>::T name4, \
            ParamType<Type5>::T name5, \
            ParamType<Type6>::T name6, \
            ParamType<Type7>::T name7, \
            ParamType<Type8>::T name8, \
            ParamType<Type9>::T name9, \
            ParamType<Type10>::T name10, \
            ParamType<Type11>::T name11 \
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
        sysinline Kit(const OldKit& oldKit, Kit_ReplaceConstructor, const NewKit& newKit) \
            : \
            name0(Kit_Replacer<OldKit, NewKit, name0##_Tag>::func(&oldKit, &newKit)->name0), \
            name1(Kit_Replacer<OldKit, NewKit, name1##_Tag>::func(&oldKit, &newKit)->name1), \
            name2(Kit_Replacer<OldKit, NewKit, name2##_Tag>::func(&oldKit, &newKit)->name2), \
            name3(Kit_Replacer<OldKit, NewKit, name3##_Tag>::func(&oldKit, &newKit)->name3), \
            name4(Kit_Replacer<OldKit, NewKit, name4##_Tag>::func(&oldKit, &newKit)->name4), \
            name5(Kit_Replacer<OldKit, NewKit, name5##_Tag>::func(&oldKit, &newKit)->name5), \
            name6(Kit_Replacer<OldKit, NewKit, name6##_Tag>::func(&oldKit, &newKit)->name6), \
            name7(Kit_Replacer<OldKit, NewKit, name7##_Tag>::func(&oldKit, &newKit)->name7), \
            name8(Kit_Replacer<OldKit, NewKit, name8##_Tag>::func(&oldKit, &newKit)->name8), \
            name9(Kit_Replacer<OldKit, NewKit, name9##_Tag>::func(&oldKit, &newKit)->name9), \
            name10(Kit_Replacer<OldKit, NewKit, name10##_Tag>::func(&oldKit, &newKit)->name10), \
            name11(Kit_Replacer<OldKit, NewKit, name11##_Tag>::func(&oldKit, &newKit)->name11) \
        { \
        } \
    }

#define KIT_CREATE12(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11) \
    KIT__CREATE12(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11)


//----------------------------------------------------------------

#define KIT__CREATE13(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, Type12, name12) \
    \
    struct Kit \
        : \
        Kit_FieldTag<struct name0##_Tag>, \
        Kit_FieldTag<struct name1##_Tag>, \
        Kit_FieldTag<struct name2##_Tag>, \
        Kit_FieldTag<struct name3##_Tag>, \
        Kit_FieldTag<struct name4##_Tag>, \
        Kit_FieldTag<struct name5##_Tag>, \
        Kit_FieldTag<struct name6##_Tag>, \
        Kit_FieldTag<struct name7##_Tag>, \
        Kit_FieldTag<struct name8##_Tag>, \
        Kit_FieldTag<struct name9##_Tag>, \
        Kit_FieldTag<struct name10##_Tag>, \
        Kit_FieldTag<struct name11##_Tag>, \
        Kit_FieldTag<struct name12##_Tag> \
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
        sysinline Kit \
        ( \
            ParamType<Type0>::T name0, \
            ParamType<Type1>::T name1, \
            ParamType<Type2>::T name2, \
            ParamType<Type3>::T name3, \
            ParamType<Type4>::T name4, \
            ParamType<Type5>::T name5, \
            ParamType<Type6>::T name6, \
            ParamType<Type7>::T name7, \
            ParamType<Type8>::T name8, \
            ParamType<Type9>::T name9, \
            ParamType<Type10>::T name10, \
            ParamType<Type11>::T name11, \
            ParamType<Type12>::T name12 \
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
        sysinline Kit(const OldKit& oldKit, Kit_ReplaceConstructor, const NewKit& newKit) \
            : \
            name0(Kit_Replacer<OldKit, NewKit, name0##_Tag>::func(&oldKit, &newKit)->name0), \
            name1(Kit_Replacer<OldKit, NewKit, name1##_Tag>::func(&oldKit, &newKit)->name1), \
            name2(Kit_Replacer<OldKit, NewKit, name2##_Tag>::func(&oldKit, &newKit)->name2), \
            name3(Kit_Replacer<OldKit, NewKit, name3##_Tag>::func(&oldKit, &newKit)->name3), \
            name4(Kit_Replacer<OldKit, NewKit, name4##_Tag>::func(&oldKit, &newKit)->name4), \
            name5(Kit_Replacer<OldKit, NewKit, name5##_Tag>::func(&oldKit, &newKit)->name5), \
            name6(Kit_Replacer<OldKit, NewKit, name6##_Tag>::func(&oldKit, &newKit)->name6), \
            name7(Kit_Replacer<OldKit, NewKit, name7##_Tag>::func(&oldKit, &newKit)->name7), \
            name8(Kit_Replacer<OldKit, NewKit, name8##_Tag>::func(&oldKit, &newKit)->name8), \
            name9(Kit_Replacer<OldKit, NewKit, name9##_Tag>::func(&oldKit, &newKit)->name9), \
            name10(Kit_Replacer<OldKit, NewKit, name10##_Tag>::func(&oldKit, &newKit)->name10), \
            name11(Kit_Replacer<OldKit, NewKit, name11##_Tag>::func(&oldKit, &newKit)->name11), \
            name12(Kit_Replacer<OldKit, NewKit, name12##_Tag>::func(&oldKit, &newKit)->name12) \
        { \
        } \
    }

#define KIT_CREATE13(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, Type12, name12) \
    KIT__CREATE13(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, Type12, name12)


//----------------------------------------------------------------

#define KIT__CREATE14(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, Type12, name12, Type13, name13) \
    \
    struct Kit \
        : \
        Kit_FieldTag<struct name0##_Tag>, \
        Kit_FieldTag<struct name1##_Tag>, \
        Kit_FieldTag<struct name2##_Tag>, \
        Kit_FieldTag<struct name3##_Tag>, \
        Kit_FieldTag<struct name4##_Tag>, \
        Kit_FieldTag<struct name5##_Tag>, \
        Kit_FieldTag<struct name6##_Tag>, \
        Kit_FieldTag<struct name7##_Tag>, \
        Kit_FieldTag<struct name8##_Tag>, \
        Kit_FieldTag<struct name9##_Tag>, \
        Kit_FieldTag<struct name10##_Tag>, \
        Kit_FieldTag<struct name11##_Tag>, \
        Kit_FieldTag<struct name12##_Tag>, \
        Kit_FieldTag<struct name13##_Tag> \
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
        sysinline Kit \
        ( \
            ParamType<Type0>::T name0, \
            ParamType<Type1>::T name1, \
            ParamType<Type2>::T name2, \
            ParamType<Type3>::T name3, \
            ParamType<Type4>::T name4, \
            ParamType<Type5>::T name5, \
            ParamType<Type6>::T name6, \
            ParamType<Type7>::T name7, \
            ParamType<Type8>::T name8, \
            ParamType<Type9>::T name9, \
            ParamType<Type10>::T name10, \
            ParamType<Type11>::T name11, \
            ParamType<Type12>::T name12, \
            ParamType<Type13>::T name13 \
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
        sysinline Kit(const OldKit& oldKit, Kit_ReplaceConstructor, const NewKit& newKit) \
            : \
            name0(Kit_Replacer<OldKit, NewKit, name0##_Tag>::func(&oldKit, &newKit)->name0), \
            name1(Kit_Replacer<OldKit, NewKit, name1##_Tag>::func(&oldKit, &newKit)->name1), \
            name2(Kit_Replacer<OldKit, NewKit, name2##_Tag>::func(&oldKit, &newKit)->name2), \
            name3(Kit_Replacer<OldKit, NewKit, name3##_Tag>::func(&oldKit, &newKit)->name3), \
            name4(Kit_Replacer<OldKit, NewKit, name4##_Tag>::func(&oldKit, &newKit)->name4), \
            name5(Kit_Replacer<OldKit, NewKit, name5##_Tag>::func(&oldKit, &newKit)->name5), \
            name6(Kit_Replacer<OldKit, NewKit, name6##_Tag>::func(&oldKit, &newKit)->name6), \
            name7(Kit_Replacer<OldKit, NewKit, name7##_Tag>::func(&oldKit, &newKit)->name7), \
            name8(Kit_Replacer<OldKit, NewKit, name8##_Tag>::func(&oldKit, &newKit)->name8), \
            name9(Kit_Replacer<OldKit, NewKit, name9##_Tag>::func(&oldKit, &newKit)->name9), \
            name10(Kit_Replacer<OldKit, NewKit, name10##_Tag>::func(&oldKit, &newKit)->name10), \
            name11(Kit_Replacer<OldKit, NewKit, name11##_Tag>::func(&oldKit, &newKit)->name11), \
            name12(Kit_Replacer<OldKit, NewKit, name12##_Tag>::func(&oldKit, &newKit)->name12), \
            name13(Kit_Replacer<OldKit, NewKit, name13##_Tag>::func(&oldKit, &newKit)->name13) \
        { \
        } \
    }

#define KIT_CREATE14(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, Type12, name12, Type13, name13) \
    KIT__CREATE14(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, Type12, name12, Type13, name13)


//----------------------------------------------------------------

#define KIT__CREATE15(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, Type12, name12, Type13, name13, Type14, name14) \
    \
    struct Kit \
        : \
        Kit_FieldTag<struct name0##_Tag>, \
        Kit_FieldTag<struct name1##_Tag>, \
        Kit_FieldTag<struct name2##_Tag>, \
        Kit_FieldTag<struct name3##_Tag>, \
        Kit_FieldTag<struct name4##_Tag>, \
        Kit_FieldTag<struct name5##_Tag>, \
        Kit_FieldTag<struct name6##_Tag>, \
        Kit_FieldTag<struct name7##_Tag>, \
        Kit_FieldTag<struct name8##_Tag>, \
        Kit_FieldTag<struct name9##_Tag>, \
        Kit_FieldTag<struct name10##_Tag>, \
        Kit_FieldTag<struct name11##_Tag>, \
        Kit_FieldTag<struct name12##_Tag>, \
        Kit_FieldTag<struct name13##_Tag>, \
        Kit_FieldTag<struct name14##_Tag> \
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
        Type14 name14; \
        \
        sysinline Kit \
        ( \
            ParamType<Type0>::T name0, \
            ParamType<Type1>::T name1, \
            ParamType<Type2>::T name2, \
            ParamType<Type3>::T name3, \
            ParamType<Type4>::T name4, \
            ParamType<Type5>::T name5, \
            ParamType<Type6>::T name6, \
            ParamType<Type7>::T name7, \
            ParamType<Type8>::T name8, \
            ParamType<Type9>::T name9, \
            ParamType<Type10>::T name10, \
            ParamType<Type11>::T name11, \
            ParamType<Type12>::T name12, \
            ParamType<Type13>::T name13, \
            ParamType<Type14>::T name14 \
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
            name13(name13), \
            name14(name14) \
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
            name13(otherKit.name13), \
            name14(otherKit.name14) \
        { \
        } \
        \
        template <typename OldKit, typename NewKit> \
        sysinline Kit(const OldKit& oldKit, Kit_ReplaceConstructor, const NewKit& newKit) \
            : \
            name0(Kit_Replacer<OldKit, NewKit, name0##_Tag>::func(&oldKit, &newKit)->name0), \
            name1(Kit_Replacer<OldKit, NewKit, name1##_Tag>::func(&oldKit, &newKit)->name1), \
            name2(Kit_Replacer<OldKit, NewKit, name2##_Tag>::func(&oldKit, &newKit)->name2), \
            name3(Kit_Replacer<OldKit, NewKit, name3##_Tag>::func(&oldKit, &newKit)->name3), \
            name4(Kit_Replacer<OldKit, NewKit, name4##_Tag>::func(&oldKit, &newKit)->name4), \
            name5(Kit_Replacer<OldKit, NewKit, name5##_Tag>::func(&oldKit, &newKit)->name5), \
            name6(Kit_Replacer<OldKit, NewKit, name6##_Tag>::func(&oldKit, &newKit)->name6), \
            name7(Kit_Replacer<OldKit, NewKit, name7##_Tag>::func(&oldKit, &newKit)->name7), \
            name8(Kit_Replacer<OldKit, NewKit, name8##_Tag>::func(&oldKit, &newKit)->name8), \
            name9(Kit_Replacer<OldKit, NewKit, name9##_Tag>::func(&oldKit, &newKit)->name9), \
            name10(Kit_Replacer<OldKit, NewKit, name10##_Tag>::func(&oldKit, &newKit)->name10), \
            name11(Kit_Replacer<OldKit, NewKit, name11##_Tag>::func(&oldKit, &newKit)->name11), \
            name12(Kit_Replacer<OldKit, NewKit, name12##_Tag>::func(&oldKit, &newKit)->name12), \
            name13(Kit_Replacer<OldKit, NewKit, name13##_Tag>::func(&oldKit, &newKit)->name13), \
            name14(Kit_Replacer<OldKit, NewKit, name14##_Tag>::func(&oldKit, &newKit)->name14) \
        { \
        } \
    }

#define KIT_CREATE15(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, Type12, name12, Type13, name13, Type14, name14) \
    KIT__CREATE15(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, Type12, name12, Type13, name13, Type14, name14)


//----------------------------------------------------------------

#define KIT__CREATE16(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, Type12, name12, Type13, name13, Type14, name14, Type15, name15) \
    \
    struct Kit \
        : \
        Kit_FieldTag<struct name0##_Tag>, \
        Kit_FieldTag<struct name1##_Tag>, \
        Kit_FieldTag<struct name2##_Tag>, \
        Kit_FieldTag<struct name3##_Tag>, \
        Kit_FieldTag<struct name4##_Tag>, \
        Kit_FieldTag<struct name5##_Tag>, \
        Kit_FieldTag<struct name6##_Tag>, \
        Kit_FieldTag<struct name7##_Tag>, \
        Kit_FieldTag<struct name8##_Tag>, \
        Kit_FieldTag<struct name9##_Tag>, \
        Kit_FieldTag<struct name10##_Tag>, \
        Kit_FieldTag<struct name11##_Tag>, \
        Kit_FieldTag<struct name12##_Tag>, \
        Kit_FieldTag<struct name13##_Tag>, \
        Kit_FieldTag<struct name14##_Tag>, \
        Kit_FieldTag<struct name15##_Tag> \
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
        Type14 name14; \
        Type15 name15; \
        \
        sysinline Kit \
        ( \
            ParamType<Type0>::T name0, \
            ParamType<Type1>::T name1, \
            ParamType<Type2>::T name2, \
            ParamType<Type3>::T name3, \
            ParamType<Type4>::T name4, \
            ParamType<Type5>::T name5, \
            ParamType<Type6>::T name6, \
            ParamType<Type7>::T name7, \
            ParamType<Type8>::T name8, \
            ParamType<Type9>::T name9, \
            ParamType<Type10>::T name10, \
            ParamType<Type11>::T name11, \
            ParamType<Type12>::T name12, \
            ParamType<Type13>::T name13, \
            ParamType<Type14>::T name14, \
            ParamType<Type15>::T name15 \
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
            name13(name13), \
            name14(name14), \
            name15(name15) \
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
            name13(otherKit.name13), \
            name14(otherKit.name14), \
            name15(otherKit.name15) \
        { \
        } \
        \
        template <typename OldKit, typename NewKit> \
        sysinline Kit(const OldKit& oldKit, Kit_ReplaceConstructor, const NewKit& newKit) \
            : \
            name0(Kit_Replacer<OldKit, NewKit, name0##_Tag>::func(&oldKit, &newKit)->name0), \
            name1(Kit_Replacer<OldKit, NewKit, name1##_Tag>::func(&oldKit, &newKit)->name1), \
            name2(Kit_Replacer<OldKit, NewKit, name2##_Tag>::func(&oldKit, &newKit)->name2), \
            name3(Kit_Replacer<OldKit, NewKit, name3##_Tag>::func(&oldKit, &newKit)->name3), \
            name4(Kit_Replacer<OldKit, NewKit, name4##_Tag>::func(&oldKit, &newKit)->name4), \
            name5(Kit_Replacer<OldKit, NewKit, name5##_Tag>::func(&oldKit, &newKit)->name5), \
            name6(Kit_Replacer<OldKit, NewKit, name6##_Tag>::func(&oldKit, &newKit)->name6), \
            name7(Kit_Replacer<OldKit, NewKit, name7##_Tag>::func(&oldKit, &newKit)->name7), \
            name8(Kit_Replacer<OldKit, NewKit, name8##_Tag>::func(&oldKit, &newKit)->name8), \
            name9(Kit_Replacer<OldKit, NewKit, name9##_Tag>::func(&oldKit, &newKit)->name9), \
            name10(Kit_Replacer<OldKit, NewKit, name10##_Tag>::func(&oldKit, &newKit)->name10), \
            name11(Kit_Replacer<OldKit, NewKit, name11##_Tag>::func(&oldKit, &newKit)->name11), \
            name12(Kit_Replacer<OldKit, NewKit, name12##_Tag>::func(&oldKit, &newKit)->name12), \
            name13(Kit_Replacer<OldKit, NewKit, name13##_Tag>::func(&oldKit, &newKit)->name13), \
            name14(Kit_Replacer<OldKit, NewKit, name14##_Tag>::func(&oldKit, &newKit)->name14), \
            name15(Kit_Replacer<OldKit, NewKit, name15##_Tag>::func(&oldKit, &newKit)->name15) \
        { \
        } \
    }

#define KIT_CREATE16(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, Type12, name12, Type13, name13, Type14, name14, Type15, name15) \
    KIT__CREATE16(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, Type12, name12, Type13, name13, Type14, name14, Type15, name15)


