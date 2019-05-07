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
        sysinline Kit \
        ( \
            typenameWord() ParamType<Type0>::T name0, \
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
    }
    
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
        sysinline Kit \
        ( \
            typenameWord() ParamType<Type0>::T name0, \
            typenameWord() ParamType<Type1>::T name1 \
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
    }
    
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
        sysinline Kit \
        ( \
            typenameWord() ParamType<Type0>::T name0, \
            typenameWord() ParamType<Type1>::T name1, \
            typenameWord() ParamType<Type2>::T name2 \
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
    }
    
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
        sysinline Kit \
        ( \
            typenameWord() ParamType<Type0>::T name0, \
            typenameWord() ParamType<Type1>::T name1, \
            typenameWord() ParamType<Type2>::T name2, \
            typenameWord() ParamType<Type3>::T name3 \
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
    }
    
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
        sysinline Kit \
        ( \
            typenameWord() ParamType<Type0>::T name0, \
            typenameWord() ParamType<Type1>::T name1, \
            typenameWord() ParamType<Type2>::T name2, \
            typenameWord() ParamType<Type3>::T name3, \
            typenameWord() ParamType<Type4>::T name4 \
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
    }
    
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
        sysinline Kit \
        ( \
            typenameWord() ParamType<Type0>::T name0, \
            typenameWord() ParamType<Type1>::T name1, \
            typenameWord() ParamType<Type2>::T name2, \
            typenameWord() ParamType<Type3>::T name3, \
            typenameWord() ParamType<Type4>::T name4, \
            typenameWord() ParamType<Type5>::T name5 \
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
    }
    
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
        sysinline Kit \
        ( \
            typenameWord() ParamType<Type0>::T name0, \
            typenameWord() ParamType<Type1>::T name1, \
            typenameWord() ParamType<Type2>::T name2, \
            typenameWord() ParamType<Type3>::T name3, \
            typenameWord() ParamType<Type4>::T name4, \
            typenameWord() ParamType<Type5>::T name5, \
            typenameWord() ParamType<Type6>::T name6 \
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
    }
    
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
        sysinline Kit \
        ( \
            typenameWord() ParamType<Type0>::T name0, \
            typenameWord() ParamType<Type1>::T name1, \
            typenameWord() ParamType<Type2>::T name2, \
            typenameWord() ParamType<Type3>::T name3, \
            typenameWord() ParamType<Type4>::T name4, \
            typenameWord() ParamType<Type5>::T name5, \
            typenameWord() ParamType<Type6>::T name6, \
            typenameWord() ParamType<Type7>::T name7 \
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
    }
    
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
        sysinline Kit \
        ( \
            typenameWord() ParamType<Type0>::T name0, \
            typenameWord() ParamType<Type1>::T name1, \
            typenameWord() ParamType<Type2>::T name2, \
            typenameWord() ParamType<Type3>::T name3, \
            typenameWord() ParamType<Type4>::T name4, \
            typenameWord() ParamType<Type5>::T name5, \
            typenameWord() ParamType<Type6>::T name6, \
            typenameWord() ParamType<Type7>::T name7, \
            typenameWord() ParamType<Type8>::T name8 \
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
    }
    
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
        sysinline Kit \
        ( \
            typenameWord() ParamType<Type0>::T name0, \
            typenameWord() ParamType<Type1>::T name1, \
            typenameWord() ParamType<Type2>::T name2, \
            typenameWord() ParamType<Type3>::T name3, \
            typenameWord() ParamType<Type4>::T name4, \
            typenameWord() ParamType<Type5>::T name5, \
            typenameWord() ParamType<Type6>::T name6, \
            typenameWord() ParamType<Type7>::T name7, \
            typenameWord() ParamType<Type8>::T name8, \
            typenameWord() ParamType<Type9>::T name9 \
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
    }
    
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
        sysinline Kit \
        ( \
            typenameWord() ParamType<Type0>::T name0, \
            typenameWord() ParamType<Type1>::T name1, \
            typenameWord() ParamType<Type2>::T name2, \
            typenameWord() ParamType<Type3>::T name3, \
            typenameWord() ParamType<Type4>::T name4, \
            typenameWord() ParamType<Type5>::T name5, \
            typenameWord() ParamType<Type6>::T name6, \
            typenameWord() ParamType<Type7>::T name7, \
            typenameWord() ParamType<Type8>::T name8, \
            typenameWord() ParamType<Type9>::T name9, \
            typenameWord() ParamType<Type10>::T name10 \
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
    }
    
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
        sysinline Kit \
        ( \
            typenameWord() ParamType<Type0>::T name0, \
            typenameWord() ParamType<Type1>::T name1, \
            typenameWord() ParamType<Type2>::T name2, \
            typenameWord() ParamType<Type3>::T name3, \
            typenameWord() ParamType<Type4>::T name4, \
            typenameWord() ParamType<Type5>::T name5, \
            typenameWord() ParamType<Type6>::T name6, \
            typenameWord() ParamType<Type7>::T name7, \
            typenameWord() ParamType<Type8>::T name8, \
            typenameWord() ParamType<Type9>::T name9, \
            typenameWord() ParamType<Type10>::T name10, \
            typenameWord() ParamType<Type11>::T name11 \
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
    }
    
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
        sysinline Kit \
        ( \
            typenameWord() ParamType<Type0>::T name0, \
            typenameWord() ParamType<Type1>::T name1, \
            typenameWord() ParamType<Type2>::T name2, \
            typenameWord() ParamType<Type3>::T name3, \
            typenameWord() ParamType<Type4>::T name4, \
            typenameWord() ParamType<Type5>::T name5, \
            typenameWord() ParamType<Type6>::T name6, \
            typenameWord() ParamType<Type7>::T name7, \
            typenameWord() ParamType<Type8>::T name8, \
            typenameWord() ParamType<Type9>::T name9, \
            typenameWord() ParamType<Type10>::T name10, \
            typenameWord() ParamType<Type11>::T name11, \
            typenameWord() ParamType<Type12>::T name12 \
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
    }
    
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
        sysinline Kit \
        ( \
            typenameWord() ParamType<Type0>::T name0, \
            typenameWord() ParamType<Type1>::T name1, \
            typenameWord() ParamType<Type2>::T name2, \
            typenameWord() ParamType<Type3>::T name3, \
            typenameWord() ParamType<Type4>::T name4, \
            typenameWord() ParamType<Type5>::T name5, \
            typenameWord() ParamType<Type6>::T name6, \
            typenameWord() ParamType<Type7>::T name7, \
            typenameWord() ParamType<Type8>::T name8, \
            typenameWord() ParamType<Type9>::T name9, \
            typenameWord() ParamType<Type10>::T name10, \
            typenameWord() ParamType<Type11>::T name11, \
            typenameWord() ParamType<Type12>::T name12, \
            typenameWord() ParamType<Type13>::T name13 \
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
    }
    
#define KIT_CREATE14(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, Type12, name12, Type13, name13) \
    KIT__CREATE14(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, Type12, name12, Type13, name13, KIT__TYPENAME_NO)

#define KIT_CREATE14_(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, Type12, name12, Type13, name13) \
    KIT__CREATE14(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, Type12, name12, Type13, name13, KIT__TYPENAME_YES)

//----------------------------------------------------------------

#define KIT__CREATE15(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, Type12, name12, Type13, name13, Type14, name14, typenameWord) \
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
        KitFieldTag<struct name13##_Tag>, \
        KitFieldTag<struct name14##_Tag> \
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
            typenameWord() ParamType<Type0>::T name0, \
            typenameWord() ParamType<Type1>::T name1, \
            typenameWord() ParamType<Type2>::T name2, \
            typenameWord() ParamType<Type3>::T name3, \
            typenameWord() ParamType<Type4>::T name4, \
            typenameWord() ParamType<Type5>::T name5, \
            typenameWord() ParamType<Type6>::T name6, \
            typenameWord() ParamType<Type7>::T name7, \
            typenameWord() ParamType<Type8>::T name8, \
            typenameWord() ParamType<Type9>::T name9, \
            typenameWord() ParamType<Type10>::T name10, \
            typenameWord() ParamType<Type11>::T name11, \
            typenameWord() ParamType<Type12>::T name12, \
            typenameWord() ParamType<Type13>::T name13, \
            typenameWord() ParamType<Type14>::T name14 \
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
            name13(KitReplacer<OldKit, NewKit, name13##_Tag>::func(&oldKit, &newKit)->name13), \
            name14(KitReplacer<OldKit, NewKit, name14##_Tag>::func(&oldKit, &newKit)->name14) \
        { \
        } \
    }
    
#define KIT_CREATE15(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, Type12, name12, Type13, name13, Type14, name14) \
    KIT__CREATE15(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, Type12, name12, Type13, name13, Type14, name14, KIT__TYPENAME_NO)

#define KIT_CREATE15_(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, Type12, name12, Type13, name13, Type14, name14) \
    KIT__CREATE15(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, Type12, name12, Type13, name13, Type14, name14, KIT__TYPENAME_YES)

//----------------------------------------------------------------

#define KIT__CREATE16(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, Type12, name12, Type13, name13, Type14, name14, Type15, name15, typenameWord) \
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
        KitFieldTag<struct name13##_Tag>, \
        KitFieldTag<struct name14##_Tag>, \
        KitFieldTag<struct name15##_Tag> \
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
            typenameWord() ParamType<Type0>::T name0, \
            typenameWord() ParamType<Type1>::T name1, \
            typenameWord() ParamType<Type2>::T name2, \
            typenameWord() ParamType<Type3>::T name3, \
            typenameWord() ParamType<Type4>::T name4, \
            typenameWord() ParamType<Type5>::T name5, \
            typenameWord() ParamType<Type6>::T name6, \
            typenameWord() ParamType<Type7>::T name7, \
            typenameWord() ParamType<Type8>::T name8, \
            typenameWord() ParamType<Type9>::T name9, \
            typenameWord() ParamType<Type10>::T name10, \
            typenameWord() ParamType<Type11>::T name11, \
            typenameWord() ParamType<Type12>::T name12, \
            typenameWord() ParamType<Type13>::T name13, \
            typenameWord() ParamType<Type14>::T name14, \
            typenameWord() ParamType<Type15>::T name15 \
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
            name13(KitReplacer<OldKit, NewKit, name13##_Tag>::func(&oldKit, &newKit)->name13), \
            name14(KitReplacer<OldKit, NewKit, name14##_Tag>::func(&oldKit, &newKit)->name14), \
            name15(KitReplacer<OldKit, NewKit, name15##_Tag>::func(&oldKit, &newKit)->name15) \
        { \
        } \
    }
    
#define KIT_CREATE16(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, Type12, name12, Type13, name13, Type14, name14, Type15, name15) \
    KIT__CREATE16(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, Type12, name12, Type13, name13, Type14, name14, Type15, name15, KIT__TYPENAME_NO)

#define KIT_CREATE16_(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, Type12, name12, Type13, name13, Type14, name14, Type15, name15) \
    KIT__CREATE16(Kit, Type0, name0, Type1, name1, Type2, name2, Type3, name3, Type4, name4, Type5, name5, Type6, name6, Type7, name7, Type8, name8, Type9, name9, Type10, name10, Type11, name11, Type12, name12, Type13, name13, Type14, name14, Type15, name15, KIT__TYPENAME_YES)

