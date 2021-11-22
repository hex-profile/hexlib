#pragma once

#include "compileTools/compileTools.h"
#include "numbers/interface/exchangeInterface.h"

//================================================================
//
// Common Number Interface:
//
// Defines common template prototypes of number operations.
// Specific number implementations define these functions.
//
//================================================================

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Type traits
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// VECTOR_BASE
//
// Returns the type of element for vector types.
//
// (Also removes const/ref/etc)
//
//================================================================

template <typename Vector>
struct VectorBaseImpl
{
    using T = Vector;
};

//----------------------------------------------------------------

#define VECTOR_BASE(Vector) \
    typename VectorBaseImpl<TYPE_CLEANSE(Vector)>::T

#define VECTOR_BASE_(Vector) \
    VectorBaseImpl<TYPE_CLEANSE_(Vector)>::T

//================================================================
//
// VECTOR_REBASE
//
// Returns vector type having the same structure as specified template,
// but with different base type.
//
// (Also removes const/ref/etc)
//
//================================================================

template <typename VectorTemplate, typename NewBase>
struct VectorRebaseImpl
{
    using T = NewBase;
};

//----------------------------------------------------------------

#define VECTOR_REBASE(VectorTemplate, NewBase) \
    typename VectorRebaseImpl<TYPE_CLEANSE(VectorTemplate), TYPE_CLEANSE(NewBase)>::T

#define VECTOR_REBASE_(VectorTemplate, NewBase) \
    VectorRebaseImpl<TYPE_CLEANSE_(VectorTemplate), TYPE_CLEANSE_(NewBase)>::T

//================================================================
//
// VECTOR_BASE_REBASE_VECTOR_IMPL
//
//================================================================

#define VECTOR_BASE_REBASE_VECTOR_IMPL(Vector) \
    \
    template <typename Type> \
    struct VectorBaseImpl<Vector<Type>> \
    { \
        using T = Type; \
    }; \
    \
    template <typename OldBase, typename NewBase> \
    struct VectorRebaseImpl<Vector<OldBase>, NewBase> \
    { \
        using T = Vector<NewBase>; \
    };

//================================================================
//
// vectorExtend
//
// Converts the scalar value to a vector.
//
//================================================================

template <typename Vector>
struct VectorExtendImpl
{
    static sysinline Vector func(const Vector& value)
        {return value;} // Scalar implementation
};

//----------------------------------------------------------------

template <typename Vector>
sysinline TYPE_CLEANSE(Vector) vectorExtend(const VECTOR_BASE(Vector)& value)
    {return VectorExtendImpl<TYPE_CLEANSE(Vector)>::func(value);}

//================================================================
//
// TYPE_IS_SIGNED
//
// Determines if the numeric type can have negative value.
//
//================================================================

template <typename Type>
struct TypeIsSignedImpl;

//----------------------------------------------------------------

#define TYPE_IS_SIGNED_IMPL(Type, isSigned) \
    template <> struct TypeIsSignedImpl<Type> {static constexpr bool val = (isSigned);};

//----------------------------------------------------------------

#define TYPE_IS_SIGNED(Type) \
    TypeIsSignedImpl<VECTOR_BASE(TYPE_CLEANSE(Type))>::val

#define TYPE_IS_SIGNED_(Type) \
    TypeIsSignedImpl<VECTOR_BASE_(TYPE_CLEANSE_(Type))>::val

//================================================================
//
// TYPE_MAKE_SIGNED
// TYPE_MAKE_UNSIGNED
//
// (Also removes const, etc).
//
//================================================================

template <typename Type>
struct TypeMakeSignedImpl;

template <typename Type>
struct TypeMakeUnsignedImpl;

//----------------------------------------------------------------

#define TYPE_MAKE_SIGNED_IMPL(Src, Dst) \
    template <> struct TypeMakeSignedImpl<Src> {using T = Dst;};

#define TYPE_MAKE_UNSIGNED_IMPL(Src, Dst) \
    template <> struct TypeMakeUnsignedImpl<Src> {using T = Dst;};

//----------------------------------------------------------------

#define TYPE_MAKE_SIGNED(Type) \
    VECTOR_REBASE(Type, typename TypeMakeSignedImpl<VECTOR_BASE(Type)>::T)

#define TYPE_MAKE_SIGNED_(Type) \
    VECTOR_REBASE_(Type, TypeMakeSignedImpl<VECTOR_BASE_(Type)>::T)

////

#define TYPE_MAKE_UNSIGNED(Type) \
    VECTOR_REBASE(Type, typename TypeMakeUnsignedImpl<VECTOR_BASE(Type)>::T)

#define TYPE_MAKE_UNSIGNED_(Type) \
    VECTOR_REBASE_(Type, TypeMakeUnsignedImpl<VECTOR_BASE_(Type)>::T)

//================================================================
//
// TYPE_IS_CONTROLLED
//
// Determines wheter the numeric type has built-in error state, like IEEE float.
//
// TYPE_MAKE_CONTROLLED
// TYPE_MAKE_UNCONTROLLED
//
// (Also removes const, etc).
//
//================================================================

template <typename Type>
struct TypeIsControlledImpl;

template <typename Type>
struct TypeMakeControlledImpl;

template <typename Type>
struct TypeMakeUncontrolledImpl;

//----------------------------------------------------------------

#define TYPE_IS_CONTROLLED(Type) \
    TypeIsControlledImpl<VECTOR_BASE(Type)>::val

#define TYPE_IS_CONTROLLED_(Type) \
    TypeIsControlledImpl<VECTOR_BASE_(Type)>::val

//----------------------------------------------------------------

#define TYPE_MAKE_CONTROLLED(Type) \
    VECTOR_REBASE(Type, typename TypeMakeControlledImpl<VECTOR_BASE(Type)>::T)

#define TYPE_MAKE_CONTROLLED_(Type) \
    VECTOR_REBASE_(Type, TypeMakeControlledImpl<VECTOR_BASE_(Type)>::T)

//----------------------------------------------------------------

#define TYPE_MAKE_UNCONTROLLED(Type) \
    VECTOR_REBASE(Type, typename TypeMakeUncontrolledImpl<VECTOR_BASE(Type)>::T)

#define TYPE_MAKE_UNCONTROLLED_(Type) \
    VECTOR_REBASE_(Type, TypeMakeUncontrolledImpl<VECTOR_BASE_(Type)>::T)

//----------------------------------------------------------------

#define TYPE_IS_CONTROLLED_IMPL(Type, isControlled) \
    template <> struct TypeIsControlledImpl<Type> {static constexpr bool val = (isControlled);};

#define TYPE_MAKE_CONTROLLED_IMPL(Src, Dst) \
    template <> struct TypeMakeControlledImpl<Src> {using T = Dst;};

#define TYPE_MAKE_UNCONTROLLED_IMPL(Src, Dst) \
    template <> struct TypeMakeUncontrolledImpl<Src> {using T = Dst;};

//----------------------------------------------------------------

#define TYPE_CONTROL_VECTOR_IMPL(Vector) \
    \
    template <typename Type> \
    struct TypeIsControlledImpl<Vector<Type>> \
    { \
        static constexpr bool val = TYPE_IS_CONTROLLED(Type); \
    }; \
    \
    template <typename Type> \
    struct TypeMakeControlledImpl<Vector<Type>> \
    { \
        using T = Vector<TYPE_MAKE_CONTROLLED(Type)>; \
    }; \
    \
    template <typename Type> \
    struct TypeMakeUncontrolledImpl<Vector<Type>> \
    { \
        using T = Vector<TYPE_MAKE_UNCONTROLLED(Type)>; \
    };

//================================================================
//
// The minimal and maximal value of a type.
// There are two versions: runtime and static.
//
// Macros to use in application code:
//
// typeMin (Runtime)
// typeMax (Runtime)
// TYPE_MIN (Static)
// TYPE_MAX (Static)
//
// Implementation helper macros:
//
// TYPE_MIN_MAX_IMPL_RUNTIME
// TYPE_MIN_MAX_IMPL_STATIC
// TYPE_MIN_MAX_IMPL_BOTH
//
//================================================================

template <typename Type>
struct TypeMinMaxImpl;

template <typename Type>
struct TypeMinMaxStaticImpl;

//----------------------------------------------------------------

#define TYPE_MIN(Type) \
    TypeMinMaxStaticImpl<Type>::minVal

#define TYPE_MAX(Type) \
    TypeMinMaxStaticImpl<Type>::maxVal

//----------------------------------------------------------------

template <typename Type>
sysinline Type typeMin()
    {return vectorExtend<Type>(TypeMinMaxImpl<VECTOR_BASE(Type)>::minVal());}

template <typename Type>
sysinline Type typeMax()
    {return vectorExtend<Type>(TypeMinMaxImpl<VECTOR_BASE(Type)>::maxVal());}

//----------------------------------------------------------------

#define TYPE_MIN_MAX_IMPL_RUNTIME(Type, minValue, maxValue) \
    \
    template <> \
    struct TypeMinMaxImpl<Type> \
    { \
        static sysinline Type minVal() {return (minValue);} \
        static sysinline Type maxVal() {return (maxValue);} \
    };

//----------------------------------------------------------------

#define TYPE_MIN_MAX_IMPL_STATIC(Type, minValue, maxValue) \
    \
    template <> \
    struct TypeMinMaxStaticImpl<Type> \
    { \
        static constexpr Type minVal = (minValue); \
        static constexpr Type maxVal = (maxValue); \
    };

//----------------------------------------------------------------

#define TYPE_MIN_MAX_IMPL_BOTH(Type, minValue, maxValue) \
    \
    TYPE_MIN_MAX_IMPL_RUNTIME(Type, minValue, maxValue) \
    TYPE_MIN_MAX_IMPL_STATIC(Type, minValue, maxValue)

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Controlled type functions
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// nanOf
//
// Returns NAN of specified type.
//
//================================================================

template <typename Type>
struct NanOfImpl;

//----------------------------------------------------------------

template <typename Type>
sysinline auto nanOf()
{
    return vectorExtend<Type>(NanOfImpl<VECTOR_BASE(Type)>::func());
}

//================================================================
//
// def
//
// Checks whether the value is defined.
//
//================================================================

template <typename Type>
struct DefImpl;

//----------------------------------------------------------------

template <typename Type>
sysinline auto def(const Type& value)
{
    return DefImpl<Type>::func(value);
}

//================================================================
//
// def for multiple arguments
//
//================================================================

template <typename T0, typename... Types>
sysinline auto def(const T0& v0, const Types&... values)
{
    auto result = def(v0);
    char tmp[] = {(result = result && def(values), 'x')...};
    return result;
}

//================================================================
//
// Arithmetic operations
//
//================================================================

enum UnaryOperation {OpNeg, OpPos, OpBitInv, OpAbs};
enum BinaryOperation {OpAdd, OpSub, OpMul, OpDiv, OpRem, OpShl, OpShr, OpBitAnd, OpBitOr, OpBitXor, OpMin, OpMax};

//================================================================
//
// safeAdd, safeMul, etc.
//
// Semi-controlled validation functions,
// can be implemented by controlled types for their base types.
//
//================================================================

template <typename Type, UnaryOperation operation>
struct SafeUnaryImpl
{
    struct Code;
};

//----------------------------------------------------------------

template <typename Type, BinaryOperation operation>
struct SafeBinaryImpl
{
    struct Code;
};

//----------------------------------------------------------------

#define TMP_MACRO_CAN1(funcName, OpName) \
    template <typename Type> \
    sysinline bool funcName(const Type& value, Type& result) \
        {return SafeUnaryImpl<Type, OpName>::Code::func(value, result);}

TMP_MACRO_CAN1(safeNeg, OpNeg)
TMP_MACRO_CAN1(safeAbs, OpAbs)

#undef TMP_MACRO_CAN1

//----------------------------------------------------------------

#define TMP_MACRO_CAN2(funcName, OpName) \
    template <typename Type> \
    sysinline bool funcName(const Type& A, const Type& B, Type& result) \
        {return SafeBinaryImpl<Type, OpName>::Code::func(A, B, result);}

TMP_MACRO_CAN2(safeAdd, OpAdd)
TMP_MACRO_CAN2(safeSub, OpSub)
TMP_MACRO_CAN2(safeMul, OpMul)
TMP_MACRO_CAN2(safeDiv, OpDiv)
TMP_MACRO_CAN2(safeRem, OpRem)
TMP_MACRO_CAN2(safeShl, OpShl)
TMP_MACRO_CAN2(safeShr, OpShr)

#undef TMP_MACRO_CAN2

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Conversions: Implementors' interface layer.
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// Rounding mode
//
//================================================================

enum Rounding {RoundUp, RoundDown, RoundNearest, RoundExact, Round_COUNT};

//================================================================
//
// CONVERT_FAMILY
//
// Returns the family of Type, for group implementation of conversion routines.
// (The family is just a some type to disambiguate)
//
//================================================================

template <typename Type>
struct ConvertFamilyImpl
{
    using T = Type; // By default, the type itself is used as a family.
};

//----------------------------------------------------------------

#define CONVERT_FAMILY_IMPL(Type, Family) \
    template <> struct ConvertFamilyImpl<Type> {using T = Family;};

//----------------------------------------------------------------

#define CONVERT_FAMILY_VECTOR_IMPL(Vector, Family) \
    \
    struct Family; \
    \
    template <typename Type> \
    struct ConvertFamilyImpl<Vector<Type>> \
    { \
        using T = Family; \
    };

//----------------------------------------------------------------

#define CONVERT_FAMILY(Type) \
    typename ConvertFamilyImpl<TYPE_CLEANSE(Type)>::T

#define CONVERT_FAMILY_(Type) \
    ConvertFamilyImpl<TYPE_CLEANSE_(Type)>::T

//================================================================
//
// ConvertHint
//
//================================================================

enum ConvertHint
{
    // Standard mode: Any number range.
    ConvertNormal,

    // Optimized mode: Guaranteed to work correctly only for numbers >= 0.
    ConvertNonneg
};

//================================================================
//
// ConvertImpl
//
// "ConvertImpl" is a "guided conversion": the user specifies "guide" type;
// the function returns a value converted to the required type
// (for vector types, guide type can be not equal to destination type).
//
// The conversion can be checked or unchecked (for arithmetic overflows and undefined inputs).
//
// If the conversion is checked (specified by parameter),
// the success flag is stored in the destination type, 
// which should have built-in error state (like IEEE float).
//
// If the conversion is unchecked, no arithmetic check is performed,
// and the destination type can be either controlled or uncontrolled type
// (having built-in error state or not).
//
//================================================================

enum ConvertCheck {ConvertChecked, ConvertUnchecked};

//----------------------------------------------------------------

template <typename SrcFamily, typename DstFamily, ConvertCheck check, Rounding rounding, ConvertHint hint>
struct ConvertImpl
{
    template <typename Src, typename Dst>
    struct Convert;
};

//----------------------------------------------------------------

template <typename Src, typename Dst, ConvertCheck check, Rounding rounding, ConvertHint hint>
struct ConvertImplCall
{
    using SrcFamily = CONVERT_FAMILY(Src);
    using DstFamily = CONVERT_FAMILY(Dst);

    using Code = typename ConvertImpl<SrcFamily, DstFamily, check, rounding, hint>::template Convert<Src, Dst>;
};

//================================================================
//
// ConvertImplFlag
//
// The flag-based conversion:
// bool convert(const Src& src, Dst& dst).
//
// The result value is stored to output parameter;
// the success flag is returned as a function result.
//
// The conversion is always checked (for arithmetic overflows and undefined inputs),
// including case when both source and destination type are uncontrolled (not having built-in error state).
//
//================================================================

template <typename SrcFamily, typename DstFamily, Rounding rounding, ConvertHint hint>
struct ConvertImplFlag
{
    template <typename Src, typename Dst>
    struct Convert;
};

//----------------------------------------------------------------

template <typename Src, typename Dst, Rounding rounding, ConvertHint hint>
struct ConvertImplFlagCall
{
    using SrcFamily = CONVERT_FAMILY(Src);
    using DstFamily = CONVERT_FAMILY(Dst);

    using Code = typename ConvertImplFlag<SrcFamily, DstFamily, rounding, hint>::template Convert<Src, Dst>;
};

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Conversions: Scalar layer.
//
// Implements automatic conversions of unrelated controlled types
// via their base types.
//
// Gates: ConvertScalar, ConvertScalarFlag.
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// ConvertScalar
//
//================================================================

template <typename Src, typename Dst, ConvertCheck check, Rounding rounding, ConvertHint hint>
struct ConvertScalar
{
    using SrcBase = TYPE_MAKE_UNCONTROLLED(Src);
    using DstBase = TYPE_MAKE_UNCONTROLLED(Dst);

    static constexpr bool bothAreSelfBased = TYPE_EQUAL(Src, SrcBase) && TYPE_EQUAL(Dst, DstBase);

    ////

    using SimplePath = ConvertImplCall<Src, Dst, check, rounding, hint>;

    COMPILE_ASSERT(bothAreSelfBased); // Currently no transitive conversions are used.
    using Code = typename SimplePath::Code;
};

//================================================================
//
// ConvertScalarFlag
//
//================================================================

template <typename Src, typename Dst, Rounding rounding, ConvertHint hint>
struct ConvertScalarFlag
{
    using SrcBase = TYPE_MAKE_UNCONTROLLED(Src);
    using DstBase = TYPE_MAKE_UNCONTROLLED(Dst);

    static constexpr bool bothAreSelfBased = TYPE_EQUAL(Src, SrcBase) && TYPE_EQUAL(Dst, DstBase);

    ////

    using SimplePath = ConvertImplFlagCall<Src, Dst, rounding, hint>;

    COMPILE_ASSERT(bothAreSelfBased); // Currently no transitive conversions are used.
    using Code = typename SimplePath::Code;
};

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Conversions: Vector layer.
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// ConvertResult
//
// The type of conversion result for
// Src value being converted GUIDED by Dst type.
//
// The result can be not equal to Dst for vector types.
//
//================================================================

template <typename Src, typename Dst>
struct ConvertResult
{
    using SrcBase = VECTOR_BASE(Src);
    using DstBase = VECTOR_BASE(Dst);

    static constexpr bool srcIsVector = !TYPE_EQUAL(TYPE_CLEANSE(Src), SrcBase);
    static constexpr bool dstIsVector = !TYPE_EQUAL(TYPE_CLEANSE(Dst), DstBase);

    // Rebase if Src is vector and Dst is scalar
    using T = TYPE_SELECT(srcIsVector && !dstIsVector, VECTOR_REBASE(Src, DstBase), TYPE_CLEANSE(Dst));
};

//================================================================
//
// ConvertVector
//
//================================================================

template <typename Src, typename Dst, ConvertCheck check, Rounding rounding, ConvertHint hint>
struct ConvertVector
{

    using SrcBase = VECTOR_BASE(Src);
    using DstBase = VECTOR_BASE(Dst);

    static constexpr bool srcIsVector = !TYPE_EQUAL(TYPE_CLEANSE(Src), SrcBase);
    static constexpr bool dstIsVector = !TYPE_EQUAL(TYPE_CLEANSE(Dst), DstBase);

    using ConvertScalarScalar = typename ConvertScalar<SrcBase, DstBase, check, rounding, hint>::Code;

    // Should be implemented as ConvertImpl<Vec, Vec>
    using ConvertVectorVector = typename ConvertImplCall<Src, Dst, check, rounding, hint>::Code;

    // Guided by scalar, should be implemented as ConvertImpl<Vec, Vec>
    using ConvertVectorScalar = typename ConvertImplCall<Src, VECTOR_REBASE(Src, DstBase), check, rounding, hint>::Code;

    struct ConvertScalarVector
    {
        static sysinline Dst func(const Src& srcScalar)
            {return vectorExtend<Dst>(ConvertScalarScalar::func(srcScalar));}
    };

    using Code =
        TYPE_SELECT(!srcIsVector && !dstIsVector, ConvertScalarScalar,
        TYPE_SELECT(!srcIsVector && dstIsVector, ConvertScalarVector,
        TYPE_SELECT(srcIsVector && dstIsVector, ConvertVectorVector,
        ConvertVectorScalar)));

};

//================================================================
//
// ConvertVectorFlag
//
//================================================================

template <typename Src, typename Dst, Rounding rounding, ConvertHint hint>
struct ConvertVectorFlag
{

    using SrcBase = VECTOR_BASE(Src);
    using DstBase = VECTOR_BASE(Dst);

    static constexpr bool srcIsVector = !TYPE_EQUAL(TYPE_CLEANSE(Src), SrcBase);
    static constexpr bool dstIsVector = !TYPE_EQUAL(TYPE_CLEANSE(Dst), DstBase);

    ////

    using ConvertScalarScalar = typename ConvertScalarFlag<SrcBase, DstBase, rounding, hint>::Code;

    // Should be implemented as ConvertImplFlag
    using ConvertVectorVector = typename ConvertImplFlagCall<Src, Dst, rounding, hint>::Code; 

    struct ConvertVectorScalarProhibited;

    ////

    struct ConvertScalarVector
    {
        static sysinline VECTOR_REBASE(Dst, bool) func(const Src& srcScalar, Dst& dst)
        {
            DstBase dstScalar = ConvertScalar<int, DstBase, ConvertUnchecked, RoundNearest, ConvertNonneg>::Code::func(0);
            bool ok = ConvertScalarScalar::func(srcScalar, dstScalar);
            dst = vectorExtend<Dst>(dstScalar);
            return vectorExtend<VECTOR_REBASE(Dst, bool)>(ok);
        }
    };

    ////

    using Code =
        TYPE_SELECT(!srcIsVector && !dstIsVector, ConvertScalarScalar,
        TYPE_SELECT(!srcIsVector && dstIsVector, ConvertScalarVector,
        TYPE_SELECT(srcIsVector && dstIsVector, ConvertVectorVector,
        ConvertVectorScalarProhibited)));

};

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Conversions: User functions layer.
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// convertDown / convertUp / convertNearest / convertExact
// convertUpNonneg / convertDownNonneg / convertNearestNonneg
//
// The widely-used interface conversion functions.
//
// For the result-based error checking the best available way is used:
// If the destination type has built-in error state, the conversion is checked, otherwise not.
//
// The "Nonneg" versions are optimized functions, guaranteed to work correctly only for numbers >= 0:
//
//================================================================

#define TMP_CONVERT_FUNC(baseFunc, rounding, hint) \
    \
    template <typename DstGuide, typename Src> \
    sysinline typename ConvertResult<Src, DstGuide>::T baseFunc(const Src& src) \
    { \
        const ConvertCheck check = TYPE_IS_CONTROLLED(DstGuide) ? ConvertChecked : ConvertUnchecked; \
        using Impl = typename ConvertVector<Src, DstGuide, check, rounding, hint>::Code; \
        return Impl::func(src); \
    } \
    \
    template <typename Src, typename Dst> \
    sysinline VECTOR_REBASE(Dst, bool) baseFunc(const Src& src, Dst& dst) \
    { \
        using Impl = typename ConvertVectorFlag<Src, Dst, rounding, hint>::Code; \
        return Impl::func(src, dst); \
    }

TMP_CONVERT_FUNC(convertUp, RoundUp, ConvertNormal)
TMP_CONVERT_FUNC(convertDown, RoundDown, ConvertNormal)
TMP_CONVERT_FUNC(convertNearest, RoundNearest, ConvertNormal)
TMP_CONVERT_FUNC(convertExact, RoundExact, ConvertNormal)

TMP_CONVERT_FUNC(convertUpNonneg, RoundUp, ConvertNonneg)
TMP_CONVERT_FUNC(convertDownNonneg, RoundDown, ConvertNonneg)
TMP_CONVERT_FUNC(convertNearestNonneg, RoundNearest, ConvertNonneg)
TMP_CONVERT_FUNC(convertExactNonneg, RoundExact, ConvertNonneg)

#undef TMP_CONVERT_FUNC

//----------------------------------------------------------------

#define TMP_CONVERT_UNCHECKED(baseFunc, rounding, hint) \
    \
    template <typename DstGuide, typename Src> \
    sysinline typename ConvertResult<Src, DstGuide>::T baseFunc(const Src& src) \
    { \
        using Impl = typename ConvertVector<Src, DstGuide, ConvertUnchecked, rounding, hint>::Code; \
        return Impl::func(src); \
    } \

TMP_CONVERT_UNCHECKED(convertUpUnchecked, RoundUp, ConvertNormal)
TMP_CONVERT_UNCHECKED(convertDownUnchecked, RoundDown, ConvertNormal)
TMP_CONVERT_UNCHECKED(convertNearestUnchecked, RoundNearest, ConvertNormal)
TMP_CONVERT_UNCHECKED(convertExactUnchecked, RoundExact, ConvertNormal)

#undef TMP_CONVERT_UNCHECKED

//================================================================
//
// convertFlex
//
//================================================================

template <typename DstGuide, ConvertCheck check, Rounding rounding, ConvertHint hint, typename Src>
sysinline DstGuide convertFlex(const Src& src)
{
    using Impl = typename ConvertVector<Src, DstGuide, check, rounding, hint>::Code;
    return Impl::func(src);
}

//----------------------------------------------------------------

template <Rounding rounding, ConvertHint hint, typename Src, typename Dst>
sysinline bool convertFlex(const Src& src, Dst& dst)
{
    using Impl = typename ConvertVectorFlag<Src, Dst, rounding, hint>::Code;
    return Impl::func(src, dst);
}

//================================================================
//
// zeroOf
//
// Generates zero of a specified type.
//
//================================================================

template <typename Dst>
sysinline Dst zeroOf()
{
    using Impl = typename ConvertVector<int, Dst, ConvertUnchecked, RoundNearest, ConvertNonneg>::Code;
    return Impl::func(0);
}

//================================================================
//
// val
//
// Conversion shortcut: Converts a controlled type value to uncontrolled base type,
// without checking.
//
//================================================================

template <typename Type>
sysinline TYPE_MAKE_UNCONTROLLED(Type) val(const Type& value)
{
    using Impl = typename ConvertVector<Type, TYPE_MAKE_UNCONTROLLED(Type), ConvertUnchecked, RoundExact, ConvertNormal>::Code;
    return Impl::func(value);
}

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Basic utilities
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// minv
// maxv
// clampMin
// clampMax
// clampRange
//
//================================================================

template <typename Type>
sysinline Type minv(const Type& A, const Type& B)
    MISSING_FUNCTION_BODY

template <typename Type>
sysinline Type maxv(const Type& A, const Type& B)
    MISSING_FUNCTION_BODY

template <typename Type>
sysinline Type clampMin(const Type& X, const Type& minValue)
    MISSING_FUNCTION_BODY

template <typename Type>
sysinline Type clampMax(const Type& X, const Type& maxValue)
    MISSING_FUNCTION_BODY

template <typename Type>
sysinline Type clampRange(const Type& X, const Type& minValue, const Type& maxValue)
    MISSING_FUNCTION_BODY

//================================================================
//
// minv/maxv for N arguments
//
//================================================================

#define TMP_MACRO(func) \
    \
    template <typename Type> \
    sysinline Type func(const Type& v0, const Type& v1, const Type& v2) \
        {return func(v0, func(v1, v2));} \
    \
    template <typename Type> \
    sysinline Type func(const Type& v0, const Type& v1, const Type& v2, const Type& v3) \
        {return func(func(v0, v1), func(v2, v3));}

TMP_MACRO(minv)
TMP_MACRO(maxv)

#undef TMP_MACRO

//================================================================
//
// absv
//
//================================================================

template <typename Type>
sysinline Type absv(const Type& value)
    MISSING_FUNCTION_BODY;

//================================================================
//
// floorv
// ceilv
//
//================================================================

template <typename Type>
sysinline Type floorv(const Type& value)
    MISSING_FUNCTION_BODY;

template <typename Type>
sysinline Type ceilv(const Type& value)
    MISSING_FUNCTION_BODY;

//================================================================
//
// isPower2
//
// Returns true if the value is a power of 2,
// such as 0, 1, 2, 4, 8, 16, etc.
//
//================================================================

template <typename Type>
sysinline bool isPower2(const Type& value)
    MISSING_FUNCTION_BODY;

//================================================================
//
// square
//
//================================================================

template <typename Type>
sysinline Type square(const Type& value)
{
    return value * value;
}

//================================================================
//
// recipSqrt
//
// Should give ALMOST full precision.
//
// fastSqrt
//
// Does not handle +inf.
// Should give ALMOST full precision.
//
//================================================================

template <typename Type>
sysinline Type recipSqrt(const Type& value);

template <typename Type>
sysinline Type fastSqrt(const Type& value);

//================================================================
//
// vectorLengthSq
//
//================================================================

template <typename Vector>
sysinline VECTOR_BASE(Vector) vectorLengthSq(const Vector& vec);

//================================================================
//
// vectorLength
//
//================================================================

template <typename Vector>
sysinline auto vectorLength(const Vector& vec)
{
    return fastSqrt(vectorLengthSq(vec));
}

//================================================================
//
// VECTOR_DECOMPOSE
//
//================================================================

#define VECTOR_DECOMPOSE_EX(prefix, vec) \
    VECTOR_BASE(decltype(vec)) prefix##LengthSq; \
    VECTOR_BASE(decltype(vec)) prefix##DivLength; \
    VECTOR_BASE(decltype(vec)) prefix##Length; \
    auto prefix##Dir = vec; \
    vectorDecompose(vec, prefix##LengthSq, prefix##DivLength, prefix##Length, prefix##Dir)

#define VECTOR_DECOMPOSE(vec) \
    VECTOR_DECOMPOSE_EX(vec, vec)

//================================================================
//
// vectorNormalize
//
//================================================================

template <typename Vector>
sysinline Vector vectorNormalize(const Vector& vec)
{
    VECTOR_DECOMPOSE(vec);
    return vecDir;
}
