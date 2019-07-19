#pragma once

#include "types/lt/ltBase.h"
#include "numbers/interface/numberInterface.h"

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Traits
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// VectorBaseImpl<LinearTransform>
//
//================================================================

template <typename Type>
struct VectorBaseImpl<LinearTransform<Type>>
{
    using T = Type;
};

//================================================================
//
// VectorRebaseImpl<LinearTransform>
//
//================================================================

template <typename OldBase, typename NewBase>
struct VectorRebaseImpl<LinearTransform<OldBase>, NewBase>
{
    using T = LinearTransform<NewBase>;
};

//================================================================
//
// def<LinearTransform>
//
//================================================================

template <typename Type>
struct DefImpl<LinearTransform<Type>>
{
    static sysinline auto func(const LinearTransform<Type>& value)
    {
        return def(value.C1) && def(value.C0);
    }
};

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Conversions
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// ConvertFamilyImpl<LinearTransform<T>>
//
//================================================================

struct LtFamily;

//----------------------------------------------------------------

template <typename Type>
struct ConvertFamilyImpl<LinearTransform<Type>>
{
    using T = LtFamily;
};

//================================================================
//
// ConvertImpl
//
// LinearTransform -> LinearTransform
//
//================================================================

template <ConvertCheck check, Rounding rounding, ConvertHint hint>
struct ConvertImpl<LtFamily, LtFamily, check, rounding, hint>
{
    template <typename SrcTransform, typename DstTransform>
    struct Convert
    {
        using SrcBase = VECTOR_BASE(SrcTransform);
        using DstBase = VECTOR_BASE(DstTransform);

        using BaseImpl = typename ConvertScalar<SrcBase, DstBase, check, rounding, hint>::Code;

        static sysinline LinearTransform<DstBase> func(const LinearTransform<SrcBase>& srcTransform)
        {
            return linearTransform(BaseImpl::func(srcTransform.C1), BaseImpl::func(srcTransform.C0));
        }
    };
};

//================================================================
//
// ConvertImplFlag<LinearTransform, LinearTransform>
//
//================================================================

template <Rounding rounding, ConvertHint hint>
struct ConvertImplFlag<LtFamily, LtFamily, rounding, hint>
{
    template <typename SrcTransform, typename DstTransform>
    struct Convert
    {
        using SrcBase = VECTOR_BASE(SrcTransform);
        using DstBase = VECTOR_BASE(DstTransform);

        using BaseImpl = typename ConvertScalarFlag<SrcBase, DstBase, rounding, hint>::Code;

        static sysinline bool func(const LinearTransform<SrcBase>& src, LinearTransform<DstBase>& dst)
        {
            bool s1 = BaseImpl::func(src.C1, dst.C1);
            bool s0 = BaseImpl::func(src.C0, dst.C0);

            return s1 && s0;
        };
    };
};

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// Operations
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// ltApply
//
//================================================================

// ```
template <typename ValueType, typename CoeffType>
sysinline ValueType ltApply(const ValueType& value, const LinearTransform<CoeffType>& transform)
{
    return value * transform.C1 + transform.C0;
}

//================================================================
//
// ltByTwoPoints
//
//================================================================

template <typename Type>
sysinline LinearTransform<Type> ltByTwoPoints
(
    const Type& src0,
    const Type& dst0,
    const Type& src1,
    const Type& dst1
)
{
    Type div = convertNearest<Type>(1) / (src0 - src1);

    return linearTransform
    (
        (dst0 - dst1) * div,
        (src0 * dst1 - dst0 * src1) * div
    );
}

//================================================================
//
// ltInverse
//
//================================================================

template <typename Type>
sysinline LinearTransform<Type> ltInverse(const LinearTransform<Type>& lt)
{
    Type div = convertNearest<Type>(1) / lt.C1;

    return linearTransform
    (
        div,
        -lt.C0 * div
    );
}

//================================================================
//
// ltCombine
//
//================================================================

template <typename Type>
sysinline LinearTransform<Type> ltCombine(const LinearTransform<Type>& A, const LinearTransform<Type>& B)
{
    return linearTransform
    (
        A.C1 * B.C1,
        A.C0 * B.C1 + B.C0
    );
}

//================================================================
//
// ltPassthru
//
//================================================================

template <typename Type>
sysinline LinearTransform<Type> ltPassthru()
{
    return linearTransform
    (
        convertNearest<Type>(1),
        convertNearest<Type>(0)
    );
}

//================================================================
//
// ltOutputZero
//
//================================================================

template <typename Type>
sysinline LinearTransform<Type> ltOutputZero()
{
    return linearTransform
    (
        convertNearest<Type>(0),
        convertNearest<Type>(0)
    );
}
