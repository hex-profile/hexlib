#pragma once

#include "imageRead/interpType.h"
#include "imageRead/loadMode.h"
#include "numbers/float16/float16Type.h"
#include "data/gpuMatrix.h"
#include "imageRead/borderMode.h"
#include "readInterpolate/cubicCoeffs.h"

//================================================================
//
// ReadAsFloatResult
//
// Gives the type of interpolation result when interpolating
// an image of the specified type.
//
//================================================================

template <typename Src>
struct ExtendToFloat;

////

#define TMP_MACRO(Src, Dst) \
    template <> \
    struct ExtendToFloat<Src> {using T = Dst;};

////

TMP_MACRO(float16, float32)
TMP_MACRO(float32, float32)
TMP_MACRO(float64, float64)

TMP_MACRO(int8, float32)
TMP_MACRO(uint8, float32)
TMP_MACRO(int16, float32)
TMP_MACRO(uint16, float32)
TMP_MACRO(int32, float32)
TMP_MACRO(uint32, float32)

TMP_MACRO(int64, float64)
TMP_MACRO(uint64, float64)

////

#undef TMP_MACRO

//================================================================
//
// ReadAsFloatResult
//
//================================================================

template <typename VectorType>
struct ReadAsFloatResult
{
    using BaseType = VECTOR_BASE(VectorType);
    using BaseExtended = typename ExtendToFloat<BaseType>::T;

    using T = VECTOR_REBASE(VectorType, BaseExtended);
};

//================================================================
//
// InterpolateSpace
//
//================================================================

template <InterpType interpType, BorderMode borderMode>
struct InterpolateSpace;

//================================================================
//
// InterpolateSpace<INTERP_NEAREST, BORDER_CLAMP>
//
//================================================================

template <>
struct InterpolateSpace<INTERP_NEAREST, BORDER_CLAMP>
{
    template <typename Src, typename LoadElement>
    static sysinline typename ReadAsFloatResult<Src>::T func
    (
        const GpuMatrix<const Src>& srcMatrix,
        float32 Xs, float32 Ys
    )
    {
        using VectorFloat = typename ReadAsFloatResult<Src>::T;

        MATRIX_EXPOSE_EX(srcMatrix, src);

        if_not (srcSizeX > 0 && srcSizeY > 0)
            return convertNearest<VectorFloat>(0);

        Space srcLastX = srcSizeX-1;
        Space srcLastY = srcSizeY-1;

        ////

        Space iX = convertDown<Space>(Xs);
        Space iY = convertDown<Space>(Ys);

        ////

        Space fitX = clampRange(iX, 0, srcLastX);
        Space fitY = clampRange(iY, 0, srcLastY);

        ////

        return convertNearest<VectorFloat>(LoadElement::func(unsafePtr(MATRIX_POINTER(src, fitX, fitY), 1)));
    }
};

//================================================================
//
// InterpolateSpace<INTERP_LINEAR, BORDER_CLAMP>
//
//================================================================

template <>
struct InterpolateSpace<INTERP_LINEAR, BORDER_CLAMP>
{
    template <typename Src, typename LoadElement>
    static sysinline typename ReadAsFloatResult<Src>::T func
    (
        const GpuMatrix<const Src>& srcMatrix,
        float32 Xs, float32 Ys
    )
    {
        using VectorFloat = typename ReadAsFloatResult<Src>::T;

        MATRIX_EXPOSE_EX(srcMatrix, src);

        if_not (srcSizeX > 0 && srcSizeY > 0)
            return convertNearest<VectorFloat>(0);

        Space srcLastX = srcSizeX-1;
        Space srcLastY = srcSizeY-1;

        ////

        float32 Xg = Xs - 0.5f;
        float32 Yg = Ys - 0.5f;

        Space iX = convertDown<Space>(Xg);
        Space iY = convertDown<Space>(Yg);

        float32 oX = Xg - iX;
        float32 oY = Yg - iY;

        ////

        Space X0 = clampRange(iX, 0, srcLastX);
        Space Y0 = clampRange(iY, 0, srcLastY);

        Space X1 = clampRange(iX + 1, 0, srcLastX);
        Space Y1 = clampRange(iY + 1, 0, srcLastY);

        ////

        auto ptrY0 = MATRIX_POINTER(src, 0, Y0);
        auto ptrY1 = MATRIX_POINTER(src, 0, Y1);

        VectorFloat v00 = convertNearest<VectorFloat>(LoadElement::func(unsafePtr(ptrY0 + X0, 1)));
        VectorFloat v10 = convertNearest<VectorFloat>(LoadElement::func(unsafePtr(ptrY0 + X1, 1)));
        VectorFloat v0 = v00 + (v10 - v00) * oX;

        VectorFloat v01 = convertNearest<VectorFloat>(LoadElement::func(unsafePtr(ptrY1 + X0, 1)));
        VectorFloat v11 = convertNearest<VectorFloat>(LoadElement::func(unsafePtr(ptrY1 + X1, 1)));
        VectorFloat v1 = v01 + (v11 - v01) * oX;

        VectorFloat value = v0 + (v1 - v0) * oY;

        return value;
    }
};

//================================================================
//
// InterpolateSpace<INTERP_CUBIC, BORDER_CLAMP>
//
//================================================================

template <>
struct InterpolateSpace<INTERP_CUBIC, BORDER_CLAMP>
{
    template <typename Src, typename LoadElement>
    static sysinline typename ReadAsFloatResult<Src>::T func
    (
        const GpuMatrix<const Src>& srcMatrix,
        float32 Xs, float32 Ys
    )
    {
        using VectorFloat = typename ReadAsFloatResult<Src>::T;

        MATRIX_EXPOSE_EX(srcMatrix, src);

        if_not (srcSizeX > 0 && srcSizeY > 0)
            return convertNearest<VectorFloat>(0);

        Space srcLastX = srcSizeX-1;
        Space srcLastY = srcSizeY-1;

        ////

        float32 Xg = Xs - 0.5f;
        float32 Yg = Ys - 0.5f;

        Space bX = convertDown<Space>(Xg);
        Space bY = convertDown<Space>(Yg);

        float32 dX = Xg - convertFloat32(bX);
        float32 dY = Yg - convertFloat32(bY);

        ////

        auto ptrY0 = MATRIX_POINTER(src, 0, clampRange(bY - 1, 0, srcLastY));
        auto ptrY1 = MATRIX_POINTER(src, 0, clampRange(bY + 0, 0, srcLastY));
        auto ptrY2 = MATRIX_POINTER(src, 0, clampRange(bY + 1, 0, srcLastY));
        auto ptrY3 = MATRIX_POINTER(src, 0, clampRange(bY + 2, 0, srcLastY));

        Space X0 = clampRange(bX - 1, 0, srcLastX);
        Space X1 = clampRange(bX + 0, 0, srcLastX);
        Space X2 = clampRange(bX + 1, 0, srcLastX);
        Space X3 = clampRange(bX + 2, 0, srcLastX);

        ////

        float32 CX0, CX1, CX2, CX3;
        CubicCoeffs::func(dX, CX0, CX1, CX2, CX3);

        #define READ(X, pY) \
            (convertNearest<VectorFloat>(LoadElement::func(unsafePtr((pY) + (X), 1))))

        VectorFloat V0 =
            CX0 * READ(X0, ptrY0) +
            CX1 * READ(X1, ptrY0) +
            CX2 * READ(X2, ptrY0) +
            CX3 * READ(X3, ptrY0);

        VectorFloat V1 =
            CX0 * READ(X0, ptrY1) +
            CX1 * READ(X1, ptrY1) +
            CX2 * READ(X2, ptrY1) +
            CX3 * READ(X3, ptrY1);

        VectorFloat V2 =
            CX0 * READ(X0, ptrY2) +
            CX1 * READ(X1, ptrY2) +
            CX2 * READ(X2, ptrY2) +
            CX3 * READ(X3, ptrY2);

        VectorFloat V3 =
            CX0 * READ(X0, ptrY3) +
            CX1 * READ(X1, ptrY3) +
            CX2 * READ(X2, ptrY3) +
            CX3 * READ(X3, ptrY3);

        ////

        #undef READ

        ////

        float32 CY0, CY1, CY2, CY3;
        CubicCoeffs::func(dY, CY0, CY1, CY2, CY3);

        return CY0*V0 + CY1*V1 + CY2*V2 + CY3*V3;
    }
};

//================================================================
//
// interpolateSpace
//
//================================================================

template <InterpType interpType, BorderMode borderMode, typename Src>
sysinline typename ReadAsFloatResult<Src>::T interpolateSpace(const GpuMatrix<Src>& src, const Point<float32>& pos)
    {return InterpolateSpace<interpType, borderMode>::template func<Src, LoadNormal>(src, pos.X, pos.Y);}

template <InterpType interpType, BorderMode borderMode, typename Src>
sysinline typename ReadAsFloatResult<Src>::T interpolateSpaceViaSamplerCache(const GpuMatrix<Src>& src, const Point<float32>& pos)
    {return InterpolateSpace<interpType, borderMode>::template func<Src, LoadViaSamplerCache>(src, pos.X, pos.Y);}
