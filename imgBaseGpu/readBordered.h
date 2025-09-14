#pragma once

#include "imageRead/borderMode.h"
#include "data/matrix.h"
#include "imageRead/loadMode.h"
#include "numbers/mathIntrinsics.h"

//================================================================
//
// mirrorSpaceOneReflection
//
// Correct results only for N >= 1!
//
// On Kepler, generates 5 instructions
//
//================================================================

sysinline Space mirrorSpaceOneReflection(Space k, Space N)
{
    if (k < 0)
        k = ~k; // k = -k - 1;

    else if (k >= N)
        k = ~k + N + N; // k = -k - 1 + N + N

    return k;
}

//================================================================
//
// modulo2
//
//================================================================

sysinline float32 modulo2(float32 x)
{
    //
    // Divide by 2, take fractional part, multiply by 2
    //
    // float32 value = 0.5f * x;
    // 2 * (value - floorv(value));
    //

    return x - 2.f * floorv(0.5f * x); // optimized
}

//================================================================
//
// normCoordMirror
//
// On Kepler, generates 5 instructions
//
//================================================================

sysinline float32 normCoordMirror(float32 x)
{
    // modulo 2
    float32 value = modulo2(x);

    // Tent centered at 1
    float32 tent = absv(value - 1);

    return saturatev(tent);
}

//----------------------------------------------------------------

sysinline Point<float32> normCoordMirror(const Point<float32>& pos)
{
    return point(normCoordMirror(pos.X), normCoordMirror(pos.Y));
}

//================================================================
//
// ReadBordered
//
// Read element with border handling. SCROLL UP TO SEE USAGE EXAMPLES.
//
//================================================================

template <BorderMode borderMode>
struct ReadBordered;

//================================================================
//
// ReadBordered<BORDER_ZERO>
//
//================================================================

template <>
struct ReadBordered<BORDER_ZERO>
{
    template <typename LoadElement, typename Pointer, typename Pitch>
    static sysinline auto func(const MatrixEx<Pointer, Pitch>& img, Space X, Space Y)
    {
        using Src = typename PtrElemType<Pointer>::T;

        MATRIX_EXPOSE(img);

        bool valid = MATRIX_VALID_ACCESS(img, X, Y);
        auto ptr = MATRIX_POINTER(img, X, Y);
        auto result = zeroOf<Src>();

        if (valid)
            result = LoadElement::func(&helpRead(*ptr));

        return result;
    }
};

//================================================================
//
// ReadBordered<BORDER_CLAMP>
//
//================================================================

template <>
struct ReadBordered<BORDER_CLAMP>
{
    template <typename LoadElement, typename Pointer, typename Pitch>
    static sysinline auto func(const MatrixEx<Pointer, Pitch>& img, Space X, Space Y)
    {
        using Src = typename PtrElemType<Pointer>::T;

        MATRIX_EXPOSE(img);

        Space lastX = imgSizeX-1;
        Space lastY = imgSizeY-1;

        bool valid = (lastX >= 0) && (lastY >= 0);

        X = clampRange(X, 0, lastX);
        Y = clampRange(Y, 0, lastY);

        auto ptr = MATRIX_POINTER(img, X, Y);

        auto result = zeroOf<Src>();
        if (valid) result = LoadElement::func(&helpRead(*ptr));

        return result;
    }
};

//================================================================
//
// ReadBordered<BORDER_MIRROR>
//
//================================================================

template <>
struct ReadBordered<BORDER_MIRROR>
{
    template <typename LoadElement, typename Pointer, typename Pitch>
    static sysinline auto func(const MatrixEx<Pointer, Pitch>& img, Space X, Space Y)
    {
        using Src = typename PtrElemType<Pointer>::T;

        MATRIX_EXPOSE(img);

        ////

        X = mirrorSpaceOneReflection(X, imgSizeX);
        Y = mirrorSpaceOneReflection(Y, imgSizeY);

        ////

        auto ptr = MATRIX_POINTER(img, X, Y);
        bool valid = MATRIX_VALID_ACCESS(img, X, Y);

        auto result = zeroOf<Src>();
        if (valid) result = LoadElement::func(&helpRead(*ptr));

        return result;
    }
};

//================================================================
//
// ReadBordered<BORDER_WRAP>
//
//================================================================

template <>
struct ReadBordered<BORDER_WRAP>
{
    template <typename LoadElement, typename Pointer, typename Pitch>
    static sysinline auto func(const MatrixEx<Pointer, Pitch>& img, Space X, Space Y)
    {
        using Src = typename PtrElemType<Pointer>::T;

        MATRIX_EXPOSE(img);

        auto result = zeroOf<Src>();

        ////

        if (MATRIX_VALID_ACCESS(img, X, Y))
            goto readIt;

        ////

        X = X % imgSizeX;
        if (X < 0) X += imgSizeX;

        Y = Y % imgSizeY;
        if (Y < 0) Y += imgSizeY;

        if_not (MATRIX_VALID_ACCESS(img, X, Y))
            goto skipReading;

    readIt:

        result = LoadElement::func(&MATRIX_READ(img, X, Y));

    skipReading:

        return result;

    }
};

//================================================================
//
// readBordered
//
// Read element with border handling.
//
//================================================================

template <BorderMode borderMode, typename Pointer, typename Pitch>
sysinline auto readBordered(const MatrixEx<Pointer, Pitch>& img, Space X, Space Y)
    {return ReadBordered<borderMode>::template func<struct LoadNormal>(img, X, Y);}

template <BorderMode borderMode, typename Pointer, typename Pitch>
sysinline auto readBordered(const MatrixEx<Pointer, Pitch>& img, const Point<Space>& pos)
    {return ReadBordered<borderMode>::template func<struct LoadNormal>(img, pos.X, pos.Y);}

//----------------------------------------------------------------

template <BorderMode borderMode, typename Pointer, typename Pitch>
sysinline auto readBorderedViaSamplerCache(const MatrixEx<Pointer, Pitch>& img, Space X, Space Y)
    {return ReadBordered<borderMode>::template func<struct LoadViaSamplerCache>(img, X, Y);}

template <BorderMode borderMode, typename Pointer, typename Pitch>
sysinline auto readBorderedViaSamplerCache(const MatrixEx<Pointer, Pitch>& img, const Point<Space>& pos)
    {return ReadBordered<borderMode>::template func<struct LoadViaSamplerCache>(img, pos.X, pos.Y);}
