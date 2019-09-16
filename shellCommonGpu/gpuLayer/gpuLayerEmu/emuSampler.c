#if HEXLIB_PLATFORM == 0

#include "emuSampler.h"

#include "vectorTypes/vectorType.h"
#include "vectorTypes/vectorOperations.h"
#include "gpuDevice/devSampler/devSamplerEmu.h"
#include "errorLog/errorLog.h"
#include "readBordered.h"
#include "errorLog/debugBreak.h"

//================================================================
//
// ExtendElementType
//
// Extends the element type to the sampler result type,
// when "element" read mode is used.
//
//================================================================

template <typename MemType>
struct ExtendElementType;

//----------------------------------------------------------------

#define TMP_MACRO(Src, Dst, resultExpr) \
    \
    template <> \
    struct ExtendElementType<Src> \
        {using T = Dst;}; \
    \
    inline Dst extendElementType(const Src& src) \
        {return resultExpr;}

TMP_MACRO(int8, int32, src)
TMP_MACRO(uint8, uint32, src)
TMP_MACRO(int16, int32, src)
TMP_MACRO(uint16, uint32, src)
TMP_MACRO(int32, int32, src)
TMP_MACRO(uint32, uint32, src)
TMP_MACRO(float16, float32, unpackFloat16(src))
TMP_MACRO(float32, float32, src)

#undef TMP_MACRO

//----------------------------------------------------------------

#define TMP_MACRO(Src, Dst, resultExpr) \
    \
    template <> \
    struct ExtendElementType<Src> \
        {using T = Dst;}; \
    \
    inline Dst extendElementType(const Src& src) \
        {return resultExpr;}

TMP_MACRO(int8_x2, int32_x2, make_int32_x2(src.x, src.y))
TMP_MACRO(uint8_x2, uint32_x2, make_uint32_x2(src.x, src.y))
TMP_MACRO(int16_x2, int32_x2, make_int32_x2(src.x, src.y))
TMP_MACRO(uint16_x2, uint32_x2, make_uint32_x2(src.x, src.y))
TMP_MACRO(int32_x2, int32_x2, make_int32_x2(src.x, src.y))
TMP_MACRO(uint32_x2, uint32_x2, make_uint32_x2(src.x, src.y))
TMP_MACRO(float16_x2, float32_x2, make_float32_x2(unpackFloat16(src.x), unpackFloat16(src.y)))
TMP_MACRO(float32_x2, float32_x2, make_float32_x2(src.x, src.y))

#undef TMP_MACRO

//----------------------------------------------------------------

#define TMP_MACRO(Src, Dst, resultExpr) \
    \
    template <> \
    struct ExtendElementType<Src> \
        {using T = Dst;}; \
    \
    inline Dst extendElementType(const Src& src) \
        {return resultExpr;}

TMP_MACRO(int8_x4, int32_x4, make_int32_x4(src.x, src.y, src.z, src.w))
TMP_MACRO(uint8_x4, uint32_x4, make_uint32_x4(src.x, src.y, src.z, src.w))
TMP_MACRO(int16_x4, int32_x4, make_int32_x4(src.x, src.y, src.z, src.w))
TMP_MACRO(uint16_x4, uint32_x4, make_uint32_x4(src.x, src.y, src.z, src.w))
TMP_MACRO(int32_x4, int32_x4, make_int32_x4(src.x, src.y, src.z, src.w))
TMP_MACRO(uint32_x4, uint32_x4, make_uint32_x4(src.x, src.y, src.z, src.w))
TMP_MACRO(float16_x4, float32_x4, make_float32_x4(unpackFloat16(src.x), unpackFloat16(src.y), unpackFloat16(src.z), unpackFloat16(src.w)))
TMP_MACRO(float32_x4, float32_x4, make_float32_x4(src.x, src.y, src.z, src.w))

#undef TMP_MACRO

//================================================================
//
// NormFloatType
//
// Extends the element type to the sampler result type,
// when "normalized float" read mode is used.
//
//================================================================

template <typename MemType>
struct NormFloatType;

//----------------------------------------------------------------

#define TMP_MACRO(Src, Dst, expression) \
    \
    template <> \
    struct NormFloatType<Src> \
        {using T = Dst;}; \
    \
    inline Dst normFloatType(const Src& src) \
        {return expression;}

TMP_MACRO(uint8, float32, src * (1.f / 0xFF))
TMP_MACRO(uint16, float32, src * (1.f / 0xFFFF))
TMP_MACRO(uint32, float32, src * (1.f / 0xFFFFFFFF))

TMP_MACRO(int8, float32, clampMin(src * (1.f / 0x7F), -1.f))
TMP_MACRO(int16, float32, clampMin(src * (1.f / 0x7FFF), -1.f))
TMP_MACRO(int32, float32, clampMin(src * (1.f / 0x7FFFFFFF), -1.f))

TMP_MACRO(float16, float32, unpackFloat16(src))
TMP_MACRO(float32, float32, src)

#undef TMP_MACRO

//----------------------------------------------------------------

#define TMP_MACRO(Src, Dst) \
    \
    template <> \
    struct NormFloatType<Src> \
        {using T = Dst;}; \
    \
    inline Dst normFloatType(const Src& src) \
        {return make_##Dst(normFloatType(src.x), normFloatType(src.y));}

TMP_MACRO(uint8_x2, float32_x2)
TMP_MACRO(uint16_x2, float32_x2)
TMP_MACRO(uint32_x2, float32_x2)

TMP_MACRO(int8_x2, float32_x2)
TMP_MACRO(int16_x2, float32_x2)
TMP_MACRO(int32_x2, float32_x2)

TMP_MACRO(float16_x2, float32_x2)
TMP_MACRO(float32_x2, float32_x2)

#undef TMP_MACRO

//----------------------------------------------------------------

#define TMP_MACRO(Src, Dst) \
    \
    template <> \
    struct NormFloatType<Src> \
        {using T = Dst;}; \
    \
    inline Dst normFloatType(const Src& src) \
        {return make_##Dst(normFloatType(src.x), normFloatType(src.y), normFloatType(src.z), normFloatType(src.w));}

TMP_MACRO(uint8_x4, float32_x4)
TMP_MACRO(uint16_x4, float32_x4)
TMP_MACRO(uint32_x4, float32_x4)

TMP_MACRO(int8_x4, float32_x4)
TMP_MACRO(int16_x4, float32_x4)
TMP_MACRO(int32_x4, float32_x4)

TMP_MACRO(float16_x4, float32_x4)
TMP_MACRO(float32_x4, float32_x4)

#undef TMP_MACRO

//================================================================
//
// SamplerResultType
//
// The type of sampler result and converter function,
// according to normalized float flag.
//
//================================================================

template <bool normalizedFloat, typename Src>
struct SamplerResultType;

//----------------------------------------------------------------

template <typename Src>
struct SamplerResultType<false, Src>
{
    using T = typename ExtendElementType<Src>::T;

    static inline T func(const Src& value)
        {return extendElementType(value);}
};

//----------------------------------------------------------------

template <typename Src>
struct SamplerResultType<true, Src> : public NormFloatType<Src>
{
    using T = typename NormFloatType<Src>::T;

    static inline T func(const Src& value)
        {return normFloatType(value);}
};

//================================================================
//
// EmuSamplerInfo2D
//
// Sampler information for 2D data.
//
//================================================================

template <typename MemType>
struct EmuSamplerInfo2D
{
    Matrix<const MemType> matrix;
    Point<float32> coordScale;
};

//================================================================
//
// readSpaceNearest
//
//================================================================

template <BorderMode srcBorder, typename Src>
static inline Src readSpaceNearest(const Matrix<Src>& srcMatrix, float32 X, float32 Y)
{
    // from space to grid coordinates: I = X - 0.5; the nearest is floor(I + 0.5)
    Space iX = convertDown<Space>(X);
    Space iY = convertDown<Space>(Y);

    return readBordered<srcBorder>(srcMatrix, iX, iY);
}

template <BorderMode srcBorder, typename Src>
static inline Src readSpaceNearest(const Matrix<Src>& srcMatrix, const Point<float32>& pos)
    {return readSpaceNearest<srcBorder>(srcMatrix, pos.X, pos.Y);}

//================================================================
//
// emuSamplerRead2D
//
// Implementation for pure element reading (no interpolation).
// Converts to normalized float, if required.
//
//================================================================

template <typename MemType, BorderMode borderMode, bool normalizedFloat>
void EMU_SAMPLER_TEX_DECL emuSamplerRead2D(const EmuSamplerData& data, float32 X, float32 Y, void* result)
{
    const EmuSamplerInfo2D<MemType>& info = (const EmuSamplerInfo2D<MemType>&) data;

    // Normalized coords support
    X *= info.coordScale.X;
    Y *= info.coordScale.Y;

    // Read element
    MemType value = readSpaceNearest<borderMode>(info.matrix, X, Y);

    // Extend integer type or convert to normalized float
    using Result = SamplerResultType<normalizedFloat, MemType>;
    * (typename Result::T*) result = Result::func(value);
}

//================================================================
//
// emuSamplerLinear2D
//
// Implementation for pure element reading (no interpolation).
// Converts to normalized float, if required.
//
//================================================================

template <typename MemType, BorderMode borderMode>
static void EMU_SAMPLER_TEX_DECL emuSamplerLinear2D(const EmuSamplerData& data, float32 X, float32 Y, void* result)
{
    // Result type
    using Result = SamplerResultType<true, MemType>;

    //
    const EmuSamplerInfo2D<MemType>& info = (const EmuSamplerInfo2D<MemType>&) data;

    // Normalized coords support
    X *= info.coordScale.X;
    Y *= info.coordScale.Y;

    // From space to grid coordinates
    X -= 0.5f;
    Y -= 0.5f;

    // Integer and fractional part
    Space iX = convertDown<Space>(X);
    Space iY = convertDown<Space>(Y);
    float32 oX = X - iX;
    float32 oY = Y - iY;

    //
    // Read 4 elements
    //

    const Matrix<const MemType>& src = info.matrix;
    MATRIX_EXPOSE(src);

    using ResultType = typename Result::T;

    ResultType v00;
    ResultType v10;
    ResultType v01;
    ResultType v11;

    if
    (
        // even if overflow in iX + 1, this check should work
        iX >= 0 && iX <= srcSizeX - 2 &&
        iY >= 0 && iY <= srcSizeY - 2
    )
    {
        MatrixPtr(const MemType) ptr0 = MATRIX_POINTER(src, iX, iY);
        MatrixPtr(const MemType) ptr1 = ptr0 + srcMemPitch;

        v00 = Result::func(ptr0[0]);
        v10 = Result::func(ptr0[1]);
        v01 = Result::func(ptr1[0]);
        v11 = Result::func(ptr1[1]);
    }
    else
    {
        v00 = Result::func(readBordered<borderMode>(src, iX+0, iY+0));
        v10 = Result::func(readBordered<borderMode>(src, iX+1, iY+0));
        v01 = Result::func(readBordered<borderMode>(src, iX+0, iY+1));
        v11 = Result::func(readBordered<borderMode>(src, iX+1, iY+1));
    }

    ////

    ResultType v0 = v00 + (v10 - v00) * oX;
    ResultType v1 = v01 + (v11 - v01) * oX;

    ResultType value = v0 + (v1 - v0) * oY;

    * (ResultType*) result = value;
}

//================================================================
//
// setupSamplerImage
//
//================================================================

template <typename MemType>
static stdbool setupSamplerImage
(
    EmuSamplerState& state,
    GpuAddrU imageBaseAddr,
    Space imageBytePitch,
    const Point<Space>& imageSize,
    BorderMode borderMode,
    bool linearInterpolation,
    bool readNormalizedFloat,
    bool normalizedCoords,
    stdPars(ErrorLogKit)
)
{
    state.reset();
    auto& info = state.data.recast<EmuSamplerInfo2D<MemType>>();

    ////

    info.matrix.assign((const MemType*) imageBaseAddr, imageBytePitch / Space(sizeof(MemType)), imageSize.X, imageSize.Y);

    //
    // Normalize to float check
    //

    using MemChannel = VECTOR_BASE(MemType);

    if (TYPE_IS_BUILTIN_FLOAT(MemChannel))
        readNormalizedFloat = false;

    if (readNormalizedFloat)
        REQUIRE(TYPE_IS_BUILTIN_INT(MemChannel) && sizeof(MemChannel) <= sizeof(uint16));

    //
    // Normalized coords
    //

    info.coordScale = point(1.f);

    if (normalizedCoords)
        info.coordScale = convertFloat32(clampMin(imageSize, 1));

    //
    // Select function
    //

    EmuSamplerTex2D* func = 0;

    if (linearInterpolation)
    {
        REQUIRE(readNormalizedFloat || TYPE_IS_BUILTIN_FLOAT(MemChannel));

        #define TMP_MACRO(borderConst, _) \
            if (borderMode == borderConst) \
                func = emuSamplerLinear2D<MemType, borderConst>;

        BORDER_MODE_FOREACH(TMP_MACRO, _)

        #undef TMP_MACRO
    }
    else
    {
        #define TMP_MACRO(borderConst, normalizedFloat) \
            if (borderMode == borderConst) \
                func = emuSamplerRead2D<MemType, borderConst, normalizedFloat>;

        if (readNormalizedFloat)
        {
            BORDER_MODE_FOREACH(TMP_MACRO, true)
        }
        else
        {
            BORDER_MODE_FOREACH(TMP_MACRO, false)
        }

        #undef TMP_MACRO
    }


    ////

    REQUIRE(func);

    state.tex2D = func;

    returnTrue;
}

//================================================================
//
// SetupEmuSamplerImage
//
//================================================================

using SetupEmuSamplerImage = stdbool
(
    EmuSamplerState& state,
    GpuAddrU imageBaseAddr,
    Space imageBytePitch,
    const Point<Space>& imageSize,
    BorderMode borderMode,
    bool linearInterpolation,
    bool readNormalizedFloat,
    bool normalizedCoords,
    stdPars(ErrorLogKit)
);

//================================================================
//
// emuSetSamplerImage
//
//================================================================

stdbool emuSetSamplerImage
(
    const GpuSamplerLink& sampler,
    GpuAddrU imageBaseAddr,
    Space imageBytePitch,
    const Point<Space>& imageSize,
    GpuChannelType chanType,
    int rank,
    BorderMode borderMode,
    bool linearInterpolation,
    bool readNormalizedFloat,
    bool normalizedCoords,
    stdPars(ErrorLogKit)
)
{
    SetupEmuSamplerImage* setupFunc = 0;

    ////

    #define TMP_MACRO(chanT, Type) \
        if (chanType == (chanT)) \
            setupFunc = setupSamplerImage<Type>;

    ////

    if (rank == 1)
    {
        TMP_MACRO(GpuChannelInt8, int8)
        TMP_MACRO(GpuChannelUint8, uint8)
        TMP_MACRO(GpuChannelInt16, int16)
        TMP_MACRO(GpuChannelUint16, uint16)
        TMP_MACRO(GpuChannelInt32, int32)
        TMP_MACRO(GpuChannelUint32, uint32)
        TMP_MACRO(GpuChannelFloat16, float16)
        TMP_MACRO(GpuChannelFloat32, float32)
    }
    else if (rank == 2)
    {
        TMP_MACRO(GpuChannelInt8, int8_x2)
        TMP_MACRO(GpuChannelUint8, uint8_x2)
        TMP_MACRO(GpuChannelInt16, int16_x2)
        TMP_MACRO(GpuChannelUint16, uint16_x2)
        TMP_MACRO(GpuChannelInt32, int32_x2)
        TMP_MACRO(GpuChannelUint32, uint32_x2)
        TMP_MACRO(GpuChannelFloat16, float16_x2)
        TMP_MACRO(GpuChannelFloat32, float32_x2)
    }
    else if (rank == 4)
    {
        TMP_MACRO(GpuChannelInt8, int8_x4)
        TMP_MACRO(GpuChannelUint8, uint8_x4)
        TMP_MACRO(GpuChannelInt16, int16_x4)
        TMP_MACRO(GpuChannelUint16, uint16_x4)
        TMP_MACRO(GpuChannelInt32, int32_x4)
        TMP_MACRO(GpuChannelUint32, uint32_x4)
        TMP_MACRO(GpuChannelFloat16, float16_x4)
        TMP_MACRO(GpuChannelFloat32, float32_x4)
    }

    ////

    #undef TMP_MACRO

    ////

    REQUIRE(setupFunc != 0);
    REQUIRE(sampler.state != 0);
    require(setupFunc(*sampler.state, imageBaseAddr, imageBytePitch, imageSize, borderMode, linearInterpolation, readNormalizedFloat, normalizedCoords, stdPass));

    returnTrue;
}

//================================================================
//
// EmuSamplerInfo1D
//
// Sampler information for 1D data.
//
//================================================================

template <typename MemType>
struct EmuSamplerInfo1D
{
    Array<const MemType> array;
    float32 coordScale;
};

//================================================================
//
// emuSamplerRead1Dfetch
//
// Implementation for pure element reading (no interpolation).
// Converts to normalized float, if required.
//
//================================================================

template <typename MemType, bool normalizedFloat>
void emuSamplerRead1Dfetch(const EmuSamplerData& data, Space offset, void* result)
{
    const EmuSamplerInfo1D<MemType>& info = (const EmuSamplerInfo1D<MemType>&) data;

    ARRAY_EXPOSE_PREFIX(info.array, array);

    // Read element
    ensurev(DEBUG_BREAK_CHECK(SpaceU(offset) < SpaceU(arraySize)));
    MemType value = arrayPtr[offset];

    // Extend integer type or convert to normalized float
    using Result = SamplerResultType<normalizedFloat, MemType>;
    * (typename Result::T*) result = Result::func(value);
}

//================================================================
//
// setupSamplerArray
//
//================================================================

template <typename MemType>
static stdbool setupSamplerArray
(
    EmuSamplerState& state,
    GpuAddrU arrayAddr,
    Space arrayByteSize,
    BorderMode borderMode,
    bool linearInterpolation,
    bool readNormalizedFloat,
    bool normalizedCoords,
    stdPars(ErrorLogKit)
)
{
    state.reset();
    auto& info = state.data.recast<EmuSamplerInfo1D<MemType>>();

    ////

    REQUIRE(arrayByteSize >= 0);
    Space arraySize = SpaceU(arrayByteSize) / sizeof(MemType);

    info.array.assign((const MemType*) arrayAddr, arraySize);

    //
    // Normalize to float check
    //

    using MemChannel = VECTOR_BASE(MemType);

    if (readNormalizedFloat)
        REQUIRE(TYPE_IS_BUILTIN_INT(MemChannel) && sizeof(MemChannel) <= sizeof(uint16));

    //
    // Normalized coords
    //

    info.coordScale = 1.f;

    if (normalizedCoords)
        info.coordScale = convertFloat32(clampMin(arraySize, 1));

    //
    // Select function
    //

    EmuSamplerTex1Dfetch* func = 0;

    if (readNormalizedFloat)
    {
        func = emuSamplerRead1Dfetch<MemType, true>;
    }
    else
    {
        func = emuSamplerRead1Dfetch<MemType, false>;
    }

    #undef TMP_MACRO

    ////

    REQUIRE(func);
    state.tex1Dfetch = func;

    returnTrue;
}

//================================================================
//
// SetupEmuSamplerArray
//
//================================================================

using SetupEmuSamplerArray = stdbool
(
    EmuSamplerState& state,
    GpuAddrU arrayAddr,
    Space arrayByteSize,
    BorderMode borderMode,
    bool linearInterpolation,
    bool readNormalizedFloat,
    bool normalizedCoords,
    stdPars(ErrorLogKit)
);

//================================================================
//
// emuSetSamplerArray
//
//================================================================

stdbool emuSetSamplerArray
(
    const GpuSamplerLink& sampler,
    GpuAddrU arrayAddr,
    Space arrayByteSize,
    GpuChannelType chanType,
    int rank,
    BorderMode borderMode,
    bool linearInterpolation,
    bool readNormalizedFloat,
    bool normalizedCoords,
    stdPars(ErrorLogKit)
)
{
    SetupEmuSamplerArray* setupFunc = 0;

    ////

    #define TMP_MACRO(chanT, Type) \
        if (chanType == (chanT)) \
            setupFunc = setupSamplerArray<Type>;

    ////

    if (rank == 1)
    {
        TMP_MACRO(GpuChannelInt8, int8)
        TMP_MACRO(GpuChannelUint8, uint8)
        TMP_MACRO(GpuChannelInt16, int16)
        TMP_MACRO(GpuChannelUint16, uint16)
        TMP_MACRO(GpuChannelInt32, int32)
        TMP_MACRO(GpuChannelUint32, uint32)
        TMP_MACRO(GpuChannelFloat32, float32)
    }
    else if (rank == 2)
    {
        TMP_MACRO(GpuChannelInt8, int8_x2)
        TMP_MACRO(GpuChannelUint8, uint8_x2)
        TMP_MACRO(GpuChannelInt16, int16_x2)
        TMP_MACRO(GpuChannelUint16, uint16_x2)
        TMP_MACRO(GpuChannelInt32, int32_x2)
        TMP_MACRO(GpuChannelUint32, uint32_x2)
        TMP_MACRO(GpuChannelFloat32, float32_x2)
    }
    else if (rank == 4)
    {
        TMP_MACRO(GpuChannelInt8, int8_x4)
        TMP_MACRO(GpuChannelUint8, uint8_x4)
        TMP_MACRO(GpuChannelInt16, int16_x4)
        TMP_MACRO(GpuChannelUint16, uint16_x4)
        TMP_MACRO(GpuChannelInt32, int32_x4)
        TMP_MACRO(GpuChannelUint32, uint32_x4)
        TMP_MACRO(GpuChannelFloat32, float32_x4)
    }

    ////

    #undef TMP_MACRO

    ////

    REQUIRE(setupFunc != 0);
    REQUIRE(sampler.state != 0);
    require(setupFunc(*sampler.state, arrayAddr, arrayByteSize, borderMode, linearInterpolation, readNormalizedFloat, normalizedCoords, stdPass));

    returnTrue;
}

//----------------------------------------------------------------

#endif
