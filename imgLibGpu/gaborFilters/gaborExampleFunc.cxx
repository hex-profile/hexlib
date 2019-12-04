#if HOSTCODE
#include "gaborExampleFunc.h"
#endif

#include "gaborFilters/gaborKernelsForSingleBigImage.h"

//================================================================
//
// gaborExampleBank
//
//================================================================

#define GABOR_BANK gaborExampleBank
# include "gaborExampleBank.inl"

#define ORIENTATION_COUNT 7
COMPILE_ASSERT(gaborExampleBankOrientCount == ORIENTATION_COUNT);

//================================================================
//
// PostprocessParams
//
//================================================================

struct PostprocessParams
{
};

//================================================================
//
// postprocessAction
//
//================================================================

sysinline void postprocessAction
(
    const Point<Space>& dstPos,
    const Point<Space>& dstSize,
    PREP_ENUM_INDEXED(ORIENTATION_COUNT, float32_x2& sum),
    const PostprocessParams& params
)
{
}

//================================================================
//
// gaborSimpleBank.inl
//
//================================================================

#define FUNCNAME gaborExample
#define MASK_ENABLED 0

#define INPUT_PIXEL float16
#define COMPLEX_PIXEL float16_x2

#define ORIENT_COUNT ORIENTATION_COUNT
#define COMPRESS_OCTAVES 1
#define INPUT_BORDER_MODE BORDER_MIRROR

#define POSTPROCESS_PARAMS PostprocessParams
#define POSTPROCESS_ACTION postprocessAction

//----------------------------------------------------------------

#define HORIZONTAL 1
# include "gaborFilters/gaborKernelsForSingleBigImage.inl"
#undef HORIZONTAL

#define HORIZONTAL 0
# include "gaborFilters/gaborKernelsForSingleBigImage.inl"
#undef HORIZONTAL

//----------------------------------------------------------------

#undef FUNCNAME
#undef GABOR_BANK

#undef INPUT_PIXEL
#undef COMPLEX_PIXEL

#undef ORIENT_COUNT
#undef COMPRESS_OCTAVES
#undef INPUT_BORDER_MODE

#undef POSTPROCESS_PARAMS 
#undef POSTPROCESS_ACTION

//================================================================
//
// gaborExampleFunc
//
//================================================================

#if HOSTCODE

stdbool gaborExampleFunc
(
    const GpuMatrix<const float16>& src,
    const GpuMatrix<const float32_x2>& circleTable,
    const GpuLayeredMatrix<float16_x2>& dst,
    bool intermIsHorizontal,
    bool simpleVersion,
    stdPars(GpuProcessKit)
)
{
    require
    (
        (intermIsHorizontal ? gaborExampleProcessFullHor<> : gaborExampleProcessFullVer<>)
        (
            src, 
            circleTable, 
            dst, 
            PostprocessParams{},
            simpleVersion, 
            stdPass
        )
    );

    ////

    returnTrue;
}

#endif
