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
// gaborSimpleBank.inl
//
//================================================================

#define FUNCNAME gaborExample
#define GABOR_ENABLED 1
#define ENVELOPE_ENABLED 0
#define COMPRESS_FACTOR 2

#define GABOR_INPUT_PIXEL float16
#define GABOR_COMPLEX_PIXEL float16_x2

#define GABOR_ORIENT_COUNT ORIENTATION_COUNT
#define GABOR_BORDER_MODE BORDER_MIRROR

#define GABOR_PARAMS PostprocessParams
#define GABOR_PREPROCESS_IMAGES
#define GABOR_PREPROCESS(value, texPos)
#define GABOR_POSTPROCESS_IMAGES
#define GABOR_POSTPROCESS(value)

//----------------------------------------------------------------

#define HORIZONTAL 1
# include "gaborFilters/gaborKernelsForSingleBigImage.inl"
#undef HORIZONTAL

#define HORIZONTAL 0
# include "gaborFilters/gaborKernelsForSingleBigImage.inl"
#undef HORIZONTAL

//----------------------------------------------------------------

#undef FUNCNAME
#undef GABOR_ENABLED
#undef ENVELOPE_ENABLED
#undef COMPRESS_FACTOR

#undef GABOR_INPUT_PIXEL
#undef GABOR_COMPLEX_PIXEL
#undef GABOR_ORIENT_COUNT
#undef GABOR_BORDER_MODE
#undef GABOR_PARAMS
#undef GABOR_PREPROCESS_IMAGES
#undef GABOR_PREPROCESS
#undef GABOR_POSTPROCESS_IMAGES
#undef GABOR_POSTPROCESS

//================================================================
//
// gaborExampleFunc
//
//================================================================

#if HOSTCODE

void gaborExampleFunc
(
    const GpuMatrix<const float16>& src,
    const GpuMatrix<const float32_x2>& circleTable,
    const GpuLayeredMatrix<float16_x2>& dst,
    bool demodulateOutput,
    bool horizontallyFirst,
    bool uncachedVersion,
    stdPars(GpuProcessKit)
)
{
    (horizontallyFirst ? gaborExampleProcessFullHor : gaborExampleProcessFullVer)
    (
        src,
        circleTable,
        dst,
        demodulateOutput,
        PostprocessParams{},
        uncachedVersion,
        stdPass
    );
}

#endif
