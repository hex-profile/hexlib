#if HOSTCODE
#include "upsampleTwice.h"
#endif

#include "gaussSincResampling/gaussSincResamplingSettings.h"
#include "gaussSincResampling/common/allTypes.h"

//================================================================
//
// Instance
//
//================================================================

#if GAUSS_SINC_RESAMPLING_HQ == 0

    static devConstant float32 FILTER0[] = {+0.00265213f, -0.00623483f, +0.01352500f, -0.02747154f, +0.05375393f, -0.10846222f, +0.29078630f, +0.89693080f, -0.16504324f, +0.07548820f, -0.03850145f, +0.01940955f, -0.00926903f, +0.00411005f, -0.00167367f};
    static devConstant float32 FILTER1[] = {-0.00167367f, +0.00411005f, -0.00926903f, +0.01940955f, -0.03850145f, +0.07548820f, -0.16504324f, +0.89693080f, +0.29078630f, -0.10846222f, +0.05375393f, -0.02747154f, +0.01352500f, -0.00623483f, +0.00265213f};
    static const Space FILTER_SRC_SHIFT = -7;

#elif GAUSS_SINC_RESAMPLING_HQ == 1

    static devConstant float32 FILTER0[] = {+0.00207342f, -0.00344470f, +0.00555695f, -0.00872690f, +0.01339070f, -0.02018426f, +0.03014318f, -0.04525655f, +0.07028105f, -0.12084242f, +0.29642233f, +0.89820427f, -0.17433166f, +0.09030320f, -0.05600540f, +0.03686211f, -0.02467708f, +0.01646969f, -0.01083847f, +0.00698587f, -0.00439066f, +0.00268270f, -0.00159000f, +0.00091263f};
    static devConstant float32 FILTER1[] = {-0.00159338f, +0.00268841f, -0.00440000f, +0.00700073f, -0.01086152f, +0.01650472f, -0.02472957f, +0.03694050f, -0.05612451f, +0.09049525f, -0.17470241f, +0.90011448f, +0.29705273f, -0.12109941f, +0.07043052f, -0.04535279f, +0.03020728f, -0.02022719f, +0.01341918f, -0.00874546f, +0.00556877f, -0.00345202f, +0.00207783f, -0.00121212f};
    static const Space FILTER_SRC_SHIFT = -11;

#else

    #error

#endif

//----------------------------------------------------------------

#define FUNCSPACE gaussSincResampling
#define FUNCNAME upsampleTwiceBalanced

#define PACK_SIZE 2
#define PACK_TO_SRC_FACTOR 1

#define HORIZONTAL_FIRST 1

# include "rationalResample/rationalResampleMultiple.inl"
