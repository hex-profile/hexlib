#if HOSTCODE
#include "upsampleThreeTimes.h"
#endif

#include "gpuDevice/gpuDevice.h"
#include "gaussSincResampling/gaussSincResamplingSettings.h"
#include "gaussSincResampling/resampleThreeTimes/allTypes.h"

//================================================================
//
// Instance
//
//================================================================

#if GAUSS_SINC_RESAMPLING_HQ == 0

    #define FILTER_SRC_SHIFT -7
    static devConstant float32 FILTER0[] = {+0.00349888f, -0.00816738f, +0.01760843f, -0.03560560f, +0.06960307f, -0.14167745f, +0.40321639f, +0.82150585f, -0.18721395f, +0.08726353f, -0.04458702f, +0.02240205f, -0.01063847f, +0.00468552f, -0.00189386f};
    static devConstant float32 FILTER1[] = {+0.00000000f, -0.00000000f, +0.00000000f, -0.00000000f, +0.00000000f, -0.00000000f, +0.00000000f, +1.00000000f, +0.00000000f, -0.00000000f, +0.00000000f, -0.00000000f, +0.00000000f, -0.00000000f, +0.00000000f}; 
    static devConstant float32 FILTER2[] = {-0.00189386f, +0.00468552f, -0.01063847f, +0.02240205f, -0.04458702f, +0.08726353f, -0.18721395f, +0.82150585f, +0.40321639f, -0.14167745f, +0.06960307f, -0.03560560f, +0.01760843f, -0.00816738f, +0.00349888f};

#elif GAUSS_SINC_RESAMPLING_HQ == 1

    #error Untested

    #define FILTER_SRC_SHIFT -12
    static devConstant float32 FILTER0[] = {-0.00155334f, +0.00265565f, -0.00440084f, +0.00708274f, -0.01109983f, +0.01700246f, -0.02559876f, +0.03822029f, -0.05746610f, +0.08968608f, -0.15648902f, +0.40991230f, +0.82530836f, -0.19956288f, +0.10597257f, -0.06623255f, +0.04370552f, -0.02926820f, +0.01951763f, -0.01282481f, +0.00825001f, -0.00517352f, +0.00315326f, -0.00186401f, +0.00106699f};
    static devConstant float32 FILTER1[] = {-0.00000000f, +0.00000000f, -0.00000000f, +0.00000000f, -0.00000000f, +0.00000000f, -0.00000000f, +0.00000000f, -0.00000000f, +0.00000000f, -0.00000000f, +0.00000000f, +1.00000000f, +0.00000000f, -0.00000000f, +0.00000000f, -0.00000000f, +0.00000000f, -0.00000000f, +0.00000000f, -0.00000000f, +0.00000000f, -0.00000000f, +0.00000000f, -0.00000000f}; 
    static devConstant float32 FILTER2[] = {+0.00106699f, -0.00186401f, +0.00315326f, -0.00517352f, +0.00825001f, -0.01282481f, +0.01951763f, -0.02926820f, +0.04370552f, -0.06623255f, +0.10597257f, -0.19956288f, +0.82530836f, +0.40991230f, -0.15648902f, +0.08968608f, -0.05746610f, +0.03822029f, -0.02559876f, +0.01700246f, -0.01109983f, +0.00708274f, -0.00440084f, +0.00265565f, -0.00155334f};

#else

    #error

#endif

//----------------------------------------------------------------

#define FUNCSPACE gaussSincResampling
#define FUNCNAME upsampleThreeTimesBalanced

#define PACK_SIZE 3
#define PACK_TO_SRC_FACTOR 1

#define HORIZONTAL_FIRST 0

# include "rationalResample/rationalResampleMultiple.inl"
