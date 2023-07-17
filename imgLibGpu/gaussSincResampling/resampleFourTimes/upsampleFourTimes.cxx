#if HOSTCODE
#include "upsampleFourTimes.h"
#endif

#include "gpuDevice/gpuDevice.h"
#include "gaussSincResampling/gaussSincResamplingSettings.h"
#include "gaussSincResampling/resampleFourTimes/allTypes.h"

//================================================================
//
// Instance
//
//================================================================

#if GAUSS_SINC_RESAMPLING_HQ == 0

    #define FILTER_SRC_SHIFT -7
    static devConstant float32 FILTER0[] = {+0.00387308f, -0.00900910f, +0.01936415f, -0.03907094f, +0.07635356f, -0.15618776f, +0.46015820f, +0.77765646f, -0.19243908f, +0.09045953f, -0.04624995f, +0.02319632f, -0.01098439f, +0.00482143f, -0.00194152f};
    static devConstant float32 FILTER1[] = {+0.00128228f, -0.00304700f, +0.00667231f, -0.01364880f, +0.02676633f, -0.05343562f, +0.13340577f, +0.97357244f, -0.10091749f, +0.04460112f, -0.02265611f, +0.01147327f, -0.00552380f, +0.00247396f, -0.00101865f};
    static devConstant float32 FILTER2[] = {-0.00101865f, +0.00247396f, -0.00552380f, +0.01147327f, -0.02265611f, +0.04460112f, -0.10091749f, +0.97357244f, +0.13340577f, -0.05343562f, +0.02676633f, -0.01364880f, +0.00667231f, -0.00304700f, +0.00128228f};
    static devConstant float32 FILTER3[] = {-0.00194152f, +0.00482143f, -0.01098439f, +0.02319632f, -0.04624995f, +0.09045953f, -0.19243908f, +0.77765646f, +0.46015820f, -0.15618776f, +0.07635356f, -0.03907094f, +0.01936415f, -0.00900910f, +0.00387308f};

#elif GAUSS_SINC_RESAMPLING_HQ == 1

    #define FILTER_SRC_SHIFT -12
    static devConstant float32 FILTER0[] = {-0.00169571f, +0.00289519f, -0.00479175f, +0.00770288f, -0.01205918f, +0.01845635f, -0.02777226f, +0.04146221f, -0.06238997f, +0.09762932f, -0.17170038f, +0.46696879f, +0.78218246f, -0.20598535f, +0.11063775f, -0.06939838f, +0.04584895f, -0.03070729f, +0.02046825f, -0.01343903f, +0.00863659f, -0.00540980f, +0.00329320f, -0.00194418f, +0.00111136f};
    static devConstant float32 FILTER1[] = {-0.00061126f, +0.00105204f, -0.00175451f, +0.00284044f, -0.00447498f, +0.00688472f, -0.01039671f, +0.01553469f, -0.02328129f, +0.03591457f, -0.06055713f, +0.13710229f, +0.97422025f, -0.10557407f, +0.05237472f, -0.03206489f, +0.02101276f, -0.01405622f, +0.00939161f, -0.00619403f, +0.00400382f, -0.00252484f, +0.00154835f, -0.00092127f, +0.00053096f};
    static devConstant float32 FILTER2[] = {+0.00053096f, -0.00092127f, +0.00154835f, -0.00252484f, +0.00400382f, -0.00619403f, +0.00939161f, -0.01405622f, +0.02101276f, -0.03206489f, +0.05237472f, -0.10557407f, +0.97422025f, +0.13710229f, -0.06055713f, +0.03591457f, -0.02328129f, +0.01553469f, -0.01039671f, +0.00688472f, -0.00447498f, +0.00284044f, -0.00175451f, +0.00105204f, -0.00061126f};
    static devConstant float32 FILTER3[] = {+0.00111136f, -0.00194418f, +0.00329320f, -0.00540980f, +0.00863659f, -0.01343903f, +0.02046825f, -0.03070729f, +0.04584895f, -0.06939838f, +0.11063775f, -0.20598535f, +0.78218246f, +0.46696879f, -0.17170038f, +0.09762932f, -0.06238997f, +0.04146221f, -0.02777226f, +0.01845635f, -0.01205918f, +0.00770288f, -0.00479175f, +0.00289519f, -0.00169571f};

#else

    #error

#endif

//----------------------------------------------------------------

#define FUNCSPACE gaussSincResampling
#define FUNCNAME upsampleFourTimesBalanced

#define PACK_SIZE 4
#define PACK_TO_SRC_FACTOR 1

#define HORIZONTAL_FIRST 0

# include "rationalResample/rationalResampleMultiple.inl"

////

#undef FUNCNAME
#define FUNCNAME upsampleFourTimesBalancedDual

#undef TASK_COUNT
#define TASK_COUNT 2

# include "rationalResample/rationalResampleMultiple.inl"
