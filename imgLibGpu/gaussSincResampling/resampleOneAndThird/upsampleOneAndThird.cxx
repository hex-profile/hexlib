#if HOSTCODE
#include "upsampleOneAndThird.h"
#endif

#include "gaussSincResampling/gaussSincResamplingSettings.h"
#include "gaussSincResampling/common/allTypes.h"

//================================================================
//
// upsampleOneAndThirdBalanced
//
//================================================================

#define FUNCSPACE gaussSincResampling

#define PACK_SIZE 4
#define PACK_TO_SRC_FACTOR 3

#define HORIZONTAL_FIRST 0

//----------------------------------------------------------------

#if GAUSS_SINC_RESAMPLING_HQ == 0

    static const Space balancedFilterSrcShift = -6;
    static devConstant float32 balancedFilter0[] = {-0.00304974f, +0.00667832f, -0.01366109f, +0.02679042f, -0.05348373f, +0.13352587f, +0.97444888f, -0.10100833f, +0.04464127f, -0.02267651f, +0.01148360f, -0.00552878f, +0.00247618f, -0.00101957f, +0.00038321f};
    static devConstant float32 balancedFilter1[] = {+0.00387308f, -0.00900910f, +0.01936415f, -0.03907094f, +0.07635356f, -0.15618776f, +0.46015820f, +0.77765646f, -0.19243908f, +0.09045953f, -0.04624995f, +0.02319632f, -0.01098439f, +0.00482143f, -0.00194152f};
    static devConstant float32 balancedFilter2[] = {-0.00194152f, +0.00482143f, -0.01098439f, +0.02319632f, -0.04624995f, +0.09045953f, -0.19243908f, +0.77765646f, +0.46015820f, -0.15618776f, +0.07635356f, -0.03907094f, +0.01936415f, -0.00900910f, +0.00387308f};
    static devConstant float32 balancedFilter3[] = {+0.00038321f, -0.00101957f, +0.00247618f, -0.00552878f, +0.01148360f, -0.02267651f, +0.04464127f, -0.10100833f, +0.97444888f, +0.13352587f, -0.05348373f, +0.02679042f, -0.01366109f, +0.00667832f, -0.00304974f};

#elif GAUSS_SINC_RESAMPLING_HQ == 1

    static const Space balancedFilterSrcShift = -11;
    static devConstant float32 balancedFilter0[] = {+0.00105171f, -0.00175396f, +0.00283954f, -0.00447357f, +0.00688255f, -0.01039344f, +0.01552980f, -0.02327396f, +0.03590326f, -0.06053805f, +0.13705908f, +0.97391324f, -0.10554080f, +0.05235822f, -0.03205478f, +0.02100614f, -0.01405180f, +0.00938865f, -0.00619208f, +0.00400256f, -0.00252405f, +0.00154786f, -0.00092098f, +0.00053079f, -0.00029593f};
    static devConstant float32 balancedFilter1[] = {-0.00169571f, +0.00289519f, -0.00479175f, +0.00770288f, -0.01205918f, +0.01845635f, -0.02777226f, +0.04146221f, -0.06238997f, +0.09762932f, -0.17170038f, +0.46696879f, +0.78218246f, -0.20598535f, +0.11063775f, -0.06939838f, +0.04584895f, -0.03070729f, +0.02046825f, -0.01343903f, +0.00863659f, -0.00540980f, +0.00329320f, -0.00194418f, +0.00111136f};
    static devConstant float32 balancedFilter2[] = {+0.00111136f, -0.00194418f, +0.00329320f, -0.00540980f, +0.00863659f, -0.01343903f, +0.02046825f, -0.03070729f, +0.04584895f, -0.06939838f, +0.11063775f, -0.20598535f, +0.78218246f, +0.46696879f, -0.17170038f, +0.09762932f, -0.06238997f, +0.04146221f, -0.02777226f, +0.01845635f, -0.01205918f, +0.00770288f, -0.00479175f, +0.00289519f, -0.00169571f};
    static devConstant float32 balancedFilter3[] = {-0.00029593f, +0.00053079f, -0.00092098f, +0.00154786f, -0.00252405f, +0.00400256f, -0.00619208f, +0.00938865f, -0.01405180f, +0.02100614f, -0.03205478f, +0.05235822f, -0.10554080f, +0.97391324f, +0.13705908f, -0.06053805f, +0.03590326f, -0.02327396f, +0.01552980f, -0.01039344f, +0.00688255f, -0.00447357f, +0.00283954f, -0.00175396f, +0.00105171f};

#else

    #error

#endif

//----------------------------------------------------------------

#define FILTER0 balancedFilter0
#define FILTER1 balancedFilter1
#define FILTER2 balancedFilter2
#define FILTER3 balancedFilter3
#define FILTER_SRC_SHIFT balancedFilterSrcShift

#define FUNCNAME upsampleOneAndThirdBalanced

# include "rationalResample/rationalResampleMultiple.inl"
