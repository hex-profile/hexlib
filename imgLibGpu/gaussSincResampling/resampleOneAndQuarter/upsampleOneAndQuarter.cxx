#if HOSTCODE
#include "upsampleOneAndQuarter.h"
#endif

#include "gaussSincResampling/gaussSincResamplingSettings.h"
#include "gaussSincResampling/common/allTypes.h"

//================================================================
//
// upsampleOneAndQuarterBalanced
//
//================================================================

#define FUNCSPACE gaussSincResampling

#define PACK_SIZE 5
#define PACK_TO_SRC_FACTOR 4

#define HORIZONTAL_FIRST 0

//----------------------------------------------------------------

#if GAUSS_SINC_RESAMPLING_HQ == 0

    static const Space balancedFilterSrcShift = -6;
    static devConstant float32 balancedFilter0[] = {-0.00241229f, +0.00529252f, -0.01084216f, +0.02127401f, -0.04239411f, +0.10456193f, +0.98382546f, -0.08367051f, +0.03668917f, -0.01861798f, +0.00943610f, -0.00455023f, +0.00204196f, -0.00084263f, +0.00031745f, -0.00010867f};
    static devConstant float32 balancedFilter1[] = {+0.00317089f, -0.00742270f, +0.01604222f, -0.03249610f, +0.06354554f, -0.12887265f, +0.35761652f, +0.85318919f, -0.18014379f, +0.08336349f, -0.04256729f, +0.02141671f, -0.01019346f, +0.00450172f, -0.00182500f, +0.00067471f};
    static devConstant float32 balancedFilter2[] = {-0.00186663f, +0.00468803f, -0.01079123f, +0.02298772f, -0.04609567f, +0.09006436f, -0.18746129f, +0.62847471f, +0.62847471f, -0.18746129f, +0.09006436f, -0.04609567f, +0.02298772f, -0.01079123f, +0.00468803f, -0.00186663f};
    static devConstant float32 balancedFilter3[] = {+0.00067471f, -0.00182500f, +0.00450172f, -0.01019346f, +0.02141671f, -0.04256729f, +0.08336349f, -0.18014379f, +0.85318919f, +0.35761652f, -0.12887265f, +0.06354554f, -0.03249610f, +0.01604222f, -0.00742270f, +0.00317089f};
    static devConstant float32 balancedFilter4[] = {-0.00010867f, +0.00031745f, -0.00084263f, +0.00204196f, -0.00455023f, +0.00943610f, -0.01861798f, +0.03668917f, -0.08367051f, +0.98382546f, +0.10456193f, -0.04239411f, +0.02127401f, -0.01084216f, +0.00529252f, -0.00241229f};

#elif GAUSS_SINC_RESAMPLING_HQ == 1

    static const Space balancedFilterSrcShift = -10;
    static devConstant float32 balancedFilter0[] = {-0.00139992f, +0.00226800f, -0.00357543f, +0.00550374f, -0.00831447f, +0.01242505f, -0.01861510f, +0.02868111f, -0.04818730f, +0.10758809f, +0.98391009f, -0.08732521f, +0.04290602f, -0.02619444f, +0.01714939f, -0.01146970f, +0.00766497f, -0.00505741f, +0.00327097f, -0.00206406f, +0.00126670f, -0.00075428f, +0.00043507f, -0.00024276f, +0.00013091f};
    static devConstant float32 balancedFilter1[] = {+0.00243595f, -0.00404086f, +0.00650948f, -0.01020994f, +0.01565006f, -0.02357334f, +0.03519908f, -0.05289187f, +0.08237901f, -0.14287272f, +0.36403766f, +0.85624384f, -0.19137180f, +0.10065243f, -0.06271872f, +0.04134566f, -0.02768458f, +0.01846788f, -0.01214246f, +0.00781720f, -0.00490653f, +0.00299349f, -0.00177142f, +0.00101510f, -0.00056260f};
    static devConstant float32 balancedFilter2[] = {-0.00196599f, +0.00334333f, -0.00551268f, +0.00883108f, -0.01378306f, +0.02104244f, -0.03161385f, +0.04719397f, -0.07120625f, +0.11239875f, -0.20293352f, +0.63364619f, +0.63364619f, -0.20293352f, +0.11239875f, -0.07120625f, +0.04719397f, -0.03161385f, +0.02104244f, -0.01378306f, +0.00883108f, -0.00551268f, +0.00334333f, -0.00196599f, +0.00111920f};
    static devConstant float32 balancedFilter3[] = {+0.00101597f, -0.00177294f, +0.00299606f, -0.00491076f, +0.00782394f, -0.01215292f, +0.01848378f, -0.02770843f, +0.04138127f, -0.06277275f, +0.10073914f, -0.19153666f, +0.85698146f, +0.36435127f, -0.14299580f, +0.08244997f, -0.05293743f, +0.03522940f, -0.02359364f, +0.01566354f, -0.01021874f, +0.00651509f, -0.00404434f, +0.00243805f, -0.00142455f};
    static devConstant float32 balancedFilter4[] = {-0.00024259f, +0.00043476f, -0.00075374f, +0.00126580f, -0.00206260f, +0.00326865f, -0.00505383f, +0.00765955f, -0.01146159f, +0.01713726f, -0.02617591f, +0.04287567f, -0.08726344f, +0.98321411f, +0.10751199f, -0.04815322f, +0.02866082f, -0.01860193f, +0.01241626f, -0.00830859f, +0.00549985f, -0.00357291f, +0.00226640f, -0.00139893f, +0.00083818f};

#else

    #error

#endif

//----------------------------------------------------------------

#define FILTER0 balancedFilter0
#define FILTER1 balancedFilter1
#define FILTER2 balancedFilter2
#define FILTER3 balancedFilter3
#define FILTER4 balancedFilter4
#define FILTER_SRC_SHIFT balancedFilterSrcShift

#define FUNCNAME upsampleOneAndQuarterBalanced

# include "rationalResample/rationalResampleMultiple.inl"
