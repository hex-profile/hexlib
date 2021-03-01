#if HOSTCODE
#include "upsampleOneAndHalf.h"
#endif

#include "gaussSincResampling/gaussSincResamplingSettings.h"
#include "gaussSincResampling/common/allTypes.h"

//================================================================
//
// upsampleOneAndHalfBalanced
//
//================================================================

#define FUNCSPACE gaussSincResampling

#define PACK_SIZE 3
#define PACK_TO_SRC_FACTOR 2

#define HORIZONTAL_FIRST 0

//----------------------------------------------------------------

#if GAUSS_SINC_RESAMPLING_HQ == 0

    static const Space balancedFilterSrcShift = -7;
    static devConstant float32 balancedFilter0[] = {+0.00173900f, -0.00411747f, +0.00898795f, -0.01834153f, +0.03593885f, -0.07198057f, +0.18364449f, +0.95286833f, -0.12640519f, +0.05655808f, -0.02877429f, +0.01455068f, -0.00698670f, +0.00311880f, -0.00127944f, +0.00047900f};
    static devConstant float32 balancedFilter1[] = {-0.00186663f, +0.00468803f, -0.01079123f, +0.02298772f, -0.04609567f, +0.09006436f, -0.18746129f, +0.62847471f, +0.62847471f, -0.18746129f, +0.09006436f, -0.04609567f, +0.02298772f, -0.01079123f, +0.00468803f, -0.00186663f};
    static devConstant float32 balancedFilter2[] = {+0.00047900f, -0.00127944f, +0.00311880f, -0.00698670f, +0.01455068f, -0.02877429f, +0.05655808f, -0.12640519f, +0.95286833f, +0.18364449f, -0.07198057f, +0.03593885f, -0.01834153f, +0.00898795f, -0.00411747f, +0.00173900f};

#elif GAUSS_SINC_RESAMPLING_HQ == 1

    static const Space balancedFilterSrcShift = -11;
    static devConstant float32 balancedFilter0[] = {+0.00140444f, -0.00233923f, +0.00378257f, -0.00595291f, +0.00915032f, -0.01380936f, +0.02062970f, -0.03093434f, +0.04782178f, -0.08113647f, +0.18827869f, +0.95402942f, -0.13270355f, +0.06684736f, -0.04111017f, +0.02698152f, -0.01805412f, +0.01205858f, -0.00794726f, +0.00513223f, -0.00323284f, +0.00198011f, -0.00117665f, +0.00067722f, -0.00037703f};
    static devConstant float32 balancedFilter1[] = {-0.00196599f, +0.00334333f, -0.00551268f, +0.00883108f, -0.01378306f, +0.02104244f, -0.03161385f, +0.04719397f, -0.07120625f, +0.11239875f, -0.20293352f, +0.63364619f, +0.63364619f, -0.20293352f, +0.11239875f, -0.07120625f, +0.04719397f, -0.03161385f, +0.02104244f, -0.01378306f, +0.00883108f, -0.00551268f, +0.00334333f, -0.00196599f, +0.00111920f};
    static devConstant float32 balancedFilter2[] = {+0.00067751f, -0.00117716f, +0.00198098f, -0.00323426f, +0.00513449f, -0.00795076f, +0.01206389f, -0.01806207f, +0.02699339f, -0.04112827f, +0.06687679f, -0.13276197f, +0.95444944f, +0.18836158f, -0.08117219f, +0.04784283f, -0.03094796f, +0.02063879f, -0.01381544f, +0.00915435f, -0.00595553f, +0.00378423f, -0.00234026f, +0.00140506f, -0.00081747f};

#else

    #error

#endif

//----------------------------------------------------------------

#define FILTER0 balancedFilter0
#define FILTER1 balancedFilter1
#define FILTER2 balancedFilter2
#define FILTER_SRC_SHIFT balancedFilterSrcShift

#define FUNCNAME upsampleOneAndHalfBalanced

# include "rationalResample/rationalResampleMultiple.inl"
