#if HOSTCODE
#include "downsampleTwicePacketized2.h"
#endif

#include "gaussSincResampling/gaussSincResamplingSettings.h"
#include "gaussSincResampling/common/allTypes.h"

//================================================================
//
// Downsample 2X (packet 2X)
//
//================================================================

#if GAUSS_SINC_RESAMPLING_HQ == 0

    static devConstant float32 FILTER0[] = {+0.00109807f, +0.00182181f, -0.00034342f, -0.00378384f, -0.00299301f, +0.00400249f, +0.00905726f, +0.00147555f, -0.01412048f, -0.01546785f, +0.00947198f, +0.03457897f, +0.01599327f, -0.04479528f, -0.07276104f, +0.01456780f, +0.20211261f, +0.36062647f, +0.36062647f, +0.20211261f, +0.01456780f, -0.07276104f, -0.04479528f, +0.01599327f, +0.03457897f, +0.00947198f, -0.01546785f, -0.01412048f, +0.00147555f, +0.00905726f, +0.00400249f, -0.00299301f, -0.00378384f, -0.00034342f, +0.00182181f, +0.00109807f, -0.00036606f, -0.00071667f};
    static devConstant float32 FILTER1[] = {-0.00071667f, -0.00036606f, +0.00109807f, +0.00182181f, -0.00034342f, -0.00378384f, -0.00299301f, +0.00400249f, +0.00905726f, +0.00147555f, -0.01412048f, -0.01546785f, +0.00947198f, +0.03457897f, +0.01599327f, -0.04479528f, -0.07276104f, +0.01456780f, +0.20211261f, +0.36062647f, +0.36062647f, +0.20211261f, +0.01456780f, -0.07276104f, -0.04479528f, +0.01599327f, +0.03457897f, +0.00947198f, -0.01546785f, -0.01412048f, +0.00147555f, +0.00905726f, +0.00400249f, -0.00299301f, -0.00378384f, -0.00034342f, +0.00182181f, +0.00109807f};
    static const Space FILTER_SRC_SHIFT = -17;

#elif GAUSS_SINC_RESAMPLING_HQ == 1

    static devConstant float32 FILTER0[] = {-0.00083082f, -0.00013692f, +0.00120640f, +0.00098423f, -0.00121221f, -0.00225422f, +0.00038831f, +0.00357901f, +0.00166868f, -0.00416069f, -0.00498408f, +0.00286935f, +0.00884340f, +0.00138118f, -0.01157248f, -0.00901341f, +0.01064860f, +0.01910488f, -0.00319780f, -0.02889819f, -0.01336371f, +0.03355934f, +0.04134849f, -0.02523844f, -0.08643698f, -0.01626689f, +0.19458208f, +0.38732168f, +0.38732168f, +0.19458208f, -0.01626689f, -0.08643698f, -0.02523844f, +0.04134849f, +0.03355934f, -0.01336371f, -0.02889819f, -0.00319780f, +0.01910488f, +0.01064860f, -0.00901341f, -0.01157248f, +0.00138118f, +0.00884340f, +0.00286935f, -0.00498408f, -0.00416069f, +0.00166868f, +0.00357901f, +0.00038831f, -0.00225422f, -0.00121221f, +0.00098423f, +0.00120640f, -0.00013692f, -0.00083082f, -0.00025475f, +0.00041717f};
    static devConstant float32 FILTER1[] = {+0.00041717f, -0.00025475f, -0.00083082f, -0.00013692f, +0.00120640f, +0.00098423f, -0.00121221f, -0.00225422f, +0.00038831f, +0.00357901f, +0.00166868f, -0.00416069f, -0.00498408f, +0.00286935f, +0.00884340f, +0.00138118f, -0.01157248f, -0.00901341f, +0.01064860f, +0.01910488f, -0.00319780f, -0.02889819f, -0.01336371f, +0.03355934f, +0.04134849f, -0.02523844f, -0.08643698f, -0.01626689f, +0.19458208f, +0.38732168f, +0.38732168f, +0.19458208f, -0.01626689f, -0.08643698f, -0.02523844f, +0.04134849f, +0.03355934f, -0.01336371f, -0.02889819f, -0.00319780f, +0.01910488f, +0.01064860f, -0.00901341f, -0.01157248f, +0.00138118f, +0.00884340f, +0.00286935f, -0.00498408f, -0.00416069f, +0.00166868f, +0.00357901f, +0.00038831f, -0.00225422f, -0.00121221f, +0.00098423f, +0.00120640f, -0.00013692f, -0.00083082f};
    static const Space FILTER_SRC_SHIFT = -27;

#else

    #error

#endif

//----------------------------------------------------------------

#define FUNCSPACE gaussSincResampling
#define FUNCNAME downsampleTwicePacketized2

#define PACK_SIZE 2
#define PACK_TO_SRC_FACTOR 4

#define HORIZONTAL_FIRST 1

#define TASK_COUNT 1

# include "rationalResample/rationalResampleMultiple.inl"
