#pragma once

#include "numbers/float/floatBase.h"
#include "gpuDevice/gpuDevice.h"

//================================================================
//
// genCosShape
//
//================================================================

void genCosShape(float32* arrPtr, int arrSize);
void genCosShapeHalf(float32* arrPtr, int arrSize);

//================================================================
//
// cosShape*
//
//================================================================

static devConstant float32 cosShape1[1] = {1.00000000f};
static devConstant float32 cosShape2[2] = {0.50000000f, 0.50000000f};
static devConstant float32 cosShape3[3] = {0.16666667f, 0.66666667f, 0.16666667f};
static devConstant float32 cosShape4[4] = {0.07322330f, 0.42677670f, 0.42677670f, 0.07322330f};
static devConstant float32 cosShape5[5] = {0.03819660f, 0.26180340f, 0.40000000f, 0.26180340f, 0.03819660f};
static devConstant float32 cosShape6[6] = {0.02232910f, 0.16666667f, 0.31100423f, 0.31100423f, 0.16666667f, 0.02232910f};
static devConstant float32 cosShape7[7] = {0.01414730f, 0.11106844f, 0.23192711f, 0.28571429f, 0.23192711f, 0.11106844f, 0.01414730f};
static devConstant float32 cosShape8[8] = {0.00951506f, 0.07716457f, 0.17283543f, 0.24048494f, 0.24048494f, 0.17283543f, 0.07716457f, 0.00951506f};
static devConstant float32 cosShape9[9] = {0.00670082f, 0.05555556f, 0.13040535f, 0.19622716f, 0.22222222f, 0.19622716f, 0.13040535f, 0.05555556f, 0.00670082f};
static devConstant float32 cosShape10[10] = {0.00489435f, 0.04122147f, 0.10000000f, 0.15877853f, 0.19510565f, 0.19510565f, 0.15877853f, 0.10000000f, 0.04122147f, 0.00489435f};
static devConstant float32 cosShape11[11] = {0.00368246f, 0.03137630f, 0.07797138f, 0.12867409f, 0.16738668f, 0.18181818f, 0.16738668f, 0.12867409f, 0.07797138f, 0.03137630f, 0.00368246f};
static devConstant float32 cosShape12[12] = {0.00283951f, 0.02440777f, 0.06176508f, 0.10490159f, 0.14225890f, 0.16382715f, 0.16382715f, 0.14225890f, 0.10490159f, 0.06176508f, 0.02440777f, 0.00283951f};
static devConstant float32 cosShape13[13] = {0.00223524f, 0.01934533f, 0.04964578f, 0.08619513f, 0.12062037f, 0.14503508f, 0.15384615f, 0.14503508f, 0.12062037f, 0.08619513f, 0.04964578f, 0.01934533f, 0.00223524f};
static devConstant float32 cosShape14[14] = {0.00179086f, 0.01558347f, 0.04043688f, 0.07142857f, 0.10242027f, 0.12727368f, 0.14106628f, 0.14106628f, 0.12727368f, 0.10242027f, 0.07142857f, 0.04043688f, 0.01558347f, 0.00179086f};
static devConstant float32 cosShape15[15] = {0.00145683f, 0.01273220f, 0.03333333f, 0.05969810f, 0.08726780f, 0.11127537f, 0.12756970f, 0.13333333f, 0.12756970f, 0.11127537f, 0.08726780f, 0.05969810f, 0.03333333f, 0.01273220f, 0.00145683f};
static devConstant float32 cosShape16[16] = {0.00120092f, 0.01053315f, 0.02777686f, 0.05030685f, 0.07469315f, 0.09722314f, 0.11446685f, 0.12379908f, 0.12379908f, 0.11446685f, 0.09722314f, 0.07469315f, 0.05030685f, 0.02777686f, 0.01053315f, 0.00120092f};
static devConstant float32 cosShape17[17] = {0.00100158f, 0.00881076f, 0.02337443f, 0.04272571f, 0.06425108f, 0.08504343f, 0.10229464f, 0.11367484f, 0.11764706f, 0.11367484f, 0.10229464f, 0.08504343f, 0.06425108f, 0.04272571f, 0.02337443f, 0.00881076f, 0.00100158f};
static devConstant float32 cosShape18[18] = {0.00084401f, 0.00744303f, 0.01984513f, 0.03655444f, 0.05555556f, 0.07455667f, 0.09126598f, 0.10366808f, 0.11026710f, 0.11026710f, 0.10366808f, 0.09126598f, 0.07455667f, 0.05555556f, 0.03655444f, 0.01984513f, 0.00744303f, 0.00084401f};
static devConstant float32 cosShape19[19] = {0.00071783f, 0.00634349f, 0.01698518f, 0.03148971f, 0.04828530f, 0.06555187f, 0.08141832f, 0.09416529f, 0.10241143f, 0.10526316f, 0.10241143f, 0.09416529f, 0.08141832f, 0.06555187f, 0.04828530f, 0.03148971f, 0.01698518f, 0.00634349f, 0.00071783f};
static devConstant float32 cosShape20[20] = {0.00061558f, 0.00544967f, 0.01464466f, 0.02730048f, 0.04217828f, 0.05782172f, 0.07269952f, 0.08535534f, 0.09455033f, 0.09938442f, 0.09938442f, 0.09455033f, 0.08535534f, 0.07269952f, 0.05782172f, 0.04217828f, 0.02730048f, 0.01464466f, 0.00544967f, 0.00061558f};
static devConstant float32 cosShape21[21] = {0.00053187f, 0.00471577f, 0.01271182f, 0.02380952f, 0.03702281f, 0.05117762f, 0.06501624f, 0.07730904f, 0.08696375f, 0.09312251f, 0.09523810f, 0.09312251f, 0.08696375f, 0.07730904f, 0.06501624f, 0.05117762f, 0.03702281f, 0.02380952f, 0.01271182f, 0.00471577f, 0.00053187f};
static devConstant float32 cosShape22[22] = {0.00046266f, 0.00410764f, 0.01110229f, 0.02087996f, 0.03264852f, 0.04545455f, 0.05826057f, 0.07002913f, 0.07980680f, 0.08680145f, 0.09044643f, 0.09044643f, 0.08680145f, 0.07980680f, 0.07002913f, 0.05826057f, 0.04545455f, 0.03264852f, 0.02087996f, 0.01110229f, 0.00410764f, 0.00046266f};
static devConstant float32 cosShape23[23] = {0.00040496f, 0.00359951f, 0.00975168f, 0.01840520f, 0.02891828f, 0.04051120f, 0.05232417f, 0.06348109f, 0.07315448f, 0.08062693f, 0.08534423f, 0.08695652f, 0.08534423f, 0.08062693f, 0.07315448f, 0.06348109f, 0.05232417f, 0.04051120f, 0.02891828f, 0.01840520f, 0.00975168f, 0.00359951f, 0.00040496f};
static devConstant float32 cosShape24[24] = {0.00035646f, 0.00317169f, 0.00861028f, 0.01630161f, 0.02572152f, 0.03622808f, 0.04710526f, 0.05761181f, 0.06703173f, 0.07472306f, 0.08016165f, 0.08297687f, 0.08297687f, 0.08016165f, 0.07472306f, 0.06703173f, 0.05761181f, 0.04710526f, 0.03622808f, 0.02572152f, 0.01630161f, 0.00861028f, 0.00317169f, 0.00035646f};
static devConstant float32 cosShape25[25] = {0.00031541f, 0.00280894f, 0.00763932f, 0.01450304f, 0.02296883f, 0.03250475f, 0.04251162f, 0.05236068f, 0.06143307f, 0.06915875f, 0.07505227f, 0.07874333f, 0.08000000f, 0.07874333f, 0.07505227f, 0.06915875f, 0.06143307f, 0.05236068f, 0.04251162f, 0.03250475f, 0.02296883f, 0.01450304f, 0.00763932f, 0.00280894f, 0.00031541f};
static devConstant float32 cosShape26[26] = {0.00028043f, 0.00249938f, 0.00680831f, 0.01295682f, 0.02058757f, 0.02925709f, 0.03846154f, 0.04766599f, 0.05633551f, 0.06396626f, 0.07011476f, 0.07442370f, 0.07664265f, 0.07664265f, 0.07442370f, 0.07011476f, 0.06396626f, 0.05633551f, 0.04766599f, 0.03846154f, 0.02925709f, 0.02058757f, 0.01295682f, 0.00680831f, 0.00249938f, 0.00028043f};
static devConstant float32 cosShape27[27] = {0.00025043f, 0.00223361f, 0.00609304f, 0.01162068f, 0.01851852f, 0.02641470f, 0.03488352f, 0.04346845f, 0.05170666f, 0.05915402f, 0.06540905f, 0.07013454f, 0.07307574f, 0.07407407f, 0.07307574f, 0.07013454f, 0.06540905f, 0.05915402f, 0.05170666f, 0.04346845f, 0.03488352f, 0.02641470f, 0.01851852f, 0.01162068f, 0.00609304f, 0.00223361f, 0.00025043f};
static devConstant float32 cosShape28[28] = {0.00022456f, 0.00200417f, 0.00547414f, 0.01046047f, 0.01671314f, 0.02391860f, 0.03171555f, 0.03971302f, 0.04750997f, 0.05471543f, 0.06096810f, 0.06595444f, 0.06942440f, 0.07120401f, 0.07120401f, 0.06942440f, 0.06595444f, 0.06096810f, 0.05471543f, 0.04750997f, 0.03971302f, 0.03171555f, 0.02391860f, 0.01671314f, 0.01046047f, 0.00547414f, 0.00200417f, 0.00022456f};
static devConstant float32 cosShape29[29] = {0.00020214f, 0.00180506f, 0.00493596f, 0.00944843f, 0.01513148f, 0.02171937f, 0.02890407f, 0.03634962f, 0.04370787f, 0.05063477f, 0.05680642f, 0.06193424f, 0.06577846f, 0.06815933f, 0.06896552f, 0.06815933f, 0.06577846f, 0.06193424f, 0.05680642f, 0.05063477f, 0.04370787f, 0.03634962f, 0.02890407f, 0.02171937f, 0.01513148f, 0.00944843f, 0.00493596f, 0.00180506f, 0.00020214f};
static devConstant float32 cosShape30[30] = {0.00018260f, 0.00163145f, 0.00446582f, 0.00856184f, 0.01374049f, 0.01977545f, 0.02640294f, 0.03333333f, 0.04026372f, 0.04689122f, 0.05292618f, 0.05810483f, 0.06220085f, 0.06503522f, 0.06648406f, 0.06648406f, 0.06503522f, 0.06220085f, 0.05810483f, 0.05292618f, 0.04689122f, 0.04026372f, 0.03333333f, 0.02640294f, 0.01977545f, 0.01374049f, 0.00856184f, 0.00446582f, 0.00163145f, 0.00018260f};
static devConstant float32 cosShape31[31] = {0.00016551f, 0.00147938f, 0.00405333f, 0.00778200f, 0.01251271f, 0.01805180f, 0.02417250f, 0.03062422f, 0.03714283f, 0.04346146f, 0.04932142f, 0.05448280f, 0.05873430f, 0.06190186f, 0.06385580f, 0.06451613f, 0.06385580f, 0.06190186f, 0.05873430f, 0.05448280f, 0.04932142f, 0.04346146f, 0.03714283f, 0.03062422f, 0.02417250f, 0.01805180f, 0.01251271f, 0.00778200f, 0.00405333f, 0.00147938f, 0.00016551f};
static devConstant float32 cosShape32[32] = {0.00015048f, 0.00134561f, 0.00368996f, 0.00709342f, 0.01142521f, 0.01651885f, 0.02217860f, 0.02818696f, 0.03431304f, 0.04032140f, 0.04598115f, 0.05107479f, 0.05540658f, 0.05881004f, 0.06115439f, 0.06234952f, 0.06234952f, 0.06115439f, 0.05881004f, 0.05540658f, 0.05107479f, 0.04598115f, 0.04032140f, 0.03431304f, 0.02818696f, 0.02217860f, 0.01651885f, 0.01142521f, 0.00709342f, 0.00368996f, 0.00134561f, 0.00015048f};
static devConstant float32 cosShape33[33] = {0.00013721f, 0.00122749f, 0.00336862f, 0.00648324f, 0.01045877f, 0.01515152f, 0.02039188f, 0.02599046f, 0.03174491f, 0.03744724f, 0.04289136f, 0.04788051f, 0.05223436f, 0.05579556f, 0.05843539f, 0.06005845f, 0.06060606f, 0.06005845f, 0.05843539f, 0.05579556f, 0.05223436f, 0.04788051f, 0.04289136f, 0.03744724f, 0.03174491f, 0.02599046f, 0.02039188f, 0.01515152f, 0.01045877f, 0.00648324f, 0.00336862f, 0.00122749f, 0.00013721f};
