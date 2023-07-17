//================================================================
//
// Generated by S:/hexbase/hexlib/moduleLibGpu/tests/fourierFilterBank/fourierFilterBank.cxx
//
// Pyramid filter compensation is OFF.
// Direct Gabor computation is ON.
//
//================================================================

constexpr Space PREP_PASTE(GABOR_BANK, OrientCount) = 7;
constexpr Space PREP_PASTE(GABOR_BANK, SizeX) = 22;
constexpr Space PREP_PASTE(GABOR_BANK, SizeY) = 22;

constexpr float32 PREP_PASTE(GABOR_BANK, Freq) = +0.247729599f;
constexpr float32 PREP_PASTE(GABOR_BANK, Sigma) = +0.060737871f;

static devConstant float32 PREP_PASTE(GABOR_BANK, ShapeX)[22] = { +0.000049645f, +0.000212998f, +0.000790004f, +0.002532984f, +0.007020775f, +0.016822383f, +0.034844905f, +0.062393721f, +0.096581094f, +0.129238784f, +0.149500698f, +0.149500698f, +0.129238784f, +0.096581094f, +0.062393721f, +0.034844905f, +0.016822383f, +0.007020775f, +0.002532984f, +0.000790004f, +0.000212998f, +0.000049645f, };
static devConstant float32 PREP_PASTE(GABOR_BANK, ShapeY)[22] = { +0.000049645f, +0.000212998f, +0.000790004f, +0.002532984f, +0.007020776f, +0.016822383f, +0.034844905f, +0.062393717f, +0.096581087f, +0.129238784f, +0.149500698f, +0.149500698f, +0.129238784f, +0.096581087f, +0.062393717f, +0.034844905f, +0.016822383f, +0.007020776f, +0.002532984f, +0.000790004f, +0.000212998f, +0.000049645f, };

static devConstant float32_x2 PREP_PASTE(GABOR_BANK, Freq0) = {+0.247729599f, +0.000000000f};
static devConstant float32_x2 PREP_PASTE(GABOR_BANK, DataX0)[22] = { {-0.000039949f, -0.000029473f}, {-0.000128883f, +0.000169580f}, {+0.000622084f, +0.000486947f}, {+0.001589586f, -0.001972112f}, {-0.005402777f, -0.004483446f}, {-0.010926296f, +0.012790957f}, {+0.026168918f, +0.023007719f}, {+0.041862167f, -0.046265919f}, {-0.070684768f, -0.065814666f}, {-0.089409374f, +0.093320027f}, {+0.106464274f, +0.104956262f}, {+0.106464274f, -0.104956262f}, {-0.089409374f, -0.093320027f}, {-0.070684768f, +0.065814666f}, {+0.041862167f, +0.046265919f}, {+0.026168918f, -0.023007719f}, {-0.010926296f, -0.012790957f}, {-0.005402777f, +0.004483446f}, {+0.001589586f, +0.001972112f}, {+0.000622084f, -0.000486947f}, {-0.000128883f, -0.000169580f}, {-0.000039949f, +0.000029473f}, };
static devConstant float32_x2 PREP_PASTE(GABOR_BANK, DataY0)[22] = { {+0.000049645f, +0.000000000f}, {+0.000212998f, +0.000000000f}, {+0.000790004f, +0.000000000f}, {+0.002532984f, +0.000000000f}, {+0.007020776f, +0.000000000f}, {+0.016822383f, +0.000000000f}, {+0.034844905f, +0.000000000f}, {+0.062393717f, +0.000000000f}, {+0.096581087f, +0.000000000f}, {+0.129238784f, +0.000000000f}, {+0.149500698f, +0.000000000f}, {+0.149500698f, -0.000000000f}, {+0.129238784f, -0.000000000f}, {+0.096581087f, -0.000000000f}, {+0.062393717f, -0.000000000f}, {+0.034844905f, -0.000000000f}, {+0.016822383f, -0.000000000f}, {+0.007020776f, -0.000000000f}, {+0.002532984f, -0.000000000f}, {+0.000790004f, -0.000000000f}, {+0.000212998f, -0.000000000f}, {+0.000049645f, -0.000000000f}, };

static devConstant float32_x2 PREP_PASTE(GABOR_BANK, Freq1) = {+0.223196656f, +0.107485846f};
static devConstant float32_x2 PREP_PASTE(GABOR_BANK, DataX1)[22] = { {-0.000027533f, +0.000041310f}, {+0.000154931f, +0.000146166f}, {+0.000630774f, -0.000475637f}, {-0.001164463f, -0.002249452f}, {-0.006687684f, +0.002136860f}, {+0.002361752f, +0.016655771f}, {+0.034831680f, +0.000959919f}, {+0.012148692f, -0.061199553f}, {-0.090240315f, -0.034417897f}, {-0.065644555f, +0.111325897f}, {+0.114229463f, +0.096447341f}, {+0.114229463f, -0.096447341f}, {-0.065644555f, -0.111325897f}, {-0.090240315f, +0.034417897f}, {+0.012148692f, +0.061199553f}, {+0.034831680f, -0.000959919f}, {+0.002361752f, -0.016655771f}, {-0.006687684f, -0.002136860f}, {-0.001164463f, +0.002249452f}, {+0.000630774f, +0.000475637f}, {+0.000154931f, -0.000146166f}, {-0.000027533f, -0.000041310f}, };
static devConstant float32_x2 PREP_PASTE(GABOR_BANK, DataY1)[22] = { {+0.000034301f, +0.000035889f}, {+0.000211126f, +0.000028176f}, {+0.000676502f, -0.000407984f}, {+0.000875124f, -0.002377008f}, {-0.002225763f, -0.006658624f}, {-0.014136835f, -0.009118249f}, {-0.034662012f, +0.003565422f}, {-0.044450548f, +0.043784987f}, {-0.011330563f, +0.095914155f}, {+0.068405077f, +0.109651305f}, {+0.141057923f, +0.049528975f}, {+0.141057923f, -0.049528975f}, {+0.068405077f, -0.109651305f}, {-0.011330563f, -0.095914155f}, {-0.044450548f, -0.043784987f}, {-0.034662012f, -0.003565422f}, {-0.014136835f, +0.009118249f}, {-0.002225763f, +0.006658624f}, {+0.000875124f, +0.002377008f}, {+0.000676502f, +0.000407984f}, {+0.000211126f, -0.000028176f}, {+0.000034301f, -0.000035889f}, };

static devConstant float32_x2 PREP_PASTE(GABOR_BANK, Freq2) = {+0.154456869f, +0.193682805f};
static devConstant float32_x2 PREP_PASTE(GABOR_BANK, DataX2)[22] = { {-0.000035803f, -0.000034391f}, {-0.000208529f, +0.000043403f}, {-0.000304079f, +0.000729138f}, {+0.001378317f, +0.002125147f}, {+0.007018592f, +0.000175098f}, {+0.009846237f, -0.013639802f}, {-0.011791772f, -0.032789044f}, {-0.060374670f, -0.015744055f}, {-0.072903030f, +0.063348666f}, {+0.014839299f, +0.128384024f}, {+0.132242754f, +0.069730289f}, {+0.132242754f, -0.069730289f}, {+0.014839299f, -0.128384024f}, {-0.072903030f, -0.063348666f}, {-0.060374670f, +0.015744055f}, {-0.011791772f, +0.032789044f}, {+0.009846237f, +0.013639802f}, {+0.007018592f, -0.000175098f}, {+0.001378317f, -0.002125147f}, {-0.000304079f, -0.000729138f}, {-0.000208529f, -0.000043403f}, {-0.000035803f, +0.000034391f}, };
static devConstant float32_x2 PREP_PASTE(GABOR_BANK, DataY2)[22] = { {+0.000048538f, +0.000010424f}, {+0.000114115f, -0.000179850f}, {-0.000479069f, -0.000628171f}, {-0.002421574f, +0.000742959f}, {-0.000394082f, +0.007009707f}, {+0.015428077f, +0.006705743f}, {+0.024102803f, -0.025163908f}, {-0.027312119f, -0.056098346f}, {-0.096105978f, +0.009568040f}, {-0.032552514f, +0.125071973f}, {+0.122668631f, +0.085456803f}, {+0.122668631f, -0.085456803f}, {-0.032552514f, -0.125071973f}, {-0.096105978f, -0.009568040f}, {-0.027312119f, +0.056098346f}, {+0.024102803f, +0.025163908f}, {+0.015428077f, -0.006705743f}, {-0.000394082f, -0.007009707f}, {-0.002421574f, -0.000742959f}, {-0.000479069f, +0.000628171f}, {+0.000114115f, +0.000179850f}, {+0.000048538f, -0.000010424f}, };

static devConstant float32_x2 PREP_PASTE(GABOR_BANK, Freq3) = {+0.055125002f, +0.241518497f};
static devConstant float32_x2 PREP_PASTE(GABOR_BANK, DataX3)[22] = { {-0.000043681f, -0.000023591f}, {-0.000210643f, -0.000031584f}, {-0.000774642f, +0.000155035f}, {-0.002167483f, +0.001310735f}, {-0.004417604f, +0.005456745f}, {-0.005517764f, +0.015891721f}, {+0.000424178f, +0.034842324f}, {+0.021894081f, +0.058426239f}, {+0.062580064f, +0.073563859f}, {+0.112185359f, +0.064164691f}, {+0.147264421f, +0.025761355f}, {+0.147264421f, -0.025761355f}, {+0.112185359f, -0.064164691f}, {+0.062580064f, -0.073563859f}, {+0.021894081f, -0.058426239f}, {+0.000424178f, -0.034842324f}, {-0.005517764f, -0.015891721f}, {-0.004417604f, -0.005456745f}, {-0.002167483f, -0.001310735f}, {-0.000774642f, -0.000155035f}, {-0.000210643f, +0.000031584f}, {-0.000043681f, +0.000023591f}, };
static devConstant float32_x2 PREP_PASTE(GABOR_BANK, DataY3)[22] = { {-0.000048384f, -0.000011117f}, {-0.000058686f, +0.000204754f}, {+0.000746754f, +0.000257808f}, {+0.000952968f, -0.002346883f}, {-0.006355023f, -0.002984121f}, {-0.007951144f, +0.014824705f}, {+0.029786171f, +0.018081795f}, {+0.035172462f, -0.051535171f}, {-0.076759547f, -0.058616359f}, {-0.083796553f, +0.098391056f}, {+0.108491868f, +0.102859005f}, {+0.108491868f, -0.102859005f}, {-0.083796553f, -0.098391056f}, {-0.076759547f, +0.058616359f}, {+0.035172462f, +0.051535171f}, {+0.029786171f, -0.018081795f}, {-0.007951144f, -0.014824705f}, {-0.006355023f, +0.002984121f}, {+0.000952968f, +0.002346883f}, {+0.000746754f, -0.000257808f}, {-0.000058686f, -0.000204754f}, {-0.000048384f, +0.000011117f}, };

static devConstant float32_x2 PREP_PASTE(GABOR_BANK, Freq4) = {-0.055125054f, +0.241518497f};
static devConstant float32_x2 PREP_PASTE(GABOR_BANK, DataX4)[22] = { {-0.000043681f, +0.000023591f}, {-0.000210643f, +0.000031585f}, {-0.000774643f, -0.000155033f}, {-0.002167486f, -0.001310730f}, {-0.004417615f, -0.005456736f}, {-0.005517791f, -0.015891712f}, {+0.000424124f, -0.034842324f}, {+0.021894017f, -0.058426261f}, {+0.062580012f, -0.073563904f}, {+0.112185329f, -0.064164743f}, {+0.147264421f, -0.025761381f}, {+0.147264421f, +0.025761381f}, {+0.112185329f, +0.064164743f}, {+0.062580012f, +0.073563904f}, {+0.021894017f, +0.058426261f}, {+0.000424124f, +0.034842324f}, {-0.005517791f, +0.015891712f}, {-0.004417615f, +0.005456736f}, {-0.002167486f, +0.001310730f}, {-0.000774643f, +0.000155033f}, {-0.000210643f, -0.000031585f}, {-0.000043681f, -0.000023591f}, };
static devConstant float32_x2 PREP_PASTE(GABOR_BANK, DataY4)[22] = { {-0.000048384f, -0.000011117f}, {-0.000058686f, +0.000204754f}, {+0.000746754f, +0.000257808f}, {+0.000952968f, -0.002346883f}, {-0.006355023f, -0.002984121f}, {-0.007951144f, +0.014824705f}, {+0.029786171f, +0.018081795f}, {+0.035172462f, -0.051535171f}, {-0.076759547f, -0.058616359f}, {-0.083796553f, +0.098391056f}, {+0.108491868f, +0.102859005f}, {+0.108491868f, -0.102859005f}, {-0.083796553f, -0.098391056f}, {-0.076759547f, +0.058616359f}, {+0.035172462f, +0.051535171f}, {+0.029786171f, -0.018081795f}, {-0.007951144f, -0.014824705f}, {-0.006355023f, +0.002984121f}, {+0.000952968f, +0.002346883f}, {+0.000746754f, -0.000257808f}, {-0.000058686f, -0.000204754f}, {-0.000048384f, +0.000011117f}, };

static devConstant float32_x2 PREP_PASTE(GABOR_BANK, Freq5) = {-0.154456928f, +0.193682775f};
static devConstant float32_x2 PREP_PASTE(GABOR_BANK, DataX5)[22] = { {-0.000035803f, +0.000034391f}, {-0.000208529f, -0.000043402f}, {-0.000304081f, -0.000729137f}, {+0.001378311f, -0.002125151f}, {+0.007018592f, -0.000175111f}, {+0.009846268f, +0.013639779f}, {-0.011791711f, +0.032789066f}, {-0.060374655f, +0.015744112f}, {-0.072903089f, -0.063348599f}, {+0.014839223f, -0.128384039f}, {+0.132242739f, -0.069730312f}, {+0.132242739f, +0.069730312f}, {+0.014839223f, +0.128384039f}, {-0.072903089f, +0.063348599f}, {-0.060374655f, -0.015744112f}, {-0.011791711f, -0.032789066f}, {+0.009846268f, -0.013639779f}, {+0.007018592f, +0.000175111f}, {+0.001378311f, +0.002125151f}, {-0.000304081f, +0.000729137f}, {-0.000208529f, +0.000043402f}, {-0.000035803f, -0.000034391f}, };
static devConstant float32_x2 PREP_PASTE(GABOR_BANK, DataY5)[22] = { {+0.000048538f, +0.000010424f}, {+0.000114115f, -0.000179850f}, {-0.000479070f, -0.000628170f}, {-0.002421574f, +0.000742959f}, {-0.000394078f, +0.007009707f}, {+0.015428080f, +0.006705735f}, {+0.024102781f, -0.025163930f}, {-0.027312146f, -0.056098331f}, {-0.096105970f, +0.009568087f}, {-0.032552470f, +0.125071988f}, {+0.122668639f, +0.085456796f}, {+0.122668639f, -0.085456796f}, {-0.032552470f, -0.125071988f}, {-0.096105970f, -0.009568087f}, {-0.027312146f, +0.056098331f}, {+0.024102781f, +0.025163930f}, {+0.015428080f, -0.006705735f}, {-0.000394078f, -0.007009707f}, {-0.002421574f, -0.000742959f}, {-0.000479070f, +0.000628170f}, {+0.000114115f, +0.000179850f}, {+0.000048538f, -0.000010424f}, };

static devConstant float32_x2 PREP_PASTE(GABOR_BANK, Freq6) = {-0.223196670f, +0.107485816f};
static devConstant float32_x2 PREP_PASTE(GABOR_BANK, DataX6)[22] = { {-0.000027533f, -0.000041310f}, {+0.000154931f, -0.000146166f}, {+0.000630775f, +0.000475636f}, {-0.001164460f, +0.002249453f}, {-0.006687686f, -0.002136853f}, {+0.002361736f, -0.016655773f}, {+0.034831680f, -0.000959919f}, {+0.012148721f, +0.061199546f}, {-0.090240300f, +0.034417938f}, {-0.065644555f, -0.111325897f}, {+0.114229456f, -0.096447349f}, {+0.114229456f, +0.096447349f}, {-0.065644555f, +0.111325897f}, {-0.090240300f, -0.034417938f}, {+0.012148721f, -0.061199546f}, {+0.034831680f, +0.000959919f}, {+0.002361736f, +0.016655773f}, {-0.006687686f, +0.002136853f}, {-0.001164460f, -0.002249453f}, {+0.000630775f, -0.000475636f}, {+0.000154931f, +0.000146166f}, {-0.000027533f, +0.000041310f}, };
static devConstant float32_x2 PREP_PASTE(GABOR_BANK, DataY6)[22] = { {+0.000034301f, +0.000035889f}, {+0.000211126f, +0.000028176f}, {+0.000676501f, -0.000407986f}, {+0.000875122f, -0.002377009f}, {-0.002225769f, -0.006658622f}, {-0.014136846f, -0.009118232f}, {-0.034662012f, +0.003565447f}, {-0.044450514f, +0.043785017f}, {-0.011330529f, +0.095914155f}, {+0.068405114f, +0.109651282f}, {+0.141057938f, +0.049528960f}, {+0.141057938f, -0.049528960f}, {+0.068405114f, -0.109651282f}, {-0.011330529f, -0.095914155f}, {-0.044450514f, -0.043785017f}, {-0.034662012f, -0.003565447f}, {-0.014136846f, +0.009118232f}, {-0.002225769f, +0.006658622f}, {+0.000875122f, +0.002377009f}, {+0.000676501f, +0.000407986f}, {+0.000211126f, -0.000028176f}, {+0.000034301f, -0.000035889f}, };
