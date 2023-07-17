#pragma once

#include "prepTools/prepBase.h"

//================================================================
//
// PREP_INC
// PREP_DEC
//
// Increment/decrement value at preprocessing time.
//
//================================================================

#define PREP_INC(n) \
    PREP_INC_EX(n)

#define PREP_INC_EX(n) \
    PREP_INC_##n

//----------------------------------------------------------------

#define PREP_DEC(n) \
    PREP_DEC_EX(n)

#define PREP_DEC_EX(n) \
    PREP_DEC_##n

//================================================================
//
// PREP_INC
//
//================================================================

#define PREP_INC_0 1
#define PREP_INC_1 2
#define PREP_INC_2 3
#define PREP_INC_3 4
#define PREP_INC_4 5
#define PREP_INC_5 6
#define PREP_INC_6 7
#define PREP_INC_7 8
#define PREP_INC_8 9
#define PREP_INC_9 10
#define PREP_INC_10 11
#define PREP_INC_11 12
#define PREP_INC_12 13
#define PREP_INC_13 14
#define PREP_INC_14 15
#define PREP_INC_15 16
#define PREP_INC_16 17
#define PREP_INC_17 18
#define PREP_INC_18 19
#define PREP_INC_19 20
#define PREP_INC_20 21
#define PREP_INC_21 22
#define PREP_INC_22 23
#define PREP_INC_23 24
#define PREP_INC_24 25
#define PREP_INC_25 26
#define PREP_INC_26 27
#define PREP_INC_27 28
#define PREP_INC_28 29
#define PREP_INC_29 30
#define PREP_INC_30 31
#define PREP_INC_31 32
#define PREP_INC_32 33
#define PREP_INC_33 34
#define PREP_INC_34 35
#define PREP_INC_35 36
#define PREP_INC_36 37
#define PREP_INC_37 38
#define PREP_INC_38 39
#define PREP_INC_39 40
#define PREP_INC_40 41
#define PREP_INC_41 42
#define PREP_INC_42 43
#define PREP_INC_43 44
#define PREP_INC_44 45
#define PREP_INC_45 46
#define PREP_INC_46 47
#define PREP_INC_47 48
#define PREP_INC_48 49
#define PREP_INC_49 50
#define PREP_INC_50 51
#define PREP_INC_51 52
#define PREP_INC_52 53
#define PREP_INC_53 54
#define PREP_INC_54 55
#define PREP_INC_55 56
#define PREP_INC_56 57
#define PREP_INC_57 58
#define PREP_INC_58 59
#define PREP_INC_59 60
#define PREP_INC_60 61
#define PREP_INC_61 62
#define PREP_INC_62 63
#define PREP_INC_63 64
#define PREP_INC_64 65
#define PREP_INC_65 66
#define PREP_INC_66 67
#define PREP_INC_67 68
#define PREP_INC_68 69
#define PREP_INC_69 70
#define PREP_INC_70 71
#define PREP_INC_71 72
#define PREP_INC_72 73
#define PREP_INC_73 74
#define PREP_INC_74 75
#define PREP_INC_75 76
#define PREP_INC_76 77
#define PREP_INC_77 78
#define PREP_INC_78 79
#define PREP_INC_79 80
#define PREP_INC_80 81
#define PREP_INC_81 82
#define PREP_INC_82 83
#define PREP_INC_83 84
#define PREP_INC_84 85
#define PREP_INC_85 86
#define PREP_INC_86 87
#define PREP_INC_87 88
#define PREP_INC_88 89
#define PREP_INC_89 90
#define PREP_INC_90 91
#define PREP_INC_91 92
#define PREP_INC_92 93
#define PREP_INC_93 94
#define PREP_INC_94 95
#define PREP_INC_95 96
#define PREP_INC_96 97
#define PREP_INC_97 98
#define PREP_INC_98 99
#define PREP_INC_99 100
#define PREP_INC_100 101
#define PREP_INC_101 102
#define PREP_INC_102 103
#define PREP_INC_103 104
#define PREP_INC_104 105
#define PREP_INC_105 106
#define PREP_INC_106 107
#define PREP_INC_107 108
#define PREP_INC_108 109
#define PREP_INC_109 110
#define PREP_INC_110 111
#define PREP_INC_111 112
#define PREP_INC_112 113
#define PREP_INC_113 114
#define PREP_INC_114 115
#define PREP_INC_115 116
#define PREP_INC_116 117
#define PREP_INC_117 118
#define PREP_INC_118 119
#define PREP_INC_119 120
#define PREP_INC_120 121
#define PREP_INC_121 122
#define PREP_INC_122 123
#define PREP_INC_123 124
#define PREP_INC_124 125
#define PREP_INC_125 126
#define PREP_INC_126 127
#define PREP_INC_127 128
#define PREP_INC_128 129
#define PREP_INC_129 130
#define PREP_INC_130 131
#define PREP_INC_131 132
#define PREP_INC_132 133
#define PREP_INC_133 134
#define PREP_INC_134 135
#define PREP_INC_135 136
#define PREP_INC_136 137
#define PREP_INC_137 138
#define PREP_INC_138 139
#define PREP_INC_139 140
#define PREP_INC_140 141
#define PREP_INC_141 142
#define PREP_INC_142 143
#define PREP_INC_143 144
#define PREP_INC_144 145
#define PREP_INC_145 146
#define PREP_INC_146 147
#define PREP_INC_147 148
#define PREP_INC_148 149
#define PREP_INC_149 150
#define PREP_INC_150 151
#define PREP_INC_151 152
#define PREP_INC_152 153
#define PREP_INC_153 154
#define PREP_INC_154 155
#define PREP_INC_155 156
#define PREP_INC_156 157
#define PREP_INC_157 158
#define PREP_INC_158 159
#define PREP_INC_159 160
#define PREP_INC_160 161
#define PREP_INC_161 162
#define PREP_INC_162 163
#define PREP_INC_163 164
#define PREP_INC_164 165
#define PREP_INC_165 166
#define PREP_INC_166 167
#define PREP_INC_167 168
#define PREP_INC_168 169
#define PREP_INC_169 170
#define PREP_INC_170 171
#define PREP_INC_171 172
#define PREP_INC_172 173
#define PREP_INC_173 174
#define PREP_INC_174 175
#define PREP_INC_175 176
#define PREP_INC_176 177
#define PREP_INC_177 178
#define PREP_INC_178 179
#define PREP_INC_179 180
#define PREP_INC_180 181
#define PREP_INC_181 182
#define PREP_INC_182 183
#define PREP_INC_183 184
#define PREP_INC_184 185
#define PREP_INC_185 186
#define PREP_INC_186 187
#define PREP_INC_187 188
#define PREP_INC_188 189
#define PREP_INC_189 190
#define PREP_INC_190 191
#define PREP_INC_191 192
#define PREP_INC_192 193
#define PREP_INC_193 194
#define PREP_INC_194 195
#define PREP_INC_195 196
#define PREP_INC_196 197
#define PREP_INC_197 198
#define PREP_INC_198 199
#define PREP_INC_199 200
#define PREP_INC_200 201
#define PREP_INC_201 202
#define PREP_INC_202 203
#define PREP_INC_203 204
#define PREP_INC_204 205
#define PREP_INC_205 206
#define PREP_INC_206 207
#define PREP_INC_207 208
#define PREP_INC_208 209
#define PREP_INC_209 210
#define PREP_INC_210 211
#define PREP_INC_211 212
#define PREP_INC_212 213
#define PREP_INC_213 214
#define PREP_INC_214 215
#define PREP_INC_215 216
#define PREP_INC_216 217
#define PREP_INC_217 218
#define PREP_INC_218 219
#define PREP_INC_219 220
#define PREP_INC_220 221
#define PREP_INC_221 222
#define PREP_INC_222 223
#define PREP_INC_223 224
#define PREP_INC_224 225
#define PREP_INC_225 226
#define PREP_INC_226 227
#define PREP_INC_227 228
#define PREP_INC_228 229
#define PREP_INC_229 230
#define PREP_INC_230 231
#define PREP_INC_231 232
#define PREP_INC_232 233
#define PREP_INC_233 234
#define PREP_INC_234 235
#define PREP_INC_235 236
#define PREP_INC_236 237
#define PREP_INC_237 238
#define PREP_INC_238 239
#define PREP_INC_239 240
#define PREP_INC_240 241
#define PREP_INC_241 242
#define PREP_INC_242 243
#define PREP_INC_243 244
#define PREP_INC_244 245
#define PREP_INC_245 246
#define PREP_INC_246 247
#define PREP_INC_247 248
#define PREP_INC_248 249
#define PREP_INC_249 250
#define PREP_INC_250 251
#define PREP_INC_251 252
#define PREP_INC_252 253
#define PREP_INC_253 254
#define PREP_INC_254 255
#define PREP_INC_255 256
#define PREP_INC_256 257

//================================================================
//
// PREP_DEC
//
//================================================================

#define PREP_DEC_0 MINUS1
#define PREP_DEC_1 0
#define PREP_DEC_2 1
#define PREP_DEC_3 2
#define PREP_DEC_4 3
#define PREP_DEC_5 4
#define PREP_DEC_6 5
#define PREP_DEC_7 6
#define PREP_DEC_8 7
#define PREP_DEC_9 8
#define PREP_DEC_10 9
#define PREP_DEC_11 10
#define PREP_DEC_12 11
#define PREP_DEC_13 12
#define PREP_DEC_14 13
#define PREP_DEC_15 14
#define PREP_DEC_16 15
#define PREP_DEC_17 16
#define PREP_DEC_18 17
#define PREP_DEC_19 18
#define PREP_DEC_20 19
#define PREP_DEC_21 20
#define PREP_DEC_22 21
#define PREP_DEC_23 22
#define PREP_DEC_24 23
#define PREP_DEC_25 24
#define PREP_DEC_26 25
#define PREP_DEC_27 26
#define PREP_DEC_28 27
#define PREP_DEC_29 28
#define PREP_DEC_30 29
#define PREP_DEC_31 30
#define PREP_DEC_32 31
#define PREP_DEC_33 32
#define PREP_DEC_34 33
#define PREP_DEC_35 34
#define PREP_DEC_36 35
#define PREP_DEC_37 36
#define PREP_DEC_38 37
#define PREP_DEC_39 38
#define PREP_DEC_40 39
#define PREP_DEC_41 40
#define PREP_DEC_42 41
#define PREP_DEC_43 42
#define PREP_DEC_44 43
#define PREP_DEC_45 44
#define PREP_DEC_46 45
#define PREP_DEC_47 46
#define PREP_DEC_48 47
#define PREP_DEC_49 48
#define PREP_DEC_50 49
#define PREP_DEC_51 50
#define PREP_DEC_52 51
#define PREP_DEC_53 52
#define PREP_DEC_54 53
#define PREP_DEC_55 54
#define PREP_DEC_56 55
#define PREP_DEC_57 56
#define PREP_DEC_58 57
#define PREP_DEC_59 58
#define PREP_DEC_60 59
#define PREP_DEC_61 60
#define PREP_DEC_62 61
#define PREP_DEC_63 62
#define PREP_DEC_64 63
#define PREP_DEC_65 64
#define PREP_DEC_66 65
#define PREP_DEC_67 66
#define PREP_DEC_68 67
#define PREP_DEC_69 68
#define PREP_DEC_70 69
#define PREP_DEC_71 70
#define PREP_DEC_72 71
#define PREP_DEC_73 72
#define PREP_DEC_74 73
#define PREP_DEC_75 74
#define PREP_DEC_76 75
#define PREP_DEC_77 76
#define PREP_DEC_78 77
#define PREP_DEC_79 78
#define PREP_DEC_80 79
#define PREP_DEC_81 80
#define PREP_DEC_82 81
#define PREP_DEC_83 82
#define PREP_DEC_84 83
#define PREP_DEC_85 84
#define PREP_DEC_86 85
#define PREP_DEC_87 86
#define PREP_DEC_88 87
#define PREP_DEC_89 88
#define PREP_DEC_90 89
#define PREP_DEC_91 90
#define PREP_DEC_92 91
#define PREP_DEC_93 92
#define PREP_DEC_94 93
#define PREP_DEC_95 94
#define PREP_DEC_96 95
#define PREP_DEC_97 96
#define PREP_DEC_98 97
#define PREP_DEC_99 98
#define PREP_DEC_100 99
#define PREP_DEC_101 100
#define PREP_DEC_102 101
#define PREP_DEC_103 102
#define PREP_DEC_104 103
#define PREP_DEC_105 104
#define PREP_DEC_106 105
#define PREP_DEC_107 106
#define PREP_DEC_108 107
#define PREP_DEC_109 108
#define PREP_DEC_110 109
#define PREP_DEC_111 110
#define PREP_DEC_112 111
#define PREP_DEC_113 112
#define PREP_DEC_114 113
#define PREP_DEC_115 114
#define PREP_DEC_116 115
#define PREP_DEC_117 116
#define PREP_DEC_118 117
#define PREP_DEC_119 118
#define PREP_DEC_120 119
#define PREP_DEC_121 120
#define PREP_DEC_122 121
#define PREP_DEC_123 122
#define PREP_DEC_124 123
#define PREP_DEC_125 124
#define PREP_DEC_126 125
#define PREP_DEC_127 126
#define PREP_DEC_128 127
#define PREP_DEC_129 128
#define PREP_DEC_130 129
#define PREP_DEC_131 130
#define PREP_DEC_132 131
#define PREP_DEC_133 132
#define PREP_DEC_134 133
#define PREP_DEC_135 134
#define PREP_DEC_136 135
#define PREP_DEC_137 136
#define PREP_DEC_138 137
#define PREP_DEC_139 138
#define PREP_DEC_140 139
#define PREP_DEC_141 140
#define PREP_DEC_142 141
#define PREP_DEC_143 142
#define PREP_DEC_144 143
#define PREP_DEC_145 144
#define PREP_DEC_146 145
#define PREP_DEC_147 146
#define PREP_DEC_148 147
#define PREP_DEC_149 148
#define PREP_DEC_150 149
#define PREP_DEC_151 150
#define PREP_DEC_152 151
#define PREP_DEC_153 152
#define PREP_DEC_154 153
#define PREP_DEC_155 154
#define PREP_DEC_156 155
#define PREP_DEC_157 156
#define PREP_DEC_158 157
#define PREP_DEC_159 158
#define PREP_DEC_160 159
#define PREP_DEC_161 160
#define PREP_DEC_162 161
#define PREP_DEC_163 162
#define PREP_DEC_164 163
#define PREP_DEC_165 164
#define PREP_DEC_166 165
#define PREP_DEC_167 166
#define PREP_DEC_168 167
#define PREP_DEC_169 168
#define PREP_DEC_170 169
#define PREP_DEC_171 170
#define PREP_DEC_172 171
#define PREP_DEC_173 172
#define PREP_DEC_174 173
#define PREP_DEC_175 174
#define PREP_DEC_176 175
#define PREP_DEC_177 176
#define PREP_DEC_178 177
#define PREP_DEC_179 178
#define PREP_DEC_180 179
#define PREP_DEC_181 180
#define PREP_DEC_182 181
#define PREP_DEC_183 182
#define PREP_DEC_184 183
#define PREP_DEC_185 184
#define PREP_DEC_186 185
#define PREP_DEC_187 186
#define PREP_DEC_188 187
#define PREP_DEC_189 188
#define PREP_DEC_190 189
#define PREP_DEC_191 190
#define PREP_DEC_192 191
#define PREP_DEC_193 192
#define PREP_DEC_194 193
#define PREP_DEC_195 194
#define PREP_DEC_196 195
#define PREP_DEC_197 196
#define PREP_DEC_198 197
#define PREP_DEC_199 198
#define PREP_DEC_200 199
#define PREP_DEC_201 200
#define PREP_DEC_202 201
#define PREP_DEC_203 202
#define PREP_DEC_204 203
#define PREP_DEC_205 204
#define PREP_DEC_206 205
#define PREP_DEC_207 206
#define PREP_DEC_208 207
#define PREP_DEC_209 208
#define PREP_DEC_210 209
#define PREP_DEC_211 210
#define PREP_DEC_212 211
#define PREP_DEC_213 212
#define PREP_DEC_214 213
#define PREP_DEC_215 214
#define PREP_DEC_216 215
#define PREP_DEC_217 216
#define PREP_DEC_218 217
#define PREP_DEC_219 218
#define PREP_DEC_220 219
#define PREP_DEC_221 220
#define PREP_DEC_222 221
#define PREP_DEC_223 222
#define PREP_DEC_224 223
#define PREP_DEC_225 224
#define PREP_DEC_226 225
#define PREP_DEC_227 226
#define PREP_DEC_228 227
#define PREP_DEC_229 228
#define PREP_DEC_230 229
#define PREP_DEC_231 230
#define PREP_DEC_232 231
#define PREP_DEC_233 232
#define PREP_DEC_234 233
#define PREP_DEC_235 234
#define PREP_DEC_236 235
#define PREP_DEC_237 236
#define PREP_DEC_238 237
#define PREP_DEC_239 238
#define PREP_DEC_240 239
#define PREP_DEC_241 240
#define PREP_DEC_242 241
#define PREP_DEC_243 242
#define PREP_DEC_244 243
#define PREP_DEC_245 244
#define PREP_DEC_246 245
#define PREP_DEC_247 246
#define PREP_DEC_248 247
#define PREP_DEC_249 248
#define PREP_DEC_250 249
#define PREP_DEC_251 250
#define PREP_DEC_252 251
#define PREP_DEC_253 252
#define PREP_DEC_254 253
#define PREP_DEC_255 254
#define PREP_DEC_256 255

//================================================================
//
// generator
//
//================================================================

/*
#include <stdio.h>

void main()
{
    const int maxNumber = 256;

    for (int i = 0; i <= maxNumber; ++i)
        printf("#define PREP_INC_%d %d\n", i, i+1);

    printf("\n");

    for (int i = 1; i <= maxNumber; ++i)
        printf("#define PREP_DEC_%d %d\n", i, i-1);
}
*/
