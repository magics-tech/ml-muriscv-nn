#pragma once
#include <stdint.h>

const int8_t depthwise_conv_0_output_ref[] = {127, -128, 63,  -128, -128, 127, 127, 127, -41, 127,
                                              -17, 127,  127, 127,  -128, 127, 127, 127, 127, 127};

const int8_t depthwise_conv_1_output_ref[] =
    {-42, 3, -11, -112, -42, -38, -8, -29, -26, -19, -72, 31, -73, -68, -47, -41};

const int8_t depthwise_conv_2_output_ref[] = {
    100, 10,  18,  -9,  -8,  15,  -73, -54, -24, 41,  -22, 26,  -12, 20,  1,  -48, -55, -30, 5,   -13, 6,
    -17, -14, -24, -41, -14, -36, 87,  -8,  -2,  9,   -15, 56,  -43, -25, -6, 74,  -31, 41,  -32, -15, 24,
    -47, -27, -62, 40,  14,  25,  -17, -7,  -6,  -36, -34, -42, 7,   -28, 13, 7,   1,   48,  -32, 10,  -15,
    5,   -54, 4,   -13, -33, 28,  10,  -15, 12,  5,   -21, 13,  -17, -22, 25, 28,  -31, -17};

const int8_t depthwise_conv_3_output_ref[] = {36, 103, 63, -45, 103, -2};

const int8_t depthwise_conv_4_output_ref[] = {6,   21,  57,  -6,  16,  61, 5,   -29, 98, -5,  0,  42,
                                              -97, -31, 21,  -8,  -26, 26, 44,  25,  56, 2,   7,  54,
                                              5,   -7,  127, -10, -11, 35, -79, -56, 15, -32, -1, 38};

const int8_t depthwise_conv_5_output_ref[] = {1,  -91, 52, -57,  65,  -99, 62, 59,   55, -110, 97,  -2,  -15, -128,
                                              85, 36,  38, -128, -13, -56, 49, -107, 29, 62,   -26, -96, 29,  -128,
                                              3,  -16, 41, -13,  14,  -54, 22, -96,  17, -17,  13,  -15};

const int8_t depthwise_conv_6_output_ref[] = {
    -15, 73,  43,  33,  19,  -14, -58,  11,  -65, 53,  57,  -12, -6,   -29, -69, -1,  -27, 29,  27,  43,
    61,  -5,  -65, -12, -16, 37,  6,    18,  8,   -5,  11,  25,  -104, 52,  78,  36,  -39, -63, -36, 37,
    -36, 66,  26,  71,  22,  -50, -123, -13, -47, 40,  56,  34,  66,   -7,  -57, 12,  -38, 61,  10,  15,
    6,   1,   11,  16,  -30, 79,  79,   21,  55,  -25, -41, -24, -113, 71,  93,  19,  -27, -92, -45, 25,
    -22, 95,  35,  17,  -27, -60, -82,  55,  -44, 47,  38,  -46, 81,   29,  -85, -23, -72, 94,  60,  29,
    9,   -20, -58, 6,   -35, 70,  90,   7,   95,  -14, -70, -48, -109, 78,  57,  4,   -25, -93, -18, 36,
    -5,  34,  -7,  43,  39,  -16, -19,  42,  -50, 81,  81,  -95, 23,   -25, 19,  14,  -10, 55,  64,  -59,
    -1,  -18, 31,  33,  -20, 60,  66,   -66, 58,  -13, -15, -29, -37,  41,  34,  -40, -21, -30, 54,  62};

const int8_t depthwise_conv_7_output_ref[] = {
    -19, 8,  18,  0,   2,   14,  -38, 2,   -41, 3,   7,  -7,  -19, 44,  8,   -43, -3, -33, 30,  -6,  11,
    -31, 23, 24,  -46, 14,  -46, 11,  2,   -9,  -21, 4,  29,  -70, 10,  -25, 11,  -6, 18,  -36, -16, 0,
    -70, 27, -45, 9,   13,  -22, -4,  -13, 38,  -54, 18, -41, 16,  2,   2,   -31, 5,  29,  -46, -10, -33,
    80,  -2, -13, -42, -24, 31,  -70, 6,   -66, 54,  11, -26, -33, -15, 21,  -23, 4,  -65};