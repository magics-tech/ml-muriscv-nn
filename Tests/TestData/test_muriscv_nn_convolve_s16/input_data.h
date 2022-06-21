#pragma once
#include <stdint.h>

const int64_t conv_0_biases[] = {-2495236, -432952, -2150622, 3037347};
const int16_t conv_0_input[] = {
    11645,  22850,  31319,  -24644, 2819,   1883,   -26935, -18937, 22831,  28845,  31347,  -31674, -27727, -23552,
    -10251, -13891, 26029,  17279,  -25122, 6570,   2539,   -4245,  -23085, 18088,  -23145, -32457, -29942, -6541,
    -31537, -1148,  8908,   3454,   20995,  2559,   9127,   -7295,  20003,  -19625, 3042,   5151,   -9477,  -32504,
    -32554, -7262,  16464,  -9399,  3808,   -16672, 16164,  -3372,  -22958, 7014,   -4084,  18590,  -13109, 26520,
    4925,   -3094,  -27051, -14280, -9399,  29082,  -16574, 5120,   -1813,  24619,  13455,  -7827,  -21207, 26555,
    -380,   31052,  9576,   -10092, -21755, -16522, 4976,   -438,   3725,   -22109, -18201, 9708,   32288,  8873,
    4352,   5560,   -4372,  19609,  32721,  2994,   23154,  -10829, -22219, -13091, 5930,   -17281, -19958, -3651,
    4127,   9918,   28773,  -2436,  -30694, -28239, -12289, -25622, -17760, -14319, -27872, 6694,   -522,   27721,
    -12826, -32554, -11147, 18112,  7633,   13587,  -23309, 14490,  28782,  -2318,  29818,  -24295, 21169,  18585,
    31536,  -8885,  4362,   14410,  8865,   -22691, -6651,  -6278,  -31346, 14182,  21068,  8149,   -4410,  -2909,
    -16315, -2756,  -28748, -28150, -2663,  1654,   -21926, 1999,   11965,  -12122, -12450, -27546, 31965,  -21814,
    26971,  -32390, 2906,   -8975,  -17418, -17112, 19933,  24141,  24802,  19217,  7530,   -3580,  1147,   -20321};
const int32_t conv_0_output_mult[] = {1388772064, 1391969415, 1360038279, 1365367254};
const int32_t conv_0_output_shift[] = {-9, -9, -9, -9};
const int8_t conv_0_weights[] = {
    -113, 41,  -99,  -23, -59,  -65, -88,  -123, 83,   127, -121, 97,  -96, -94,  52,  -8,   6,    -6,  90,   15,
    87,   111, -56,  -42, 80,   115, -1,   72,   116,  107, 67,   124, -50, -30,  -6,  23,   -127, 102, 58,   64,
    34,   52,  111,  19,  -104, -2,  -95,  -125, 77,   -16, -5,   20,  -21, -3,   91,  -124, -47,  -83, -67,  -3,
    124,  111, 101,  -21, 79,   127, -124, 69,   -15,  -82, 11,   113, 46,  -102, 95,  32,   -32,  -7,  -114, -102,
    32,   62,  -127, 87,  83,   -3,  -67,  97,   -100, -7,  34,   -73, 99,  -61,  -15, -104};

const int64_t conv_1_biases[] = {-10468545, -9859915};
const int16_t conv_1_input[] = {
    -68,  -99,  -36,  -73,  -123, -90,  -48,  -50,  71,   -110, 76,   -76,  -28,  -27,  -43,  -62,  -32,  74,   -8,
    56,   -11,  22,   -105, 99,   -127, -119, -78,  48,   68,   91,   -82,  -54,  7,    -72,  -2,   126,  -2,   -32,
    -59,  108,  53,   -67,  -103, -127, -112, -58,  49,   -3,   -88,  24,   -112, -6,   57,   -86,  123,  -57,  -103,
    -114, -17,  24,   -83,  72,   -80,  -109, 109,  23,   -124, 55,   -114, -102, -123, 23,   77,   -34,  82,   -26,
    57,   41,   121,  121,  -61,  26,   108,  -101, 95,   31,   54,   78,   -12,  -35,  12,   -6,   52,   -14,  82,
    108,  101,  -46,  73,   -52,  -110, -127, 107,  69,   8,    16,   73,   -109, -120, 51,   64,   -101, -63,  -34,
    104,  -75,  -92,  -31,  -10,  -79,  28,   -112, -33,  -117, -23,  -21,  -115, 9,    -88,  32,   86,   -41,  17,
    70,   -9,   41,   8,    48,   -106, -31,  -107, 122,  -83,  112,  -17,  35,   55,   94,   1,    -40,  -27,  42,
    25,   -45,  102,  -124, 33,   -29,  -101, -114, -125, 87,   -96,  13,   -69,  28,   -115, 101,  -1,   103,  0,
    98,   111,  42,   19,   71,   102,  18,   79,   90,   -50,  -4,   -86,  -89,  10,   82,   30,   -82,  38,   -115,
    49,   -41,  -125, 93,   -30,  14,   20,   -14,  -106, 112,  -58,  -8,   46,   25,   57,   -128, -115, 115,  109,
    -111, -120, -123, -14,  -74,  -94,  -96,  63,   72,   29,   63,   -46,  49,   26,   100,  -112, 76,   -37,  115,
    125,  -117, -120, 104,  -113, -103, -76,  16,   108,  19,   19,   -22,  113,  -72,  -12,  -72,  -23,  -54,  -71,
    -82,  -88,  -17,  88,   -36,  -61,  28,   -52,  -93,  -4,   105,  -111, 15,   19,   4,    -124, -103, 13,   -50,
    64,   126,  -89,  17,   115,  34,   -78,  48,   -35,  -103, -37,  11,   -124, 84,   22,   -92,  -87,  12,   0,
    52,   -67,  67,   77,   8,    -120, -16,  -18,  -37,  4,    124,  -18,  74,   67,   -60,  8,    13,   -120, 73,
    100,  121,  70,   -100, -96,  -85,  77,   75,   95,   -121, -4,   14,   72,   -74,  95,   61,   -4,   -87,  -49,
    -53,  21,   -44,  -3,   -41,  13,   99,   124,  12,   -82,  121,  67,   8,    -112, -109, 101,  -5,   -16,  -98,
    -126, -23,  17,   -95,  -7,   116,  -2,   55,   98,   -67,  -66,  121,  34,   -6,   -108, -34,  7,    -117, -14,
    -19,  8,    83,   -61,  41,   -74,  120,  -8,   -3,   -55,  -32,  112,  -9,   -96,  88,   55,   58,   -127, 35,
    29,   8,    124,  116,  108,  55,   -96,  -10,  43,   -42,  120,  78,   1,    27,   37,   99,   -74,  123,  116,
    60,   -12,  -78,  75,   -54,  -95,  47,   94,   49,   -76,  -63,  -104, -31,  21,   57,   -10,  67,   8,    -13,
    94,   42,   45,   -91,  -117, 53,   110,  -32,  -102, -126, 35,   -90,  -75,  111,  76,   114,  -53,  34,   -70,
    -117, 82,   20,   96,   48,   -25,  105,  33,   66,   -74,  -41,  122,  61,   113,  -97,  -75,  106,  -119, -48,
    38,   82,   43,   74,   -34,  94,   29,   108,  70,   -63,  109,  -45,  123,  -30,  -123, -105, 43,   -124, -37,
    -92,  55,   -73,  92,   -119, -57,  -18,  110,  -99,  70,   -57,  -15,  -44,  -128, -126, 51,   100,  -79,  47,
    77,   -76,  52,   -9,   99,   -60,  -52,  -64,  48,   -70,  80,   2,    79,   110,  -13,  45,   -86,  83,   -38,
    -86,  33,   -17,  109,  -112, 45,   117,  60,   -67,  -123, 2,    67,   48,   -75,  -88,  121,  -117, -111, 5,
    87,   -128, -69,  -40,  119,  108,  -109, 79,   -15,  -31,  102,  120,  16,   37,   -39,  -51,  66,   102,  -35,
    107,  84,   10,   36,   -57,  108,  -1,   51,   27,   -82,  -77,  24,   121,  -67,  -18,  -34,  -92,  77,   62,
    -107, 52,   50,   96,   -88,  -23,  110,  -35,  -24,  9,    -33,  -68,  -61,  27,   -81,  -47,  13,   19,   108,
    32,   50,   -67,  78,   -124, -22,  19,   62,   -50,  -38,  -70,  93,   -78,  -7,   27,   72,   53,   49,   -121,
    98,   -10,  -6,   -125, -74,  -23,  -17,  -84,  -43,  14,   107,  -128, 107,  -74,  -111, -65,  124,  118,  70,
    72,   -109, -33,  -31,  -76,  66,   125,  61,   -60,  -54,  66,   101,  103,  31,   -81,  -88,  -78,  49,   35,
    -32,  -40,  125,  79,   -86,  123,  -89,  -3,   38,   92,   -35,  55,   -117, -86,  -84,  31,   36,   75,   -53,
    116,  3,    51,   71,   125,  -32,  23,   89,   96,   24,   -86,  18,   35,   7,    -19,  10,   -10,  -108, -13,
    -22,  -57,  -62,  -69,  -41,  -8,   116,  76,   -51,  -84,  -37,  -115, -61,  64,   -118, 39,   -62,  56,   -90,
    106,  -65,  -116, 88,   -102, 33,   -4,   -53,  40,   54,   -69,  -11,  97,   -117, 112,  91,   1,    -96,  -21,
    108,  115,  6,    -107, 23,   108,  81,   45,   -36,  26,   -16,  81,   -44,  117,  -117, 117,  56,   43,   -63,
    25,   -66,  -16,  -17,  -25,  61,   -14,  -23,  106,  72,   -9,   -67,  -89,  91,   81,   102,  50,   -37,  -78,
    -75,  -16,  17,   -50,  -39,  -99,  10,   -21,  85,   59,   17,   -61,  79,   -92,  95,   -98,  -8,   85,   49,
    -52,  -93,  114,  -95,  2,    64,   -57,  -92,  -117, 106,  122,  36,   -71,  120,  -40,  -12,  -62,  -81,  1,
    18,   40,   19,   -111, 66,   13,   -38,  -50,  77,   33,   -26,  88,   -60,  -111, 82,   -48,  -18,  68,   -73,
    101,  77,   102,  76,   -2,   62,   63,   99,   -9,   18,   55,   17,   -34,  -59,  -11,  3,    2,    -32,  68,
    -66,  -25,  -115, -59,  -86,  100,  124,  11,   -64,  5,    -49,  -50,  -54,  0,    113,  11,   -63,  75,   -125,
    105,  -95,  -6,   -84,  31,   -116, 5,    -80,  34,   9,    53,   -58,  -36,  39,   48,   92,   -33,  -82,  105,
    88,   -53,  -72,  -92,  -67,  -92,  81,   -39,  -61,  -61,  -73,  112,  108,  -104, -28,  21,   1,    -95,  -126,
    -24,  -50,  -41,  79,   -90,  108,  -17,  -60,  38,   37,   39,   -115, 28,   6,    -3,   -29,  -65,  38,   -10,
    18,   86,   -23,  -8,   -77,  -36,  -64,  43,   -97,  19,   93,   -108, 2,    33,   -32,  -107, -98,  -108, 14,
    -91,  -27,  77,   -44,  8,    31,   121,  -121, -79,  94,   -72,  -20,  90,   65,   -40,  52,   4,    11,   -46,
    79,   -121, -43,  113,  -44,  103,  -61,  -102, 19,   85,   -18,  -94,  89,   -86,  -126, 26,   -88,  -51,  91,
    -21,  42,   115,  9,    10,   -44,  31,   -52,  118,  -121, -72,  54,   62,   51,   28,   69,   -108, -121, -67,
    121,  -100, 99,   74,   -50,  -90,  88,   -79,  19,   109,  -105, -73,  49,   -126, -69,  -61,  25,   -124, -56,
    -39,  -108, 68,   -51,  -6,   -99,  121,  40,   19,   14,   -78,  -15,  74,   64,   35,   92,   -61,  93,   -56,
    -78,  69,   30,   37,   -120, 42,   -7,   18,   115,  34,   -127, -49,  -87,  58,   -123, -1,   -74,  34,   -126,
    77,   -1,   -58,  42,   -13,  39,   10,   -70,  101,  -45,  -57,  -54,  -50,  19,   111,  50,   -114, 59,   59,
    67,   -69,  -120, 60,   64,   -67,  -57,  -32,  -122, 121,  -61,  -10,  97,   46,   104,  60,   102,  33,   1,
    28,   -63,  -63,  -37,  1,    -60,  5,    -51,  -122, -99,  -113, 9,    87,   74,   -27,  64,   12,   -124, 28,
    15,   28,   47,   126,  -41,  -42,  -87,  125,  -110, 31,   -11,  115,  -77,  -102, -87,  -62,  122,  54,   -83,
    -21,  62,   4,    94,   73,   -38,  -90,  101,  -34,  -102, 6,    44,   109,  -48,  95,   46,   116,  113,  96,
    18,   -9,   -21,  26,   -105, 112,  126,  -126, 58,   -99,  -119, -86,  104,  31,   -8,   46,   -82,  60,   -73,
    -23,  -22,  -16,  -55,  48,   -22,  -111, -110, -23,  42,   118,  -45,  -47,  27,   -60,  -62,  -58,  -46,  95,
    94,   -45,  51,   -44,  -80,  90,   47,   -23,  82,   -7,   -117, 88,   -53,  -111, 20,   -19,  88,   52,   -26,
    -8,   92,   -45,  113,  45,   -92,  3,    86,   -40,  100,  100,  92,   -68,  -43,  46,   -114, 54,   17,   -82,
    0,    25,   -126, -84,  -104, 83,   -102, 13,   27,   -114, 44,   50,   -100, -61,  84,   9,    -105, 102,  -44,
    126,  58,   -70,  -71,  -93,  -5,   -58,  116,  34,   -68,  -108, 55,   -75,  -10,  9,    -106, 117,  111,  -46,
    -5,   33,   71,   -30,  119,  35,   -11,  -3,   -55,  -114, -107, -110, 29,   -95,  14,   48,   -64,  70,   -36,
    117,  97,   -104, 112,  -17,  2,    36,   -99,  83,   -52,  29,   77,   16,   -77,  -113, 78,   104,  -114, 78,
    25,   -10,  40,   33,   17,   -27,  113,  54,   -18,  16,   -4,   -83,  25,   106,  -51,  -19,  -122, 97,   3,
    -97,  30,   -62,  26,   22,   -84,  -89,  75,   123,  -127, 8,    -53,  47,   -125, 35,   -122, -28,  45,   -56,
    -64,  -41,  -40,  -52,  -66,  -87,  0,    -16,  -58,  40,   -31,  -88,  82,   -90,  92,   -65,  61,   -32,  -41,
    -59,  21,   -1,   56,   85,   -115, -85,  42,   -121, 117,  -47,  119,  113,  -52,  23,   -101, -105, -49,  106,
    26,   -93,  105,  92,   -49,  -94,  -32,  -13,  45,   2,    42,   67,   82,   116,  93,   -121, 57,   7,    67,
    -121, -93,  -109, -68,  125,  -46,  -53,  -72,  -42,  -29,  64,   -8,   37,   -52,  53,   88,   -21,  87,   78,
    -60,  -94,  -57,  105,  -64,  -93,  -36,  42,   -51,  102,  -53,  -42,  -8,   30,   20,   76,   -32,  78,   -89,
    -50,  -85,  15,   82,   -13,  -117, 64,   -34,  -10,  -119, -36,  59,   -37,  -108, 78,   -31,  -34,  56,   -67,
    13,   82,   -26,  -53,  66,   66,   -42,  36,   23,   -122, 113,  -68,  -54,  12,   -84,  -2,   70,   78,   -115,
    -21,  -21,  -91,  -125, 106,  -22,  -115, -82,  89,   86,   -123, 97,   49,   -24,  -46,  -106, -47,  31,   -60,
    -127, 24,   -29,  -116, -100, 97,   124,  51,   122,  89,   41,   55,   -36,  -103, 78,   -126, 114,  -72,  39,
    64,   106,  -99,  121,  -50,  50,   -124, -83,  6,    -107, 70,   -15,  -69,  -84,  106,  115,  -69,  -101, 45,
    -55,  -88,  86,   50,   -30,  -125, -95,  -28,  59,   97,   66,   -118, 119,  72,   61,   60,   28,   54,   21,
    -3,   125,  7,    -16,  -98,  7,    -14,  -43,  48,   47,   -60,  118,  -83,  -4,   -5,   -68,  -73,  37,   -99,
    -61,  24,   73,   80,   11,   -92,  -65,  -24,  -30,  -41,  108,  57,   6,    55,   -79,  -90,  49,   -50,  7,
    -42,  44,   20,   37,   -118, -126, -126, -31,  13,   -87,  -117, 43,   -100, -80,  76,   33,   117,  47,   113,
    -92,  116,  -120, 2,    96,   -53,  -39,  27,   119,  -84,  -117, -123, -116, 16,   -5,   45,   47,   95,   -97,
    -55,  -31,  -51,  70,   87,   -122, 11,   -35,  -28,  98,   66,   97,   -98,  -1,   -91,  41,   -8,   120,  58,
    76,   58,   10,   83,   -83,  4,    -3,   -82,  62,   -125, -101, 74,   -123, 122,  -106, 83,   -75,  -73,  -117,
    23,   -107, 4,    38,   -125, 117,  -125, 109,  -109, 53,   125,  28,   -107, 64,   38,   5,    42,   76,   -23,
    32,   80,   -10,  35,   5,    20,   84,   -6,   -59,  19,   103,  -102, -88,  84,   -95,  -53,  30,   89,   73,
    117,  -25,  -34,  74,   82,   65,   8,    -5,   110,  -72,  -120, 92,   -62,  46,   -94,  -6,   54,   73,   -69,
    -72,  -85,  -46,  76,   39,   -79,  -46,  77,   126,  -83,  -115, -38,  25,   -27,  -64,  106,  49,   -109, 92,
    -94,  123,  -62,  -117, 48,   28,   -44,  73,   1,    -31,  103,  -74,  96,   93,   79,   -3,   61,   -79,  75,
    125,  -127, -56,  98,   -58,  76,   -41,  -20,  69,   -85,  -45,  117,  -34,  28,   118,  -87,  -66,  -45,  -22,
    96,   -22,  -105, -20,  99,   92,   -80,  18,   82,   -95,  26,   -61,  33,   -75,  -5,   -95,  66,   119,  -109,
    -57,  91,   71,   45,   -74,  53,   52,   93,   -106, 17,   112,  -32,  -45,  6,    -78,  -3,   112,  -3,   -96,
    -122, -16,  107,  -117, -9,   51,   29,   61,   -70,  18,   50,   -80,  -49,  -85,  55,   22,   -119, -6,   75,
    88,   -117, -91,  -3,   -11,  -111, -105, 92,   9,    90,   -94,  80,   37,   61,   -92,  24,   15,   4,    -48,
    47,   -35,  -14,  -41,  49,   -118, 56,   -15,  76,   100,  -53,  107,  -118, -74,  -32,  -90,  77,   39,   118,
    37,   13,   -18,  109,  62,   -112, -69,  -90,  -94,  -55,  113,  -36,  5,    10,   8,    -58,  -79,  66,   -67,
    -6,   120,  -80,  -128, -20,  105,  -83,  26,   108,  37,   -78,  112,  63,   -80,  104,  -90,  125,  59,   -57,
    64,   -126, -93,  -52,  46,   100,  -55,  -25,  67,   90,   11,   110,  73,   64,   63,   -19,  115,  -1,   38,
    25,   -65,  -82,  15,   97,   119,  -73,  -46,  50,   12,   36,   63,   60,   -52,  -35,  -53,  44,   95,   5,
    77,   52,   98,   36,   56,   -124, -25,  25,   123,  -79,  94,   -45,  -101, 87,   78,   -30,  -61,  -15,  82,
    -109, -118, -3,   40,   -102, -26,  61,   4,    -39,  -102, 72,   80,   7,    113,  101,  -79,  -77,  -36,  -13,
    109,  94,   -112, 24,   22,   -7,   3,    -80,  -31,  -8,   -124, 110,  -107, -76,  3,    113,  4,    -7,   108,
    47,   35,   110,  -116, 34,   -53,  -111, 33,   82,   -55,  -45,  -55,  55,   -116, -52,  -86,  -47,  -105, -63,
    72,   -53,  119,  -1,   -84,  31,   67,   -51,  115,  -2,   111,  -37,  25,   112,  -126, -53,  49,   -97,  -119,
    92,   -111, 74,   -17,  -31,  126,  101,  -16,  -16,  -73,  82,   -45,  -70,  105,  -47};
const int32_t conv_1_output_mult[] = {1262151703, 1094522185};
const int32_t conv_1_output_shift[] = {-8, -8};
const int8_t conv_1_weights[] = {-127, -37, 74, 1, -85, 102, -110, 65, -77, 127, 110, 124, 17, -10, 110, -110};

const int64_t conv_2_biases[] = {1881123, 1799323, 4169011, 55291};
const int16_t conv_2_input[] = {
    -9997,  -26684, -22625, 9860,   -7119,  3946,   5699,   -25198, -24833, -15555, 3079,   -27173, -21358, 3917,
    -1696,  14479,  10971,  21255,  -22335, -11726, 13600,  -17634, 1256,   11445,  -4672,  27058,  16084,  21145,
    -20474, -10845, 24412,  -28996, -19903, 15489,  17206,  20096,  29753,  5103,   -15117, -16354, -4922,  -21126,
    -18776, -25143, 2266,   -10870, -17366, -31253, -31491, 16635,  -3999,  -12444, -5915,  31962,  -27654, 3354,
    30634,  14862,  -18714, 18441,  -25732, 19837,  -30463, -8374,  25331,  -22389, -18338, 17860,  -17223, 14626,
    -29728, 15063,  7026,   22518,  2993,   -27498, 7522,   -4529,  -17694, 23098,  -2758,  1547,   -20666, -9655,
    17218,  27521,  19508,  12593,  6328,   -14431, 5098,   5220,   -8352,  -16552, 7970,   17228,  22360,  30392,
    27135,  6543,   -17362, -24786, -28911, 3713,   14363,  3559,   -15191, -27406, -23409, -4915,  3827,   32694,
    12319,  3250,   -11814, 20081,  31788,  -18456, 11232,  -27035, 23199,  -25357, 1068,   -4752,  -27389, 18750,
    18904,  -26189, 9888,   4823,   -12577, -24432, -15987, -26733, 27168,  22207,  -20325, -24075, 21000,  -6149,
    29076,  2872,   -19297, -8430,  29042,  4529,   -2422,  4202,   -13338, 13571,  31829,  -16031, 26367,  30145,
    -10791, -22846, -2650,  -15889, -22542, -8871,  -20122, -20216, -7519,  -9263,  -2299,  -20099, 17531,  7557};
const int32_t conv_2_output_mult[] = {1566522842, 1222196941, 1511961819, 1495244605};
const int32_t conv_2_output_shift[] = {-9, -9, -9, -9};
const int8_t conv_2_weights[] = {
    2,   -66, -95,  -127, 123,  -110, 104,  44,   -28, -123, -45,  -109, -67, -51,  6,   -127, 117, -122, -86, 105,
    13,  -67, 125,  -42,  -121, -110, 84,   -112, -90, -8,   -81,  -127, 71,  -116, 47,  82,   -40, 22,   78,  93,
    -43, -27, -117, -27,  -101, 95,   -46,  127,  -64, -100, -104, -14,  16,  -58,  91,  127,  2,   -101, 94,  70,
    -66, 127, 55,   28,   76,   123,  -116, -102, 95,  -18,  7,    107,  45,  -107, -60, -5,   -28, 35,   -96, -83,
    -99, 127, 34,   32,   -61,  103,  25,   82,   -78, 81,   29,   119,  -29, 73,   126, -11};

const int64_t conv_3_biases[] = {3253270, -379387, -3473006, 2667560};
const int16_t conv_3_input[] = {
    -14335, -13563, -1151,  4896,   3487,   14760,  23587,  3009,   -19250, -26250, 24043,  -31488, -12301, 3383,
    18632,  12738,  7495,   18532,  -322,   26808,  -13048, 28209,  -19470, -25814, -30794, 21600,  4091,   -10746,
    27164,  661,    26525,  18515,  -854,   -5794,  25511,  -24450, -21260, 32150,  20751,  15592,  -27913, 6226,
    21824,  5724,   -8939,  -24434, -5088,  -25471, 5492,   -19950, -9028,  -18054, -23405, -8994,  9474,   17276,
    -23781, -23713, -25161, 11316,  20898,  3354,   -26968, -24528, 3759,   -19703, 6561,   -3430,  -12289, 3328,
    10127,  18563,  19987,  32486,  28335,  16569,  2695,   17856,  11091,  17305,  4179,   -17479, 24304,  7209,
    -30083, 19196,  8363,   6884,   -22388, 10299,  23126,  6868,   7406,   14759,  -29226, 11115,  27897,  -7151,
    22816,  -25750, -944,   20681,  31088,  31874,  -29860, 575,    -23018, -28047, -28317, 25030,  -31083, 21647,
    3507,   -15077, 548,    -32374, 31463,  -17201, -2610,  13439,  26054,  -11237, 17577,  13063,  29033,  5694,
    -26938, -4761,  2538,   -30921, 3324,   -10102, 25910,  -28792, 26924,  1547,   -3017,  -2242,  -12661, -15283,
    13874,  -24357, 1959,   27224,  -6901,  -23878, -19841, -29555, -14918, -10397, 7862,   -23046, -22905, 22477,
    -17349, -28765, 1619,   -14410, -27190, 32254,  -21091, 21515,  -14513, 29553,  29341,  -17152, -32349, 29466};
const int32_t conv_3_output_mult[] = {1507306161, 1465701884, 1457399652, 1519318956};
const int32_t conv_3_output_shift[] = {-8, -8, -8, -8};
const int8_t conv_3_weights[] = {
    82,  70,   -127, -60,  81,  37,   -123, -5,   71,  -80, 37,  86,  31,  18,  -70,  -37,  -44,  -41,  -15,  -14,
    124, 4,    -21,  -43,  95,  2,    -23,  32,   0,   -36, 15,  101, -43, -11, 51,   -40,  -127, -116, -64,  82,
    -59, 43,   99,   -104, -51, -119, 118,  -105, 40,  88,  71,  94,  127, -91, -95,  -36,  -65,  123,  -115, -91,
    -60, 124,  106,  52,   101, -51,  114,  38,   78,  -44, -91, -76, 88,  44,  3,    -104, 87,   -59,  87,   -61,
    59,  -127, -46,  -26,  58,  -87,  -48,  127,  -82, -10, 127, -34, -16, -50, -126, -26};

const int64_t conv_4_biases[] = {-21812718, 1073741952};
const int16_t conv_4_input[] = {82, 66, -30, -105, 95, 74, 124, 93, -25, 93, 57, -32};
const int32_t conv_4_output_mult[] = {1451496532, 2138792311};
const int32_t conv_4_output_shift[] = {-14, -15};
const int8_t conv_4_weights[] = {-23, 101, 3, -55, -120, 127, -120, 6, 103, 103, -95, 60, 102, 9, -60, -90};