#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>

#define ROUND_UP(f) ((int) ((f) >= 0.0 ? (f) + 0.5F : (f) - 0.5F))
#define UINT8 unsigned char
#define INT32 int
#define INT_MAX 0x7fffffff

/* pixel types */
#define IMAGING_TYPE_UINT8 0
#define IMAGING_TYPE_INT32 1
#define IMAGING_TYPE_FLOAT32 2
#define IMAGING_TYPE_SPECIAL 3 /* check mode for details */

#define IMAGING_MODE_LENGTH 6+1 /* Band names ("1", "L", "P", "RGB", "RGBA", "CMYK", "YCbCr", "BGR;xy") */


/* standard transforms */
#define IMAGING_TRANSFORM_AFFINE 0
#define IMAGING_TRANSFORM_PERSPECTIVE 2
#define IMAGING_TRANSFORM_QUAD 3


/* standard filters */
#define IMAGING_TRANSFORM_NEAREST 0
#define IMAGING_TRANSFORM_BOX 4
#define IMAGING_TRANSFORM_BILINEAR 2
#define IMAGING_TRANSFORM_HAMMING 5
#define IMAGING_TRANSFORM_BICUBIC 3
#define IMAGING_TRANSFORM_LANCZOS 1


typedef void (*ResampleFunction)(unsigned char *pOut, unsigned char *pIn, int offset,
                               int ksize, int *bounds, 
                               double *prekk, 
                               int inpWd, int inpHt, int inpStride, 
                               int outWd, int outHt, int outStride, 
                               int imType, int channels);

struct filter {
    double (*filter)(double x);
    double support;
};

static inline double box_filter(double x)
{
    if (x >= -0.5 && x < 0.5)
        return 1.0;
    return 0.0;
}

static inline double bilinear_filter(double x)
{
    if (x < 0.0)
        x = -x;
    if (x < 1.0)
        return 1.0-x;
    return 0.0;
}

static inline double hamming_filter(double x)
{
    if (x < 0.0)
        x = -x;
    if (x == 0.0)
        return 1.0;
    if (x >= 1.0)
        return 0.0;
    x = x * M_PI;
    return sin(x) / x * (0.54f + 0.46f * cos(x));
}

static inline double bicubic_filter(double x)
{
    /* https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm */
#define a -0.5
    if (x < 0.0)
        x = -x;
    if (x < 1.0)
        return ((a + 2.0) * x - (a + 3.0)) * x*x + 1;
    if (x < 2.0)
        return (((x - 5) * x + 8) * x - 4) * a;
    return 0.0;
#undef a
}

static inline double sinc_filter(double x)
{
    if (x == 0.0)
        return 1.0;
    x = x * M_PI;
    return sin(x) / x;
}

static inline double lanczos_filter(double x)
{
    /* truncated sinc */
    if (-3.0 <= x && x < 3.0)
        return sinc_filter(x) * sinc_filter(x/3);
    return 0.0;
}

static struct filter BOX = { box_filter, 0.5 };
static struct filter BILINEAR = { bilinear_filter, 1.0 };
static struct filter HAMMING = { hamming_filter, 1.0 };
static struct filter BICUBIC = { bicubic_filter, 2.0 };
static struct filter LANCZOS = { lanczos_filter, 3.0 };


/* 8 bits for result. Filter can have negative areas.
   In one cases the sum of the coefficients will be negative,
   in the other it will be more than 1.0. That is why we need
   two extra bits for overflow and int type. */
#define PRECISION_BITS (32 - 8 - 2)


/* Handles values form -640 to 639. */
UINT8 _clip8_lookups[1280] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
    96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
    112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
    128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
    144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
    160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
    176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
    192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
    208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
    224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
    240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
};

UINT8 *clip8_lookups = &_clip8_lookups[640];

static inline UINT8 clip8(int in)
{
    //printf("%d\n", in);
    return clip8_lookups[in >> PRECISION_BITS];
}


__device__ UINT8 _clip8_lookups_cuda[1280] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
    96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
    112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
    128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
    144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
    160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
    176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
    192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
    208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
    224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
    240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
};

__device__ UINT8 *clip8_lookups_cuda = &_clip8_lookups_cuda[640];

__device__ static inline UINT8 clip8_cuda(int in)
{
    //printf("%d\n", in);
    return clip8_lookups_cuda[in >> PRECISION_BITS];
}



int
precompute_coeffs(int inSize, float in0, float in1, int outSize,
                  struct filter *filterp, int **boundsp, double **kkp) {
    double support, scale, filterscale;
    double center, ww, ss;
    int xx, x, ksize, xmin, xmax;
    int *bounds;
    double *kk, *k;
    
    printf("precompute_coeffs\n");
    /* prepare for horizontal stretch */
    printf("outsize = %d :: in1 = %f :: in0 = %f \n", outSize, in1, in0);
    filterscale = scale = (double) (in1 - in0) / outSize;
    if (filterscale < 1.0) {
        filterscale = 1.0;
    }

    /* determine support size (length of resampling filter) */
    support = filterp->support * filterscale;

    /* maximum number of coeffs */
    ksize = (int) ceil(support) * 2 + 1;

    printf("ksize = %d\n", ksize);
    printf("support = %f\n", ceil(support));
    printf("filterscale = %f\n", filterscale);
    // check for overflow
    if (outSize > INT_MAX / (ksize * sizeof(double))) {
        return 0;
    }

    /* coefficient buffer */
    /* malloc check ok, overflow checked above */
    kk = (double *) malloc(outSize * ksize * sizeof(double));
    if ( ! kk) {
        return 0;
    }

    /* malloc check ok, ksize*sizeof(double) > 2*sizeof(int) */
    bounds = (int *) malloc(outSize * 2 * sizeof(int));
    if ( ! bounds) {
        free(kk);
        return 0;
    }

    for (xx = 0; xx < outSize; xx++) {
        center = in0 + (xx + 0.5) * scale;
        // printf("center = %f\n", center);
        ww = 0.0;
        ss = 1.0 / filterscale;
        // Round the value
        xmin = (int) (center - support + 0.5); // -1.8 -> -2 ,,  3.7 - 5 + 0.5 -> -0.8 . -1
        // printf("support = %f\n", support);
        // printf("xmin = %d\n", xmin);
        // TODO
        // if (xmin < 0)
        //     xmin = 0;
        // Round the value
        xmax = (int) (center + support + 0.5); // 3.2 + 5 + 0.5 -> 8 . 3.7 + 5 + 0.5  -> 9.2  9
        // printf("xmax = %d\n", xmax);
        
        // TODO
        // if (xmax > inSize)
        //     xmax = inSize;
        
        xmax -= xmin;  // 10, 10 0..
        
        k = &kk[xx * ksize];
        for (x = 0; x < xmax; x++) {
            if(x + xmin < 0 || x + xmin >= (int)(in1 - in0) ){
                k[x] = 0;
                continue;
            }
            
            double w = filterp->filter((x + xmin - center + 0.5) * ss);
               
            k[x] = w;
            ww += w;
        }
        for (x = 0; x < xmax; x++) {
            if (ww != 0.0)
                k[x] /= ww;
        }
        // Remaining values should stay empty if they are used despite of xmax.
        for (; x < ksize; x++) {
            k[x] = 0;
        }
        // printf("xmin = %d :: xmax = %d\n", xmin, xmax);
        // for (x = 0; x < xmax; x++)
        //     printf("%f ", k[x]);
        // printf("\n");
        bounds[xx * 2 + 0] = xmin;
        bounds[xx * 2 + 1] = xmax;
    }
    *boundsp = bounds;
    *kkp = kk;
    return ksize;
}


void
normalize_coeffs_8bpc(int outSize, int ksize, double *prekk)
{
    int x;
    INT32 *kk;

    // use the same buffer for normalized coefficients
    kk = (INT32 *) prekk;

    for (x = 0; x < outSize * ksize; x++) {
        if (prekk[x] < 0) {

            kk[x] = (int) (-0.5 + prekk[x] * (1 << PRECISION_BITS));
       
        } else {

            kk[x] = (int) (0.5 + prekk[x] * (1 << PRECISION_BITS));
        }
    }
}
__global__ void verticalKernel( unsigned char *pOut, unsigned char *pIn, 
                                int *bounds,  int* kk, 
                                int dst_width, int dst_height, 
                                int ksize, int inpStride, int outStride,
                                int inpHt,
                                int channels){        
    int dx = threadIdx.x;
    int dy = threadIdx.y;
    int ss0;

    int *k = &kk[dy * ksize];
    int ymin = bounds[dy * 2 + 0];
    int ymax = bounds[dy * 2 + 1];
    for (int c = 0; c < channels; c++){ // channel - 1
        ss0 = 1 << (PRECISION_BITS -1);
        for (int y = 0; y < ymax; y++)
        {   
            if(y + ymin >= 0 && y + ymin < inpHt){
                // if(xx == 12)
                    // printf("%d ", ((UINT8) pIn[(y + ymin)*inpStride + channels*xx + c]) * k[y]);
                ss0 += ((UINT8) pIn[(y + ymin)*inpStride + channels*dx + c]) * k[y];
            }
        }
        // if(xx == 12)
            // printf("\n");                  
        pOut[dy*outStride + channels*dx + c] = clip8_cuda(ss0);
    }
    
}


void
ImagingResampleVertical_8bpc(unsigned char *pOut, unsigned char *pIn, int offset,
                             int ksize, int *bounds, 
                             double *prekk, int inpWd, int inpHt, int inpStride, 
                             int outWd, int outHt, int outStride, 
                             int imType, int channels)
{
    int ss0, ss1, ss2, ss3;
    int xx, yy, y, ymin, ymax;
    int *k, *kk;
    int c;

    // use the same buffer for normalized coefficients
    kk = (INT32 *) prekk;
    normalize_coeffs_8bpc(outHt, ksize, prekk);
    
    printf("\n");
    printf("CALL VERTICAL \n");
    printf("%d %d\n", outHt, outWd);
    printf("input Height : %d \n", inpHt);
    printf("ksize : %d \n", ksize);
    printf("outStride : %d \n", outStride);
    printf("inpStride : %d \n", inpStride);
    
    dim3 dimBlock(outWd,outHt);
    
    int outSize = 32;
    int size = 32;
    int *bounds_gpu;
    int *kk_gpu;

    unsigned char *pOut_gpu;
    unsigned char *pIn_gpu;

    cudaMalloc((void**)&bounds_gpu, outSize * 2 * sizeof(int));
    cudaMalloc((void**)&kk_gpu, outSize * ksize * sizeof(float));
    
    cudaMalloc((void**)&pIn_gpu, 32 * inpHt * 1 * 4  );

    cudaMemcpy( bounds_gpu , bounds ,outSize * 2 * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy( kk_gpu , kk ,outSize * ksize * sizeof(int),cudaMemcpyHostToDevice);

    // TODO xsize is 32
    cudaMemcpy(pIn_gpu, pIn ,32 * inpHt * 1 * 4 ,cudaMemcpyHostToDevice);

    // Pout init
    cudaMalloc((void**)&pOut_gpu, size * size );
    
    printf("CALL VERTICAL cuda\n");
    verticalKernel<<<1, dimBlock>>>(pOut_gpu, pIn_gpu, bounds_gpu, kk_gpu, outWd, outHt, ksize, inpStride, outStride, inpHt, channels);
    cudaDeviceSynchronize();
    cudaMemcpy(pOut,pOut_gpu,size*size ,cudaMemcpyDeviceToHost);
    for (yy = 0; yy < outHt; yy++) {
        for (xx = 0; xx < outWd; xx++) {
            for (c = 0; c < channels; c++){ // channel - 1
                    printf("%d ", (pOut[yy*outStride + channels*xx + c]));
                }
            }
    }

    printf("\n");    

    // CPU call    
    // for (yy = 0; yy < outHt; yy++) {
    //     k = &kk[yy * ksize];
    //     ymin = bounds[yy * 2 + 0];
    //     ymax = bounds[yy * 2 + 1];
    //     for (xx = 0; xx < outWd; xx++) {
    //         for (c = 0; c < channels; c++){ // channel - 1
    //                 ss0 = 1 << (PRECISION_BITS -1);
    //                 for (y = 0; y < ymax; y++)
    //                 {   
    //                     if(y + ymin >= 0 && y + ymin < inpHt){
    //                         // if(xx == 12)
    //                             // printf("%d ", ((UINT8) pIn[(y + ymin)*inpStride + channels*xx + c]) * k[y]);
    //                         ss0 += ((UINT8) pIn[(y + ymin)*inpStride + channels*xx + c]) * k[y];
    //                     }
    //                 }
    //                 // if(xx == 12)
    //                     // printf("\n");                  
    //                 pOut[yy*outStride + channels*xx + c] = clip8(ss0);
    //             }
    //         }
    // }
    
    // printf("\n");    
    // for (yy = 0; yy < outHt; yy++) {
    //     k = &kk[yy * ksize];
    //     ymin = bounds[yy * 2 + 0];
    //     ymax = bounds[yy * 2 + 1];
    //     for (xx = 0; xx < outWd; xx++) {
    //         for (c = 0; c < channels; c++){ // channel - 1
    //                 printf("%d ", (pOut[yy*outStride + channels*xx + c]));
    //             }
    //         }
    // }
    
}


__global__ void horizontalKernel(unsigned char *pOut, unsigned char *pIn, 
                                int *bounds,  int* kk, 
                                int dst_width, int dst_height, 
                                int ksize, 
                                int inpStride, int outStride,
                                int inpWd,
                                int channels){
    
    int dx = threadIdx.x;
    int dy = threadIdx.y;
    int ss0;

    
    int xmin = bounds[dx * 2 + 0];
    int xmax = bounds[dx * 2 + 1];
    int *k = &kk[dx * ksize];
    for (int c = 0; c < channels; c++){
        ss0 = 1 << (PRECISION_BITS -1);
        for (int x = 0; x < xmax ; x++)
        {
            if(x + xmin >= 0 && x + xmin < inpWd){
                ss0 += ((UINT8) pIn[inpStride*(dy + 0) + channels*(x + xmin) + c]) * k[x];
            }
        }
        pOut[dy*outStride + channels*dx + c] = clip8_cuda(ss0);
    }

} 

void
ImagingResampleHorizontal_8bpc(unsigned char *pOut, unsigned char *pIn, int offset,
                               int ksize, int *bounds, double *prekk, int inpWd, int inpHt, int inpStride, int outWd, int outHt, int outStride, int imType, int channels)
{
    int ss0, ss1, ss2, ss3;
    int xx, yy, x, xmin, xmax;
    int *k, *kk;
    int c;
    kk = (int *) prekk;
    normalize_coeffs_8bpc(outWd, ksize, prekk);
    
    printf("\n");
    printf("CALL HORIZONTAL \n");
    printf("%d %d\n", outHt, outWd);
    printf("input Width : %d \n", inpWd);
    printf("input Height : %d \n", inpHt);
    
    printf("ksize : %d \n", ksize);
    printf("outStride : %d \n", outStride);
    printf("inpStride : %d \n", inpStride);
    
    dim3 dimBlock(outWd,outHt);
    
    int outSize = 32;
    int xsize = 32;
    int *bounds_gpu;
    int *kk_gpu;

    unsigned char *pOut_gpu;
    unsigned char *pIn_gpu;

    cudaMalloc((void**)&bounds_gpu, outSize * 2 * sizeof(int));
    cudaMalloc((void**)&kk_gpu, outSize * ksize * sizeof(float));
    
    cudaMalloc((void**)&pIn_gpu, inpWd * inpHt * 1);

    cudaMemcpy( bounds_gpu , bounds ,outSize * 2 * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy( kk_gpu , kk ,outSize * ksize * sizeof(int),cudaMemcpyHostToDevice);

    
    cudaMemcpy(pIn_gpu, pIn , inpWd * inpHt * 1 ,cudaMemcpyHostToDevice);

    // TODO xsize is 32 
    cudaMalloc((void**)&pOut_gpu,  32 * inpHt * 1 * 4 );
    
    // printf("CALL HORIZONTAL cuda\n");
    // horizontalKernel<<<1, dimBlock>>>(pOut_gpu, pIn_gpu, bounds_gpu, kk_gpu, outWd, outHt, ksize, inpStride, outStride, inpWd, channels);
    // cudaDeviceSynchronize();
    // cudaMemcpy(pOut,pOut_gpu, 32 * inpHt * 1 * 4 , cudaMemcpyDeviceToHost);


    for (yy = 0; yy < outHt; yy++) {
        for (xx = 0; xx < outWd; xx++) {
            xmin = bounds[xx * 2 + 0];
            xmax = bounds[xx * 2 + 1];
            k = &kk[xx * ksize];
            for (c = 0; c < channels; c++){
                ss0 = 1 << (PRECISION_BITS -1);
                for (x = 0; x < xmax ; x++)
                {
                    if(x + xmin >= 0 && x + xmin < inpWd){
                        ss0 += ((UINT8) pIn[inpStride*(yy + 0) + channels*(x + xmin) + c]) * k[x];
                    }
                }
                pOut[yy*outStride + channels*xx + c] = clip8(ss0);
            }
        }
    }
    for (yy = 0; yy < outHt; yy++) {
        for (xx = 0; xx < outWd; xx++) {
            for (c = 0; c < channels; c++){
                printf("%d ", pOut[yy*outStride + channels*xx + c]);
            }
        }
    }


}

int ImagingResampleInner(   unsigned char *pIn, unsigned char *pOut,
                            int inpWd, int inpHt, int inpStride, int xsize, int ysize, int outStride,
                            struct filter *filterp, float box[4],
                     ResampleFunction ResampleHorizontal,
                     ResampleFunction ResampleVertical, 
                     int imType, int channels)
{
    unsigned char *pImTemp = NULL;

    int i, need_horizontal, need_vertical;
    int ybox_first, ybox_last;
    int ksize_horiz, ksize_vert;
    int *bounds_horiz, *bounds_vert;
    double *kk_horiz, *kk_vert;

    need_horizontal = xsize != inpWd || box[0] || box[2] != xsize;
    need_vertical = ysize != inpHt || box[1] || box[3] != ysize;
    

    printf("xsize: %d \n", xsize);
    printf("inpWd: %d \n", inpWd);
    printf("Need Horizontal: %d \n", need_horizontal);
    printf("Need Vertical: %d \n", need_vertical);

    
    ksize_horiz = precompute_coeffs(inpWd, box[0], box[2], xsize,
                                    filterp, &bounds_horiz, &kk_horiz);
    if ( ! ksize_horiz) {
        return -1;
    }

    ksize_vert = precompute_coeffs(inpHt, box[1], box[3], ysize,
                                   filterp, &bounds_vert, &kk_vert);

    if ( ! ksize_vert) {
        free(bounds_horiz);
        free(kk_horiz);
        free(bounds_vert);
        free(kk_vert);
        return -1;
    }

    // First used row in the source image
    ybox_first = bounds_vert[0];
    // Last used row in the source image
    ybox_last = bounds_vert[ysize*2 - 2] + bounds_vert[ysize*2 - 1];


    /* two-pass resize, horizontal pass */
    if (1) {
        int stride;
        // Shift bounds for vertical pass
        // TODO
        // for (i = 0; i < ysize; i++) {
        //     bounds_vert[i * 2] -= ybox_first;
        // }
    if (1)
        pImTemp = (unsigned char *)malloc(xsize * inpHt * channels * 4);
    else pImTemp = pOut;
    
    stride = 1?xsize:outStride;

    if (pImTemp) {
        
        ResampleHorizontal(pImTemp, pIn, ybox_first,
                            ksize_horiz, 
                            bounds_horiz, 
                            kk_horiz, 
                            inpWd, inpHt, inpStride, 
                            xsize, inpHt, stride, 
                            imType, channels
                            );
        
        //  ResampleHorizontal<<<1,1>>>(pImTemp, pIn, ybox_first,
        //                     ksize_horiz, 
        //                     bounds_horiz, 
        //                     kk_horiz, 
        //                     inpWd, inpHt, inpStride, 
        //                     xsize, inpHt, stride, 
        //                     imType, channels,
        //                     jobs);
    
    }
    free(bounds_horiz);
    free(kk_horiz);
    if ( ! pImTemp) {
        free(bounds_vert);
        free(kk_vert);
        return -1;
    }
    //imOut = imIn = imTemp;
    } else {
        // Free in any case
        free(bounds_horiz);
        free(kk_horiz);
    }

    /* vertical pass */
    if (1) {
        unsigned char *pIn2;
        int wd;
        int stride = 1?xsize:inpStride;
        pIn2 = 1?pImTemp:pIn;

        if (1) {
            /* imIn can be the original image or horizontally resampled one */
            ResampleVertical(  pOut, pIn2, 0,
                                ksize_vert, 
                                bounds_vert, 
                                kk_vert, 
                                xsize, inpHt, stride, xsize, ysize, 
                                outStride, imType, channels);

            // ResampleVertical<<<1,1>>>(  pOut, pIn2, 0,
            //                     ksize_vert, 
            //                     bounds_vert, 
            //                     kk_vert, 
            //                     xsize, inpHt, stride, xsize, ysize, 
            //                     outStride, imType, channels,
            //                     jobs);

        }

        /* it's safe to call ImagingDelete with empty value
           if previous step was not performed. */
        free(pImTemp);
        free(bounds_vert);
        free(kk_vert);
        return 0;
        /*if ( ! imOut) {
            return NULL;
        }*/
    } else {
        // Free in any case
        free(bounds_vert);
        free(kk_vert);
    }

    /* none of the previous steps are performed, copying */
    if ( ! (need_horizontal || need_vertical)) {
    //printf("memcpy only\n");
        //memcpy(pOut, pIn, xsize*ysize*((imType == IMAGING_TYPE_UINT8)?1:4)*channels);
    int i;
    for (i = 0; i < ysize; i++)
        memcpy(pOut + i*outStride, pIn + i*inpStride, xsize*channels*((imType == IMAGING_TYPE_UINT8)?1:4));
    }

    return 0;
}


int ImagingResample(unsigned char *pIn, unsigned char *pOut, int inpWd, int inpHt, int inpStride, int xsize, int ysize, int outStride, int filter, float box[4], int imType, int channels)
{
    struct filter *filterp;
    // ResampleFunctionCuda ResampleHorizontal;
    // ResampleFunctionCuda ResampleVertical;
    ResampleFunction ResampleHorizontal;
    ResampleFunction ResampleVertical;


    
        switch(imType) {
            case IMAGING_TYPE_UINT8:
                ResampleHorizontal = ImagingResampleHorizontal_8bpc; //ImagingResampleHorizontal_8bpc_kernel;  //ImagingResampleHorizontal_8bpc;
                ResampleVertical = ImagingResampleVertical_8bpc; // ImagingResampleVertical_8bpc_kernel; // ImagingResampleVertical_8bpc;
                break;
            case IMAGING_TYPE_INT32:
            case IMAGING_TYPE_FLOAT32:
                //ResampleHorizontal = ImagingResampleHorizontal_32bpc;
                //ResampleVertical = ImagingResampleVertical_32bpc;
                //break;
            default:
                return -1;
        }

    /* check filter */
    switch (filter) {
    case IMAGING_TRANSFORM_BOX:
        filterp = &BOX;
        break;
    case IMAGING_TRANSFORM_BILINEAR:
        filterp = &BILINEAR;
        break;
    case IMAGING_TRANSFORM_HAMMING:
        filterp = &HAMMING;
        break;
    case IMAGING_TRANSFORM_BICUBIC:
        filterp = &BICUBIC;
        break;
    case IMAGING_TRANSFORM_LANCZOS:
        filterp = &LANCZOS;
        break;
    default:
        return -1;
    }

    return ImagingResampleInner(pIn, pOut, inpWd, inpHt, inpStride, xsize, ysize, outStride, filterp, box,
                                ResampleHorizontal, ResampleVertical, imType, channels);
}


// modified resize routine
int resizeModPIL(unsigned char *pIn, unsigned char *pOut, int inpWd, int inpHt, int inpStride, int outWd, int outHt, int outStride, int channels)
{
    //Imaging imIn;
    //Imaging imOut;

    int xsize, ysize;
    int filter = IMAGING_TRANSFORM_LANCZOS;
    float box[4] = {0, 0, 0, 0};
    int imType = IMAGING_TYPE_UINT8;
    //imIn = self->image;
    box[2] = inpWd;
    box[3] = inpHt;
    
    xsize = outWd;
    ysize = outHt;
    
    
    if (xsize < 1 || ysize < 1) {
        return -1;//ImagingError_ValueError("height and width must be > 0");
    }

    if (box[0] < 0 || box[1] < 0) {
        return -1;//ImagingError_ValueError("box offset can't be negative");
    }

    if (box[2] > inpWd || box[3] > inpHt) {
        return -1;//ImagingError_ValueError("box can't exceed original image size");
    }

    if (box[2] - box[0] < 0 || box[3] - box[1] < 0) {
        return -1;//ImagingError_ValueError("box can't be empty");
    }

    // If box's coordinates are int and box size matches requested size
    if (0)/*(box[0] - (int) box[0] == 0 && box[2] - box[0] == xsize
            && box[1] - (int) box[1] == 0 && box[3] - box[1] == ysize) */{
        //imOut = ImagingCrop(imIn, box[0], box[1], box[2], box[3]);
    }
    else if (filter == IMAGING_TRANSFORM_NEAREST) {
        double a[6];

        memset(a, 0, sizeof a);
        a[0] = (double) (box[2] - box[0]) / xsize;
        a[4] = (double) (box[3] - box[1]) / ysize;
        a[2] = box[0];
        a[5] = box[1];

        /*imOut = ImagingNewDirty(imIn->mode, xsize, ysize);

        imOut = ImagingTransform(
            imOut, imIn, IMAGING_TRANSFORM_AFFINE,
            0, 0, xsize, ysize,
            a, filter, 1);*/
    }
    else {
      printf("calling imagingresample\n");
      return ImagingResample(pIn, pOut, inpWd, inpHt, inpStride, xsize, ysize, outStride, filter, box, imType, channels);
    }

    return 0;
}
int main(int argc, char *argv[])
{
    // car1.jpg JPEG 350x174 350x174+0+0 8-bit sRGB 19.7KB 0.000u 0:00.000
    // car2.jpg[1] JPEG 572x342 572x342+0+0 8-bit sRGB 45.8KB 0.000u 0:00.000
    // car3.jpg[2] JPEG 228x174 228x174+0+0 8-bit sRGB 15.2KB 0.000u 0:00.000

    unsigned char *pIn, *pOut;
    int ret, i;
    int inpWd = 1881;
    int inpHt = 926;
    int inpStride = 1881;
    int outWd = 32;
    int outHt = 32;
    int outStride = 32;
    int nCh = 1;
    FILE *fp1;
    FILE *fp = fopen("./LM_crop.raw", "rb");
    // FILE *fp = fopen("./car1.raw", "rb");
    // FILE *fp = fopen("./car2.raw", "rb");
    // FILE *fp = fopen("./car3.raw", "rb");
    // FILE *fp = fopen("./pixel.raw", "rb");
    
    pIn = (unsigned char *)malloc(inpStride*inpHt*nCh);
    pOut = (unsigned char *)malloc(outStride*outHt*nCh);
    
    for (i = 0; i < inpHt; i++)
        fread(pIn+i*inpStride, 1, inpWd*nCh, fp);
    fclose(fp);

    ret = resizeModPIL(pIn, pOut, inpWd, inpHt, inpStride, outWd, outHt, outStride, nCh);
    printf("return status = %d\n", ret);
    fp1 = fopen("./LM_resized.raw", "wb");
    // fp1 = fopen("./car1_resized.raw", "wb");
    // fp1 = fopen("./car2_resized.raw", "wb");
    // fp1 = fopen("./car3_resized.raw", "wb");
    // fp1 = fopen("./pixel_resized.raw", "wb");
    
    
    for (i = 0; i < outHt; i++)
        fwrite(pOut + i * outStride, 1, outWd*nCh, fp1);
    fclose(fp1);
}
