#pragma once
/* GLOBAL SETTING: use single/double precision floats */
#define USE_SINGLE_PRECISION /* or "USE_DOUBLE_PRECISION" */

#include <math.h>
#include <float.h>

/* Define unified macros for single/double precision math functions */
#ifdef USE_DOUBLE_PRECISION
typedef double REAL;
#define Exp    exp
#define Sqrt   sqrt
#define Sin    sin
#define Cos    cos
#define ArcCos acos
#define Pow    pow
#define Fmod   fmod
#define Modf   modf
#define Fabs   fabs
#define Round  round
#define Log    log
#else /* USE_SINGLE_PRECISION */
typedef float REAL;
#define Exp    expf
#define Sqrt   sqrtf
#define Sin    sinf
#define Cos    cosf
#define ArcCos acosf
#define Pow    powf
#define Fmod   fmodf
#define Modf   modff
#define Fabs   fabsf
#define Round  roundf
#define Log    logf
#endif

#define Ln Log

#ifdef USE_DOUBLE_PRECISION
#define REAL_MIN DBL_MIN
#define REAL_MAX DBL_MAX
#define REAL_NOISE_VALUE 0.0000001
#else
#define REAL_MIN FLT_MIN
#define REAL_MAX FLT_MAX
#define REAL_NOISE_VALUE 0.00001f
#endif

#define ZERO    ((REAL)(0.0))  /* cast constants to proper type (double or float) */
#define QUARTER ((REAL)(0.25))
#define HALF    ((REAL)(0.5))
#define ONE     ((REAL)(1.0))
#define TWO     ((REAL)(2.0))

#define HALFPI  ((REAL)(1.57079632679))
#define PI      ((REAL)(3.14159265359))
#define TWOPI   ((REAL)(6.28318530718))
#define PI_LP   ((REAL)(3.14)) /* low-precision PI */


#define MIN(a,b) (( (a) < (b) ) ? (a) : (b) )
#define MAX(a,b) (( (a) > (b) ) ? (a) : (b) )

enum VEC_Elements { X, Y, Z, S };

/* vector with two elements */
struct VEC2 {
    union {
        REAL e[2];      /* packed components */
        struct {
            REAL x, y;  /* unpacked components */
        };
    };

    VEC2() { e[X] = ZERO; e[Y] = ZERO; }
    VEC2(REAL x, REAL y) { e[X] = x; e[Y] = y; }
};

/* vector with three elements */
struct VEC3 {
    union {
        REAL e[3];
        struct {
            REAL x, y, z;
        };
    };

    VEC3() { e[X] = ZERO; e[Y] = ZERO; e[Z] = ZERO; }
    VEC3(REAL x, REAL y, REAL z) { e[X] = x; e[Y] = y; e[Z] = z; }
};

/* q = xi + yj + zk + s */
struct QUATERNION {
    union {
        REAL e[4];
        struct {
            REAL x, y, z, s;
        };
    };

    QUATERNION() { e[X] = ZERO; e[Y] = ZERO; e[Z] = ZERO; e[S] = ZERO; }
    QUATERNION(REAL x, REAL y, REAL z, REAL s) {
        e[X] = x; e[Y] = y; e[Z] = z; e[S] = s;
    }
    QUATERNION(VEC3 v, REAL s) {
        e[X] = v.e[X]; e[Y] = v.e[Y]; e[Z] = v.e[Z]; e[S] = s;
    }
};

enum MAT3x3_Elements {
    XX, XY, XZ,
    YX, YY, YZ,
    ZX, ZY, ZZ
};

/* a 3x3 matrix */
struct MAT3x3 {
    union {
        REAL e[9];
        struct {
            REAL xx, xy, xz;
            REAL yx, yy, yz;
            REAL zx, zy, zz;
        };
    };

    MAT3x3() { for (int i = 0; i < 9; i++) e[i] = ZERO; }
    MAT3x3(
        REAL xx, REAL xy, REAL xz,
        REAL yx, REAL yy, REAL yz,
        REAL zx, REAL zy, REAL zz) {
        e[0] = xx, e[1] = xy, e[2] = xz;
        e[3] = yx, e[4] = yy, e[5] = yz;
        e[6] = zx, e[7] = zy, e[8] = zz;
    }
};


/* basic mathematical operations for vectors, quaternions and matrices */
VEC3 operator + (REAL a, VEC3 v);
VEC3 operator - (REAL a, VEC3 v);
VEC3 operator * (REAL a, VEC3 v);
VEC3 operator / (REAL a, VEC3 v);
VEC3 operator + (VEC3 v, REAL a);
VEC3 operator - (VEC3 v, REAL a);
VEC3 operator * (VEC3 v, REAL a);
VEC3 operator / (VEC3 v, REAL a);

VEC3 operator + (VEC3 a, VEC3 b);
VEC3 operator - (VEC3 a, VEC3 b);
VEC3 operator - (VEC3 v); /* negate */
REAL dot(VEC3 a, VEC3 b);
VEC3 cross(VEC3 a, VEC3 b);
REAL len(VEC3 a);
REAL lensq(VEC3 a);
VEC3 normalize(VEC3 v);

QUATERNION operator + (QUATERNION q1, QUATERNION q2);
QUATERNION operator - (QUATERNION q1, QUATERNION q2);
QUATERNION operator * (QUATERNION q1, QUATERNION q2);    /* Hamilton product of two quaternions */
QUATERNION operator * (QUATERNION q, REAL a);
QUATERNION operator * (REAL a, QUATERNION q);
QUATERNION operator / (QUATERNION q1, QUATERNION q2);
QUATERNION operator / (QUATERNION q, REAL a);
QUATERNION operator / (REAL a, QUATERNION q);

QUATERNION conj(QUATERNION q);                           /* conjugate */
QUATERNION inv(QUATERNION q);                            /* inverse */
REAL norm(QUATERNION q);                                 /* returns the norm ||q|| of a quaternion */
REAL normsq(QUATERNION q);                               /* returns squared norm  */
QUATERNION normalize(QUATERNION q);                      /* normalizing a quaternion */
REAL dot(QUATERNION q1, QUATERNION q2);
QUATERNION slerp(QUATERNION q1, QUATERNION q2, REAL t);  /* spherical linear interpolation */

MAT3x3 QuaternionToMatrix(QUATERNION q);
QUATERNION MatrixToQuaternion(MAT3x3 m);

QUATERNION VectorMulQuaternion(VEC3 v, QUATERNION q);

MAT3x3 operator * (MAT3x3 a, MAT3x3 b);
VEC3 operator * (MAT3x3 a, VEC3 b);
MAT3x3 transpose(MAT3x3 a);
MAT3x3 inv(MAT3x3 a);

