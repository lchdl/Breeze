#include "basemath.h"

VEC3 operator + (REAL a, VEC3 v) { return VEC3(a + v.e[X], a + v.e[Y], a + v.e[Z]); }
VEC3 operator - (REAL a, VEC3 v) { return VEC3(a - v.e[X] ,a - v.e[Y], a - v.e[Z]); }
VEC3 operator * (REAL a, VEC3 v) { return VEC3(a * v.e[X], a * v.e[Y], a * v.e[Z]); }
VEC3 operator / (REAL a, VEC3 v) { return VEC3(a / v.e[X], a / v.e[Y], a / v.e[Z]); }
VEC3 operator + (VEC3 v, REAL a) { return VEC3(v.e[X] + a, v.e[Y] + a, v.e[Z] + a); }
VEC3 operator - (VEC3 v, REAL a) { return VEC3(v.e[X] - a, v.e[Y] - a, v.e[Z] - a); }
VEC3 operator * (VEC3 v, REAL a) { return VEC3(v.e[X] * a, v.e[Y] * a, v.e[Z] * a); }
VEC3 operator / (VEC3 v, REAL a) { return VEC3(v.e[X] / a, v.e[Y] / a, v.e[Z] / a); }

VEC3 operator + (VEC3 a, VEC3 b) { return VEC3(a.e[X] + b.e[X], a.e[Y] + b.e[Y], a.e[Z] + b.e[Z]); }
VEC3 operator - (VEC3 a, VEC3 b) { return VEC3(a.e[X] - b.e[X], a.e[Y] - b.e[Y], a.e[Z] - b.e[Z]); }
VEC3 operator-(VEC3 v) { return VEC3(-v.e[X], -v.e[Y], -v.e[Z]); }

REAL dot(VEC3 a, VEC3 b) { return a.e[X] * b.e[X] + a.e[Y] * b.e[Y] + a.e[Z] * b.e[Z]; }
VEC3 cross(VEC3 a, VEC3 b) {
    return VEC3(
        a.e[Y] * b.e[Z] - a.e[Z] * b.e[Y],
        a.e[Z] * b.e[X] - a.e[X] * b.e[Z],
        a.e[X] * b.e[Y] - a.e[Y] * b.e[X]
    );
}

REAL len(VEC3 a) { return Sqrt(a.e[X] * a.e[X] + a.e[Y] * a.e[Y] + a.e[Z] * a.e[Z]); }
REAL lensq(VEC3 a) { return (a.e[X] * a.e[X] + a.e[Y] * a.e[Y] + a.e[Z] * a.e[Z]); }
VEC3 normalize(VEC3 v) { return v / len(v); }

QUATERNION operator+(QUATERNION q1, QUATERNION q2) { return QUATERNION(q1.e[X] + q2.e[X], q1.e[Y] + q2.e[Y], q1.e[Z] + q2.e[Z], q1.e[S] + q2.e[S]); }
QUATERNION operator-(QUATERNION q1, QUATERNION q2) { return QUATERNION(q1.e[X] - q2.e[X], q1.e[Y] - q2.e[Y], q1.e[Z] - q2.e[Z], q1.e[S] - q2.e[S]); }

QUATERNION operator*(QUATERNION q1, QUATERNION q2)
{
    VEC3 v1 = VEC3(q1.e[X], q1.e[Y], q1.e[Z]);
    VEC3 v2 = VEC3(q2.e[X], q2.e[Y], q2.e[Z]);
    REAL s1 = q1.e[S];
    REAL s2 = q2.e[S];
    return QUATERNION(
        s1 * v2 + s2 * v1 + cross(v1, v2),
        s1 * s2 - dot(v1, v2)
    );
}

QUATERNION operator/(QUATERNION q1, QUATERNION q2)
{
    return q1 * inv(q2);
}

QUATERNION conj(QUATERNION q)
{
    return QUATERNION(-q.e[X], -q.e[Y], -q.e[Z], q.e[S]);
}

QUATERNION inv(QUATERNION q)
{
    return conj(q) / normsq(q);
}

REAL norm(QUATERNION q)
{
    return Sqrt( q.e[X] * q.e[X] + q.e[Y] * q.e[Y] + q.e[Z] * q.e[Z] + q.e[S] * q.e[S]);
}

REAL normsq(QUATERNION q) { return (q.e[X] * q.e[X] + q.e[Y] * q.e[Y] + q.e[Z] * q.e[Z] + q.e[S] * q.e[S]); }
QUATERNION normalize(QUATERNION q) { return q / norm(q); }

REAL dot(QUATERNION q1, QUATERNION q2)
{
    return q1.e[X] * q2.e[X] + q1.e[Y] * q2.e[Y] + q1.e[Z] * q2.e[Z] + q1.e[S] * q2.e[S];
}

QUATERNION slerp(QUATERNION q1, QUATERNION q2, REAL t)
{
    REAL theta = ArcCos(dot(q1, q2));
    QUATERNION q = (Sin((1 - t)*theta) * q1 + Sin(t*theta) * q2) / Sin(theta);
    return q;
}

MAT3x3 QuaternionToMatrix(QUATERNION q)
{
    REAL t1, t2;
    MAT3x3 m;

    m.e[XX] = ONE - TWO * (q.e[Y] * q.e[Y] + q.e[Z] * q.e[Z]);
    m.e[YY] = ONE - TWO * (q.e[X] * q.e[X] + q.e[Z] * q.e[Z]);
    m.e[ZZ] = ONE - TWO * (q.e[X] * q.e[X] + q.e[Y] * q.e[Y]);

    t1 = q.e[X] * q.e[Y];
    t2 = q.e[S] * q.e[Z];
    m.e[YX] = TWO * (t1 - t2);
    m.e[XY] = TWO * (t1 + t2);

    t1 = q.e[X] * q.e[Z];
    t2 = q.e[S] * q.e[Y];
    m.e[ZX] = TWO * (t1 + t2);
    m.e[XZ] = TWO * (t1 - t2);

    t1 = q.e[Y] * q.e[Z];
    t2 = q.e[S] * q.e[X];
    m.e[ZY] = TWO * (t1 - t2);
    m.e[YZ] = TWO * (t1 + t2);

    return m;
}

QUATERNION MatrixToQuaternion(MAT3x3 m)
{
    REAL tr, s;
    int  i = XX;
    QUATERNION q;

    tr = m.e[XX] + m.e[YY] + m.e[ZZ];

    if (tr >= 0) {
        s = Sqrt(tr + ONE);
        q.e[S] = HALF * s;
        s = HALF / s;
        q.e[X] = (m.e[YZ] - m.e[ZY]) * s;
        q.e[Y] = (m.e[ZX] - m.e[XZ]) * s;
        q.e[Z] = (m.e[XY] - m.e[YX]) * s;
    }
    else {
        if (m.e[YY] > m.e[XX]) i = YY;
        if (m.e[ZZ] > m.e[i]) i = ZZ;
        switch (i) {
        case XX:
            s = Sqrt((m.e[XX] - (m.e[YY] + m.e[ZZ])) + ONE);
            q.e[X] = HALF * s;
            s = HALF / s;
            q.e[Y] = (m.e[YX] + m.e[XY]) * s;
            q.e[Z] = (m.e[XZ] + m.e[ZX]) * s;
            q.e[S] = (m.e[YZ] - m.e[ZY]) * s;
            break;
        case YY:
            s = Sqrt((m.e[YY] - (m.e[ZZ] + m.e[XX])) + ONE);
            q.e[Y] = HALF * s;
            s = HALF / s;
            q.e[Z] = (m.e[ZY] + m.e[YZ]) * s;
            q.e[X] = (m.e[YX] + m.e[XY]) * s;
            q.e[S] = (m.e[ZX] - m.e[XZ]) * s;
            break;
        case ZZ:
            s = Sqrt((m.e[ZZ] - (m.e[XX] + m.e[YY])) + ONE);
            q.e[Z] = HALF * s;
            s = HALF / s;
            q.e[X] = (m.e[XZ] + m.e[ZX]) * s;
            q.e[Y] = (m.e[ZY] + m.e[YZ]) * s;
            q.e[S] = (m.e[XY] - m.e[YX]) * s;
            break;
        }
    }

    return q;
}

QUATERNION VectorMulQuaternion(VEC3 v, QUATERNION q)
{
    QUATERNION t(v, ZERO); /* expand vector to quaternion */
    return t * q;
}

MAT3x3 operator*(MAT3x3 a, MAT3x3 b)
{
    return MAT3x3(
        a.e[XX] * b.e[XX] + a.e[XY] * b.e[YX] + a.e[XZ] * b.e[ZX], a.e[XX] * b.e[XY] + a.e[XY] * b.e[YY] + a.e[XZ] * b.e[ZY], a.e[XX] * b.e[XZ] + a.e[XY] * b.e[YZ] + a.e[XZ] * b.e[ZZ],
        a.e[YX] * b.e[XX] + a.e[YY] * b.e[YX] + a.e[YZ] * b.e[ZX], a.e[YX] * b.e[XY] + a.e[YY] * b.e[YY] + a.e[YZ] * b.e[ZY], a.e[YX] * b.e[XZ] + a.e[YY] * b.e[YZ] + a.e[YZ] * b.e[ZZ],
        a.e[ZX] * b.e[XX] + a.e[ZY] * b.e[YX] + a.e[ZZ] * b.e[ZX], a.e[ZX] * b.e[XY] + a.e[ZY] * b.e[YY] + a.e[ZZ] * b.e[ZY], a.e[ZX] * b.e[XZ] + a.e[ZY] * b.e[YZ] + a.e[ZZ] * b.e[ZZ]
    );
}

VEC3 operator*(MAT3x3 a, VEC3 b)
{
    return VEC3(
        a.xx * b.x + a.xy * b.y + a.xz * b.z,
        a.yx * b.x + a.yy * b.y + a.yz * b.z,
        a.zx * b.x + a.zy * b.y + a.zz * b.z
    );
}

MAT3x3 transpose(MAT3x3 a)
{
    return MAT3x3(
        a.e[XX], a.e[YX], a.e[ZX],
        a.e[XY], a.e[YY], a.e[ZY],
        a.e[XZ], a.e[YZ], a.e[ZZ]
    );
}

MAT3x3 inv(MAT3x3 a)
{
	MAT3x3 aInv;
	REAL det = a.xx * (a.yy * a.zz - a.zy * a.yz) -
		a.xy * (a.yx * a.zz - a.yz * a.zx) +
		a.xz * (a.yx * a.zy - a.yy * a.zx);
	REAL detInv = ONE / det;
	aInv.xx = (a.yy * a.zz - a.zy * a.yz) * detInv;
	aInv.xy = (a.xz * a.zy - a.xy * a.zz) * detInv;
	aInv.xz = (a.xy * a.yz - a.xz * a.yy) * detInv;
	aInv.yx = (a.yz * a.zx - a.yx * a.zz) * detInv;
	aInv.yy = (a.xx * a.zz - a.xz * a.zx) * detInv;
	aInv.yz = (a.yx * a.xz - a.xx * a.yz) * detInv;
	aInv.zx = (a.yx * a.zy - a.zx * a.yy) * detInv;
	aInv.zy = (a.zx * a.xy - a.xx * a.zy) * detInv;
	aInv.zz = (a.xx * a.yy - a.yx * a.xy) * detInv;
	return aInv;
}

QUATERNION operator*(QUATERNION q, REAL a) { return QUATERNION(q.e[X] * a, q.e[Y] * a, q.e[Z] * a, q.e[S] * a); }
QUATERNION operator/(QUATERNION q, REAL a) { return QUATERNION(q.e[X] / a, q.e[Y] / a, q.e[Z] / a, q.e[S] / a); }
QUATERNION operator*(REAL a, QUATERNION q) { return QUATERNION(a * q.e[X], a * q.e[Y], a * q.e[Z], a * q.e[S]); }
QUATERNION operator/(REAL a, QUATERNION q) { return a * inv(q); }


