#pragma once
#include "nn/nn_global.h"
#include "nn/tensor.h"
#include "nn/avx.h"

/* define specialized tensor types for applying optimizations */
/* 
   _TensorBase_t
           +--- RealTensor
           |    +--- FloatTensor
           |    +--- DoubleTensor
           +--- IntTensor
*/

template<typename TensorType>
bool T_mm_check(const TensorType& m1, const TensorType& m2)
{
    /* check if two tensors can perform matrix multiplication */
    TensorShape lshape = m1.shape(), rshape = m2.shape();
    if (m1.dim() != 2 || m2.dim() != 2) {
        T_error("Cannot perform matrix multiplication for tensors that are not "
            "two dimensional. If you want to multiply a vector with a matrix "
            "please convert the vector to 2-dim matrix before multiplying.");
        return false;
    }
    if (lshape[1] != rshape[0]) {
        char s[64], t[64];
        lshape.toStr(s);
        rshape.toStr(t);
        T_error("Cannot perform matrix multiplication due to invalid matrix "
            "shapes (try to matmul matrix with shape \"%s\" to matrix with "
            "shape \"%s\").", s, t);
        return false;
    }
    return true;
}
template<typename TensorType>
bool T_bmm_check(const TensorType& m1, const TensorType& m2)
{
    /* check if two tensors can perform batched matrix multiplication */
    TensorShape lshape = m1.shape(), rshape = m2.shape();
    char s[64], t[64];
    if (m1.dim() != 3 || m2.dim() != 3) {
        lshape.toStr(s); rshape.toStr(t);
        T_error("Cannot perform batched matrix multiplication for tensors that "
            "are not three dimensional. Tensor shapes need to be (b,p,q) x (b,q,r), "
            "got (%s) x (%s).", s, t);
        return false;
    }
    if (lshape[0] != rshape[0] || lshape[2] != rshape[1]) {
        lshape.toStr(s); rshape.toStr(t);
        T_error("Cannot perform batched matrix multiplication due to invalid "
            " tensor shape. Tensor shapes need to be (b,p,q) x (b,q,r), got "
            "(%s) x (%s).", s, t);
        return false;
    }
    return true;
}

class FloatTensor : public _TensorBase_t<float> 
{
public:
    FloatTensor();
    FloatTensor(const char* shp);
    FloatTensor(const TensorShape& tshp);
    FloatTensor(const FloatTensor& that);
    FloatTensor(const _TensorBase_t<float>& that); /* NOTE: initialize from base template */
    FloatTensor(const float& value);
    virtual ~FloatTensor();
public:
    /* optimized tensor operations */
    FloatTensor mm(const FloatTensor& that) const;
    FloatTensor softmax(const int& axis) const;
    FloatTensor mm_t(const FloatTensor& that) const;
    /* 2D ordinary convolution (no padding, unit stride),       */
    /* naive version. suitable for large image and large kernel */
    FloatTensor conv2d(const FloatTensor& kernel) const;

    /* print related */
public:
    virtual void print() const;
    void setPrintOptions(int all, int right);
protected:
    char printFormatter[64];
};
class DoubleTensor : public _TensorBase_t<double>
{
public:
    DoubleTensor();
    DoubleTensor(const char* shp);
    DoubleTensor(const TensorShape& tshp);
    DoubleTensor(const DoubleTensor& that);
    DoubleTensor(const _TensorBase_t<double>& that);
    DoubleTensor(const double& value);
    virtual ~DoubleTensor();
public:
    /* optimized tensor operations */
    DoubleTensor mm(const DoubleTensor& that) const;
    DoubleTensor softmax(const int& axis) const;
    DoubleTensor mm_t(const DoubleTensor& that) const;
    /* 2D ordinary convolution (no padding, unit stride),       */
    /* naive version. suitable for large image and large kernel */
    DoubleTensor conv2d(const DoubleTensor& kernel) const;

    /* print related */
public:
    virtual void print() const;
    void setPrintOptions(int all, int right);
protected:
    char printFormatter[64];
};
class IntTensor : public _TensorBase_t<int>
{
public:
    IntTensor();
    IntTensor(const char* shp);
    IntTensor(const TensorShape& tshp);
    IntTensor(const IntTensor& that);
    IntTensor(const _TensorBase_t<int>& that);
    IntTensor(const int& value);
    virtual ~IntTensor();
public:
    /* optimized tensor operations */
    IntTensor mm(const IntTensor& that) const;
    IntTensor softmax(const int& axis) const;
    IntTensor mm_t(const IntTensor& that) const;
    /* 2D ordinary convolution (no padding, unit stride),       */
    /* naive version. suitable for large image and large kernel */
    IntTensor conv2d(const IntTensor& kernel) const;

    /* print related */
public:
    virtual void print() const;
    void setPrintOptions(int all);
protected:
    char printFormatter[64];
};

/* define RealTensor */
#ifdef USE_SINGLE_PRECISION
typedef FloatTensor RealTensor;
#else
typedef DoubleTensor RealTensor;
#endif

/* optimized tensor operations as global function */
/* matrix multiplication (p,q) x (q,r) = (p,r) */
template <typename TensorType>
TensorType T_mm(const TensorType& m1, const TensorType& m2) { return m1.mm(m2); }
/* batch matrix multiplication (b,p,q) x (b,q,r) = (b,p,r) */
template <typename TensorType>
TensorType T_bmm(const TensorType& m1, const TensorType& m2)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    T_bmm_check(m1, m2);
#endif
    TensorShape lshape = m1.shape(), rshape = m2.shape();
    int batchSize = lshape[0];
    TensorType resultTensor(T_buildShape<3>({ batchSize, lshape[1], rshape[2] }));
#ifdef NN_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < batchSize; i++) {
        /* setting up coordinates and matrices */
        TensorCoord ls = T_buildCoord<3>({ i, 0, 0 }), le = T_buildCoord<3>({ i + 1, lshape[1], lshape[2] });
        TensorCoord rs = T_buildCoord<3>({ i, 0, 0 }), re = T_buildCoord<3>({ i + 1, rshape[1], rshape[2] });
        TensorType s1 = m1.at(ls, le), s2 = m2.at(rs, re);
        s1 = s1.squeeze(1);
        s2 = s2.squeeze(1);
        /* matmul */
        TensorType s = T_mm(s1, s2);
        /* write result */
        int nElem = lshape[1] * rshape[2];
        int iStart = i * nElem;
        resultTensor._dptrWrite(iStart, nElem, s);
    }
    return resultTensor;
}
/* matrix multiplication, support the following 4 forms: */
/*   (p,q) x (q,r)   = (p,r)   , or */
/*   (p,q) x (b,q,r) = (b,p,r) , or */
/* (b,p,q) x (q,r)   = (b,p,r) , or */
/* (b,p,q) x (b,q,r) = (b,p,r) .    */
template <typename TensorType>
TensorType T_mmul(const TensorType& m1, const TensorType& m2)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (m1.dim() < 2 || m1.dim() > 3 || m2.dim() < 2 || m2.dim() > 3)
        T_error("cannot perform mmul() because tensor dims are invalid.");
#endif
    if (m1.dim() == 2 && m2.dim() == 2)
        return T_mm(m1, m2);
    else if (m1.dim() == 3 && m2.dim() == 3)
        return T_bmm(m1, m2);
    else {
        /* determine batch size */
        int batchSize;
        if (m1.dim() == 3) batchSize = m1.shape()[0];
        else batchSize = m2.shape()[0];
        /* determine the result tensor shape */
        TensorShape lshape, rshape;
        const TensorType* pBroadcast; /* pointer to tensor that needs to be brocasted */
        if (m1.dim() == 3) {
            lshape = m1.shape();
            rshape = T_buildShape<3>({ batchSize, m2.shape()[0], m2.shape()[1] });
            pBroadcast = &m2;
        }
        else {
            lshape = T_buildShape<3>({ batchSize, m1.shape()[0], m1.shape()[1] });
            rshape = m2.shape();
            pBroadcast = &m1;
        }
        /* make broadcast tensor */
        TensorType tensorBroadcast(T_buildShape<3>({ batchSize, pBroadcast->shape()[0], pBroadcast->shape()[1] }));
        int nElem = pBroadcast->shape()[0] * pBroadcast->shape()[1];
        for (int i = 0; i < batchSize; i++)
            tensorBroadcast._dptrWrite(i*nElem, nElem, (*pBroadcast));
        /* do batch matrix multiplication (order not changed) */
        if (pBroadcast == &m1)
            return T_bmm(tensorBroadcast, m2);
        else
            return T_bmm(m1, tensorBroadcast);
    }
}
template <typename TensorType>
TensorType T_softmax(const TensorType& tensor, const int& axis) { return tensor.softmax(axis); }
template <typename TensorType>
TensorType T_mm_t(const TensorType& m1, const TensorType& m2) { return m1.mm_t(m2); }
