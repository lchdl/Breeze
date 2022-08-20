/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*        tensor.h: A very simple tensor library written in C++        */
/*                                                                     */
/*                      Chenghao Liu, 2022-03-10                       */
/*                                                                     */
/* This lightweight tensor library is targeted for optimal performance,*/
/* and can be used as a robust tool for principle validation.          */
/*                                                                     */
/* shape descriptor| A C-style string represents the size of a tensor, */
/*                 | such as "5,4,3,2". The shape descriptor must not  */
/*                 | contain any space ' ' and should be separated by  */
/*                 | ',' for each dimension. A scalar will always has  */
/*                 | shape "0". Note that "0" is the only way of       */
/*                 | representing the shape of a scalar, the following */
/*                 | shape descriptors are all invalid: "0,1,3", "3,0",*/
/*                 | "0,0", etc.                                       */
/*                                                                     */
/* raw data storage| For example, "5,3,2,6" represents a 4-dim tensor  */
/*                 | with shape 5x3x2x6, the right-most axis is the    */
/*                 | lowest dimension and the left-most axis is the    */
/*                 | highest dimension. The data is stored linearly    */
/*                 | with "0,0,0,0" stored first, then "0,0,0,1", .. , */
/*                 | the last element is "4,2,1,5". Note that a tensor */
/*                 | with shape "0" represents a scalar, while a tensor*/
/*                 | with shape "1" represents a 1-dim array with only */
/*                 | one element.                                      */
/*                                                                     */
/*   tensor slicing| This library supports tensor slicing. The slicing */
/*                 | declaration is similar to NumPy library. Here are */
/*                 | some examples of this: ":,0,4", "0:3,1:5,4:6",    */
/*                 | ":,:,:", "1:,:4,7".                               */
/*                                                                     */
/*  tensor axis and| The highest dimension is axis 1 while the lowest  */
/*        dimension| dimension is axis dim(). Take a look at the       */
/*                 | following example:                                */
/*                 |                  shape=(   5,   3,   2,   6)      */
/*                 |                            ^    ^    ^    ^       */
/*                 |                   axis:    1    2    3    4       */
/*                 |                    dim:    4    3    2    1       */
/*                 | * Note that the dimension starts with 1 for a     */
/*                 |   regular tensor. Dimension for a scalar is 0.    */
/*                                                                     */
/* tensor reshaping| You can reshape a tensor to a new size (scalar do */
/*                 | not support reshaping operation) using a shape    */
/*                 | descriptor.                                       */
/*                 | * Note: one shape dimension can be undetermined,  */
/*                 |   the value is inferred from total number of elems*/
/*                 |   and the remaining dimensions. For example:      */
/*                 |   "*,16,16" will be converted to "2,16,16" if # of*/
/*                 |   elements is 512.                                */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#pragma once
#include "basedefs.h"
#include "basemath.h"
#include "nn/nn_global.h"

#define TENSOR_DESCRIPTION_LENGTH 256

/* simple error logging mechanic */
void T_error(const char* format, ...);

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* a struct expressing the shape / indexing of a tensor  */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * */
typedef class _TensorDim_t {
public:
    _TensorDim_t();
    _TensorDim_t(const char* init);
    _TensorDim_t(const int& nZ);
    _TensorDim_t(const Array<int>& init);
    _TensorDim_t(const _TensorDim_t& that);
    _TensorDim_t(const _TensorDim_t& start, const _TensorDim_t& end);
    /* * * * * * * */
    /* TensorShape */
    /* * * * * * * */
    int&       operator[](const int& index);            /* returns size of the x-th axis (write permitted)    */
    const int& operator[](const int& index) const;      /* returns size of the x-th axis (read only)          */
    int        dim() const;                             /* returns shape dimension                            */
    bool       isShapeValid() const;                    /* check if shape is valid                            */
    bool       isEqual(const _TensorDim_t& that) const; /* compare if two TensorDims are equal                */
    bool    operator==(const _TensorDim_t& that) const; /* equivalent to isEqual()                            */
    bool    operator!=(const _TensorDim_t& that) const;
    void       scalar();                                /* make scalar shape ("0")                            */
    int        numel() const;                           /* returns number of elements                         */
    _TensorDim_t toPos(const UINT& addr) const;         /* internal convert absolute address to position      */
    bool   rangeCheckAt(const _TensorDim_t& pos) const; /* checking if a given position is out of range       */
    bool  rangeCheckEnd(const _TensorDim_t& pos) const; /* checking if a given position is out of range       */
    /* * * * * * * */
    /* TensorCoord */
    /* * * * * * * */
    _TensorDim_t operator+(const _TensorDim_t& that) const; /* define common calculations of two coordinates  */
    _TensorDim_t operator-(const _TensorDim_t& that) const;
    _TensorDim_t operator-() const;
    void       zeros(const int& nZ);                    /* initialize a certain number of zeros               */
    bool       next(const _TensorDim_t& shp);           /* advance to the next position according to shp,     */
                                                        /* return false if this coord is the last coord       */
    bool       next(const _TensorDim_t& start, const _TensorDim_t& end);
    bool       next(const _TensorDim_t& shp, const Array<int>& steps);
    bool       next(const _TensorDim_t& start, const _TensorDim_t& end, const Array<int>& steps);
    bool       prev(const _TensorDim_t& shp);           /* roll back to the previous coord according to shp   */
                                                        /* return false if this coord is the first coord      */
    bool       prev(const _TensorDim_t& start, const _TensorDim_t& end);
    UINT       toAddr(const _TensorDim_t& shp) const;   /* internal convert position to absolute address      */
    UINT       toAddr(const _TensorDim_t& shp, const UINT& mask) const; /* masked access */
    /* * * * * * * * * * * * * * */
    /* TensorShape & TensorCoord */
    /* * * * * * * * * * * * * * */
    void       fromStr(const char* shp);                /* convert shape descriptor to shape object           */
    void       toStr(char* shp) const;                  /* convert shape object to shape descriptor,          */
                                                        /* you need to ensure the string buffer is big enough */
    void       print();                                 /* print the content (for debugging purpose)          */
    void       clear();
protected:
    /* * * * * * * * * * * * * * */
    /* TensorShape & TensorCoord */
    /* * * * * * * * * * * * * * */
    void       _append(int d);
protected:
    Array<int> _data;

} TensorShape, TensorCoord;

/* utility functions to quickly build an shape object         */
/* usage: TensorShape shp = T_buildShape<3>({5,5,5});         */
/* you will need to add "-std=c++11" to your compiler options */
template <int D>
const TensorShape T_buildShape(const int(&init)[D]) {
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (D < 1)
        T_error("Invalid dimension setting (%d<1) when calling T_buildShape/Coord(...).", D);
#endif
    TensorShape shape(D);
    for (int i = 0; i < D; i++)
        shape[i] = init[i];
    return shape;
}

#define T_buildCoord T_buildShape

/* build array from function                                  */
/* usage: to build an array holding 4 real numbers:           */
/* Array<REAL> arr = T_buildArray<4,REAL>({1,2,3,4});         */
/* you will need to add "-std=c++11" to your compiler options */
template <int N, typename T>
const Array<T> T_buildArray(const T(&init)[N]) {
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (N <= 0)
        T_error("Invalid size setting (%d<=0) when calling T_buildArray(...).", N);
#endif
    Array<T> arr;
    for (int i = 0; i < N; i++)
        arr.append(init[i]);
    return arr;
}

/* build array from string literal (must <512 chars!!!) */
/* Array<REAL> arr = T_buildArray<REAL>("1,2,3,4");     */
template <typename T>
const Array<T> T_buildArray(const char* sz) {
    const int slen = 512, buflen = 64;
    Array<T> arr;
    char s[slen], buf[buflen];
    strRemCh(sz, s, slen, ' ');
    //strncpy(s, shp, slen - 1);
    while (strGetWord(s, buf, buflen - 1, ",", '\0')) {
        T val = T(atof(buf));
        arr.append(val);
    }
    return arr;
}

template <typename T>
void T_swap(T& p1, T& p2) { T t = p1; p1 = p2; p2 = t; }

/* check if the two tensors have the same shape */
bool T_shapeEq(const TensorShape& shp1, const TensorShape& shp2);
/* check if two shapes can be the same after broadcasting */
bool T_shapeEqBcast(const TensorShape& shp1, const TensorShape& shp2);

UINT _T_buildMaskFromShape(const TensorShape& shp);
TensorShape _T_maxShape(const TensorShape& shp1, const TensorShape& shp2);

template <typename T>
struct TypeReturn {typedef T type;};

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* _TensorBase_t class. Defines the tensor data structure  */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
template <typename T>
class _TensorBase_t : public TypeReturn<T> {

protected:
    /* member "_shape": Shape of the tensor object. */
    TensorShape _shape;
    /* member "_dptr": Data pointer. Points to the starting memory address of the data array. */
    T* _dptr;

public:
    _TensorBase_t() { _dptr = NULL; }
    _TensorBase_t(const char* shp) {
        _dptr = NULL;
        TensorShape tshp;
        tshp.fromStr(shp);
        if (tshp.isShapeValid() == false)
            T_error("Error, invalid tensor shape indicator \"%s\".\n", shp);
        else
            _create(tshp);
    }
    _TensorBase_t(const TensorShape& tshp) {
        _dptr = NULL;
        if (tshp.isShapeValid() == false) {
            char buf[64];
            tshp.toStr(buf);
            T_error("Error, invalid tensor shape indicator \"%s\".\n", buf);
        }
        else
            _create(tshp);
    }
    _TensorBase_t(const _TensorBase_t& that) {
        _dptr = NULL;
        _copy(that);
    }
    _TensorBase_t(const T& value) {
        _dptr = NULL;
        this->_shape.scalar();
        this->_create(_shape);
        this->_dptr[0] = value;
    }
    virtual ~_TensorBase_t() { purge(); }

    _TensorBase_t& operator=(const _TensorBase_t& that) {
        if (this == &that)
            return (*this);
        _copy(that);
        return (*this);
    }
    _TensorBase_t& operator=(const T& value) {
        /* scalar assignment */
        if (isScalar())
            _dptr[0] = value;
        else {
            /* manual create object */
            purge();
            int zero = 0;
            this->_shape.scalar();
            this->_dptr = (T*)malloc(sizeof(T) * 1);
            this->_dptr[0] = value;
        }
        return (*this);
    }

    bool isNull() const { return (_dptr == NULL);}

    /* return the value if the tensor is a scalar */
    T& item() const {
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (numel() != 1)
            T_error("cannot use item() because the number of elements is not 1.\n");
#endif
        return _dptr[0];
    }

    /* set all elements to a value, inplace fill_ */
    void fill_(const T& value) {
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (isNull())
            T_error("cannot fill a null tensor.");
#endif
        int _numel = this->numel();
        for (int i = 0; i < _numel; i++)
            _dptr[i] = value;
    }
    
    /* retrieve / change the value of a single element */
    T& at(const TensorCoord& coord) const {
#ifdef NN_ENABLE_RUNTIME_CHECKING
        /* range checking */
        if (!_shape.rangeCheckAt(coord)) {
            char p[256], s[256];
            coord.toStr(p);
            _shape.toStr(s);
            T_error("Position \"%s\" is out of range. Tensor shape is \"%s\".\n", p, s);
            return _dptr[0];
        }
#endif
        unsigned int i = coord.toAddr(_shape);
        return _dptr[i];
    }
    T& operator[](const TensorCoord& coord) const { return at(coord); }
    T& at(const int& offset) const {
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (offset < 0 || offset >= numel()) {
            char a[64];
            this->shape().toStr(a);
            T_error("Index out of range. Tensor shape is (%s), "
                "trying to access index=%d (expected 0~%d).", a, offset, numel() - 1);
        }
#endif
        return _dptr[offset];
    }
    T& operator[](const int& offset) const { return at(offset); }

    /* sample from a coordinate, but if coordinate is out of bound a default value will return */
    T sample(const TensorCoord& coord, const T& defVal) const {
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (isScalar())
            T_error("scalar does not support sample().");
        if (dim() != coord.dim())
            T_error("invalid dimension.");
#endif
        /* check if out of bound */
        bool oob = false;
        TensorShape shp = shape();
        for (int i = 0; i < coord.dim(); i++) {
            if (coord[i] < 0 || coord[i] >= shp[i]) {
                oob = true;
                break;
            }
        }
        if (oob)
            return defVal;
        else
            return at(coord);
    }
    
    /* retrieve the tensor data at a certain range (slicing), returns a new tensor */
    _TensorBase_t<T> at(const TensorCoord& coordStart, const TensorCoord& coordEnd) const {
#ifdef NN_ENABLE_RUNTIME_CHECKING
        _checkCoordRange(coordStart, coordEnd);
#endif
        /* if this tensor is a scalar, then slicing operation must returns a scalar */
        if (isScalar()) {
            return _TensorBase_t<T>(_dptr[0]);
        }
        else {
            /* a regular tensor, compute tensor size from starting and ending positions */
            int nZ = (coordStart.dim() == 0 ? 1 : coordStart.dim());
            TensorShape tensorShape(nZ);
            for (int i = 0; i < nZ; i++)
                tensorShape[i] = coordEnd[i] - coordStart[i];
            TensorCoord dstPos(nZ), srcPos = coordStart;
            _TensorBase_t<T> tensorBlock(tensorShape);
            /* fill_ data one by one */
            bool ok = true;
            while (ok) {
                tensorBlock[dstPos] = this->at(srcPos);
                ok = srcPos.next(coordStart, coordEnd);
                dstPos.next(tensorShape);
            }
            return tensorBlock;
        }
    }
    _TensorBase_t<T> operator[](const char* slicing) const {
        TensorCoord coordStart, coordEnd;
        bool ok = _parseSlicing(slicing, coordStart, coordEnd);
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (!ok) {
            char buf[64];
            TensorShape shp = shape();
            shp.toStr(buf);
            T_error("invalid slicing: \"%s\". Tensor shape is \"%s\".", slicing, shp);
        }
#endif
        _TensorBase_t<T> newTensor = at(coordStart, coordEnd);
        if (newTensor.isScalar())
            return newTensor;
        /* squeeze axes with no slicing */
        char s[64],t[16];
        int n, ax = 1;
        Array<int> squeezeAxes;
        strRemCh(slicing, s, 64, ' ');
        strSplit(s, t, 16, ",", &n);
        while (n) {
            strSplit(s, t, 16, ",", NULL);
            if (!chInStr(':', t))
                squeezeAxes.append(ax);
            n--;
            ax++;
        }
        return newTensor.squeeze(squeezeAxes);
    }
    
    /* assign value/tensor to a tensor block */
    void assign(const TensorCoord& coordStart, const TensorCoord& coordEnd, const T& value) {
#ifdef NN_ENABLE_RUNTIME_CHECKING
        _checkCoordRange(coordStart, coordEnd);
#endif
        TensorCoord coord = coordStart;
        do {
            at(coord) = value;
        } while (coord.next(coordStart, coordEnd));
    }
    void assign(const TensorCoord& coordStart, const TensorCoord& coordEnd, const _TensorBase_t<T>& tensor) {
#ifdef NN_ENABLE_RUNTIME_CHECKING
        _checkCoordRange(coordStart, coordEnd);
        TensorShape dstShape(coordStart, coordEnd);
        if (dstShape != tensor.shape()) {
            char s0[64], s1[64];
            tensor.shape().toStr(s0);
            dstShape.toStr(s1);
            T_error("cannot assign tensor (shape=%s) to a block region (shape=%s), shape is different.", s0, s1);
        }
#endif
        TensorCoord srcCoord, dstCoord;
        TensorShape srcShape = tensor.shape();
        do {
            (*this)[dstCoord] = tensor[srcCoord];
            dstCoord.next(coordStart, coordEnd);
        } while (srcCoord.next(srcShape));
    }
    void assign(const char* slicing, const T& value) {
        TensorCoord coordStart, coordEnd;
        bool valid = _parseSlicing(slicing, coordStart, coordEnd);
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (!valid)
            T_error("slicing %s not valid.", slicing);
#endif
        assign(coordStart, coordEnd, value);
    }
    void assign(const char* slicing, const _TensorBase_t<T>& tensor) {
        TensorCoord coordStart, coordEnd;
        bool valid = _parseSlicing(slicing, coordStart, coordEnd);
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (!valid)
            T_error("slicing %s not valid.", slicing);
#endif
        rangeAssign(coordStart, coordEnd, tensor);
    }

    /* remove all dimensions with size 1. For example: "5,1,4,2,1" => "5,4,2" */
    _TensorBase_t<T> squeeze() const {
        /* check if this tensor is a scalar, if yes the result will be a scalar */
        if (isScalar() || numel() == 1)
            return _TensorBase_t<T>(_dptr[0]);
        TensorShape dstShape, srcShape = shape();
        int _dim = 0;
        for (int i = 0; i < srcShape.dim(); i++)
            if (srcShape[i] > 1) _dim++;
        dstShape.zeros(_dim);
        int j = _dim - 1;
        for (int i = srcShape.dim() - 1; i >= 0; i--) {
            if (srcShape[i] > 1)
                dstShape[j--] = srcShape[i];
        }
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (dstShape.numel() != numel())
            T_error("# of elements changed after squeeze().");
#endif
        _TensorBase_t<T> squeezeTensor(dstShape);
        /* squeeze() does not change the raw data storage, so we */
        /* don't need to set the new tensor element by element */
        memcpy(squeezeTensor._dptr, _dptr, sizeof(T) * numel());
        return squeezeTensor;
    }
    /* remove specified dimensions with size 1. For example: */
    /*   "5,1,4,2,1" => remove axis 2     => "5,4,2,1" */
    /*     "1,4,2,5" => remove axis 1     => "4,2,5"   */
    /* "3,6,2,1,1,1" => remove axis 4,5,6 => "3,6,2"   */
    _TensorBase_t<T> squeeze(const Array<int>& axesArr) const {
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (isScalar())
            T_error("cannot squeeze a particular axis for a scalar. Use isScalar() to "
                "check if this tensor is a scalar and use item() to extract the actual "
                "value.");
        for (int i = 0; i < axesArr.size(); i++) {
            if (axesArr[i]<1 || axesArr[i]>dim())
                T_error("cannot squeeze tensor because the axis value (%d) is invalid.", axesArr[i]);
            if (shape()[axesArr[i] - 1] != 1)
                T_error("cannot squeeze tensor because the size of the target axis (%d) is not 1.", axesArr[i]);
        }
#endif
        /* must not be a scalar */
        TensorShape srcShape = shape();
        Array<int> squeezedShape;
        for (int i = 0; i < dim(); i++) {
            int thisAxis = i + 1;
            bool del = false;
            for (int j = 0; j < axesArr.size(); j++) {
                if (thisAxis == axesArr[j])
                    del = true;
            }
            if (!del)
                squeezedShape.append(srcShape[i]);
        }
        /* checking if tensor after squeezing is a scalar */
        if (squeezedShape.size() == 0)
            return _TensorBase_t<T>(_dptr[0]);
        /* otherwise, copy all elements */
        TensorShape dstShape(squeezedShape);
        _TensorBase_t<T> dstTensor(dstShape);
        int _numel = numel();
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (_numel != dstShape.numel())
            T_error("# of elements changed after squeezing.");
#endif
        memcpy(dstTensor._dptr, _dptr, sizeof(T) * _numel);
        return dstTensor;
    }
    _TensorBase_t<T> squeeze(const char* axes) const {
        /* obtain squeeze axes */
        Array<int> axesArr;
        char s[64] = { 0 }, w[16] = { 0 };
        strRemCh(axes, s, 64, ' ');
        //strncpy(s, axes, 63);
        int n = 0;
        strSplit(s, w, 16, ",", &n);
        while (n) {
            strSplit(s, w, 16, ",", NULL);
            axesArr.append(atoi(w));
            n--;
        }
        return squeeze(axesArr);
    }
    _TensorBase_t<T> squeeze(const int& axis) const {
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (isScalar())
            T_error("cannot squeeze a particular axis for a scalar. Use isScalar() to "
                "check if this tensor is a scalar and use item() to extract the actual "
                "value.");
        if (axis<1 || axis>dim())
            T_error("cannot squeeze tensor because the axis value (%d) is invalid.", axis);
        if (shape()[axis - 1] != 1)
            T_error("cannot squeeze tensor because the size of the target axis (%d) is not 1.", axis);
#endif
        /* must not be a scalar */
        TensorShape srcShape = shape();
        Array<int> squeezedShape;
        for (int i = 0; i < dim(); i++) {
            int thisAxis = i + 1;
            bool del = false;
            if (thisAxis == axis)
                del = true;
            if (!del)
                squeezedShape.append(srcShape[i]);
        }
        /* checking if tensor after squeezing is a scalar */
        if (squeezedShape.size() == 0)
            return _TensorBase_t<T>(_dptr[0]);
        /* otherwise, copy all elements */
        TensorShape dstShape(squeezedShape);
        _TensorBase_t<T> dstTensor(dstShape);
        int _numel = numel();
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (_numel != dstShape.numel())
            T_error("# of elements changed after squeezing.");
#endif
        memcpy(dstTensor._dptr, _dptr, sizeof(T) * _numel);
        return dstTensor;
    }
    void squeeze_(const Array<int>& axesArr) {
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (isScalar())
            T_error("cannot squeeze a particular axis for a scalar. Use isScalar() to "
                "check if this tensor is a scalar and use item() to extract the actual "
                "value.");
        for (int i = 0; i < axesArr.size(); i++) {
            if (axesArr[i]<1 || axesArr[i]>dim())
                T_error("cannot squeeze tensor because the axis value (%d) is invalid.", axesArr[i]);
            if (shape()[axesArr[i] - 1] != 1)
                T_error("cannot squeeze tensor because the size of the target axis (%d) is not 1.", axesArr[i]);
        }
#endif
        /* must not be a scalar */
        TensorShape srcShape = shape();
        Array<int> squeezedShape;
        for (int i = 0; i < dim(); i++) {
            int thisAxis = i + 1;
            bool del = false;
            for (int j = 0; j < axesArr.size(); j++) {
                if (thisAxis == axesArr[j])
                    del = true;
            }
            if (!del)
                squeezedShape.append(srcShape[i]);
        }
        /* checking if tensor after squeezing is a scalar */
        if (squeezedShape.size() == 0)
            _shape = TensorShape("0");
        else {
            TensorShape dstShape(squeezedShape);
#ifdef NN_ENABLE_RUNTIME_CHECKING
            if (numel() != dstShape.numel())
                T_error("# of elements changed after squeezing.");
#endif
            _shape = dstShape;
        }
    }
    void squeeze_(const char* axes) {
        /* obtain squeeze axes */
        Array<int> axesArr;
        char s[64] = { 0 }, w[16] = { 0 };
        strRemCh(axes, s, 64, ' ');
        //strncpy(s, axes, 63);
        int n = 0;
        strSplit(s, w, 16, ",", &n);
        while (n) {
            strSplit(s, w, 16, ",", NULL);
            axesArr.append(atoi(w));
            n--;
        }
        return squeeze_(axesArr);
    }
    void squeeze_(const int& axis) {
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (isScalar())
            T_error("cannot squeeze a particular axis for a scalar. Use isScalar() to "
                "check if this tensor is a scalar and use item() to extract the actual "
                "value.");
        if (axis<1 || axis>dim())
            T_error("cannot squeeze tensor because the axis value (%d) is invalid.", axis);
        if (shape()[axis - 1] != 1)
            T_error("cannot squeeze tensor because the size of the target axis (%d) is not 1.", axis);
#endif
        /* must not be a scalar */
        TensorShape srcShape = shape();
        Array<int> squeezedShape;
        for (int i = 0; i < dim(); i++) {
            int thisAxis = i + 1;
            bool del = false;
            if (thisAxis == axis)
                del = true;
            if (!del)
                squeezedShape.append(srcShape[i]);
        }
        /* checking if tensor after squeezing is a scalar */
        if (squeezedShape.size() == 0)
            _shape = TensorShape("0");
        else {
            TensorShape dstShape(squeezedShape);
#ifdef NN_ENABLE_RUNTIME_CHECKING
            if (numel() != dstShape.numel())
                T_error("# of elements changed after squeezing.");
#endif
            _shape = dstShape;
        }
    }
    /* add specified dimensions with size 1. For example: */
    /*       "2,3,4" => add axis 0 => "1,2,3,4"    */
    /*       "4,7,3" => add axis 1 => "4,1,7,3"    */
    /*         "9,8" => add axis 2 => "9,8,1"      */
    /*    "1,9,6,9,1 => add axis 5 => "1,9,6,9,1,1" */
    _TensorBase_t<T> unsqueeze(const int& axis) const {
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (isScalar())
            T_error("cannot unsqueeze for a scalar.");
        if (axis<0 || axis > dim())
            T_error("invalid axis setting.");
#endif
        TensorShape curShape = shape();
        Array<int> newShape;
        for (int i = 0; i <= dim(); i++) {
            if (i == axis)
                newShape.append(1);
            if (i < dim())
                newShape.append(curShape[i]);
        }
        TensorShape _newShape(newShape);
        _TensorBase_t newTensor(_newShape);
        memcpy(newTensor._dptr, _dptr, sizeof(T) * numel());
        return newTensor;
    }
    void unsqueeze_(const int& axis) {
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (isScalar())
            T_error("cannot unsqueeze for a scalar.");
        if (axis<0 || axis > dim())
            T_error("invalid axis setting.");
#endif
        TensorShape curShape = shape();
        Array<int> newShape;
        for (int i = 0; i <= dim(); i++) {
            if (i == axis)
                newShape.append(1);
            if (i < dim())
                newShape.append(curShape[i]);
        }
        /* unsqueeze() just changes the tensor shape */
        /* it does not change the raw data storage order */
        _shape = TensorShape(newShape); 
    }

    /* returns tensor dimension */
    int dim() const {
        return _shape.dim();
    }
    
    /* returns true if tensor is a scalar */
    bool isScalar() const {
        if (_shape.dim() == 0)
            return true;
        else return false;
    }
    
    /* returns the number of elements in this tensor */
    int numel() const {
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (_shape.isShapeValid() == false)
            T_error("invalid tensor shape.\n");
#endif
        return _shape.numel();
    }
    
    /* returns tensor shape */
    TensorShape shape() const {
        return this->_shape;
    }
    
    /* check if two tensors have the same shape */
    bool isShapeEq(const _TensorBase_t& that) const {
        return this->_shape.isEqual(that._shape);
    }

    /* transpose tensor data */
    _TensorBase_t<T> transpose(const Array<int>& axesMap) const {
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (dim() < 2)
            T_error("scalar or vector does not support transpose operation.");
        if (axesMap.size() != dim())
            T_error("invalid tranpose axes setting.");
        for (int _ax = 1; _ax <= dim(); _ax++) {
            bool found = false;
            for (int i = 0; i < axesMap.size(); i++) {
                if (_ax == axesMap[i])
                    found = true;
            }
            if (!found)
                T_error("cannot find axis \"%d\" in axesMap.", _ax);
        }
#endif
        TensorShape srcShape = shape(), dstShape(dim());
        for (int i = 0; i < dim(); i++)
            dstShape[i] = srcShape[axesMap[i]-1];
        TensorCoord srcCoord(dim()), dstCoord(dim());
        _TensorBase_t<T> newTensor(dstShape);

        do {
            for (int i = 0; i < dim(); i++)
                dstCoord[i] = srcCoord[axesMap[i]-1];
            newTensor[dstCoord] = (*this)[srcCoord];
        } while (srcCoord.next(srcShape));
        return newTensor;
    }
    _TensorBase_t<T> transpose(const char* axes) const {
        /* obtain squeeze axes */
        Array<int> axesMap;
        char s[64] = { 0 }, w[16] = { 0 };
        strRemCh(axes, s, 64, ' ');
        //strncpy(s, axes, 63);
        int n = 0;
        strSplit(s, w, 16, ",", &n);
        while (n) {
            strSplit(s, w, 16, ",", NULL);
            axesMap.append(atoi(w));
            n--;
        }
        return transpose(axesMap);
    }
    _TensorBase_t<T> transpose() const {
        /* transpose a matrix */
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (dim() != 2)
            T_error("transpose() is only used to transpose a 2D matrix, "
                "if you want to transpose tensor with higher dimension, "
                "use transpose(const char* axes).");
#endif
        TensorShape shp = shape();
        TensorShape shpT(2);
        shpT[0] = shp[1];
        shpT[1] = shp[0];
        _TensorBase_t<T> resultTensor(shpT);
        for (int i = 0; i < shp[0]; i++)
            for (int j = 0; j < shp[1]; j++)
                resultTensor._dptr[j * shpT[1] + i] = this->_dptr[i*shp[1] + j];
        return resultTensor;
    }

    /* reshape tensor data */
    _TensorBase_t<T> reshape(const TensorShape& newShape) const {
        TensorShape newShapeCopy = newShape;
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (isScalar())
            T_error("cannot reshape a scalar.");
        if (newShapeCopy.dim() < 1)
            T_error("dimension of new shape is invalid.");
        int nD = 0;
        for (int i = 0; i < newShapeCopy.dim(); i++) {
            if (newShapeCopy[i] == '-1')
                nD++;
        }
        if (nD > 1)
            T_error("cannot determine sizes of multiple dimensions.");
#endif
        /* determine new shape */
        int ax = 0, numelD = 1;
        for (int i=0; i < newShapeCopy.dim(); i++) {
            if (newShapeCopy[i] < 0)
                ax = i+1;
            else
                numelD *= newShapeCopy[i];
        }
        if (ax > 0) {
#ifdef NN_ENABLE_RUNTIME_CHECKING
            if (numelD <= 0)
                T_error("invalid target shape.");
            if (numel() % numelD != 0)
                T_error("# of elements changed after reshaping.");
#endif
            newShapeCopy[ax-1] = numel() / numelD;
        }
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (newShapeCopy.isShapeValid() == false)
            T_error("target tensor shape is invalid.");
        if (newShapeCopy.numel() != numel())
            T_error("# of elements changed after reshaping.");
#endif
        /* reshape tensor */
        _TensorBase_t<T> newTensor(*this);
        newTensor._shape = newShapeCopy;
        return newTensor;
    }
    _TensorBase_t<T> reshape(const Array<int>& newShape) const {
        return reshape(TensorShape(newShape));
    }
    _TensorBase_t<T> reshape(const char* newShape) const {
        Array<int> _newShape;
        char s[128] = { 0 }, w[64] = { 0 };
        strRemCh(newShape, s, 128, ' ');
        int n = 0;
        strSplit(s, w, 64, ",", &n);
        while (n--) {
            strSplit(s, w, 64, ",", NULL);
            if (chInStr('*', w))
                _newShape.append(-1);
            else
                _newShape.append(atoi(w));
        }
        return reshape(TensorShape(_newShape));
    }
    void reshape_(const TensorShape& newShape) {
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (newShape.isShapeValid() == false) {
            T_error("invalid tensor shape.");
        }
        if (shape().numel() != newShape.numel()) {
            T_error("cannot reshape because the new shape changes the # of elements.");
        }
#endif
        this->_shape = newShape;
    }
    void reshape_(const Array<int>& newShape) {
        return reshape_(TensorShape(newShape));
    }
    void reshape_(const char* newShape) {
        return reshape_(TensorShape(newShape));
    }

    /* mirror the data */
    _TensorBase_t<T> flip(const Array<int>& axesArr) const {
#ifdef NN_ENABLE_RUNTIME_CHECKING
        /* scalar does not support mirroring */
        if (isScalar())
            T_error("scalar does not support flipping.");
        if (dim() > 32)
            T_error("dimension too high!");
        /* check if axes are all valid */
        for (int i = 0; i < axesArr.size(); i++)
            if(axesArr[i] <= 0 || axesArr[i] > dim())
                T_error("invalid axis setting.");
#endif
        BYTE axesNeedFlip[32] = { 0 };
        for (int i = 0; i < axesArr.size(); i++)
            axesNeedFlip[axesArr[i] - 1] = 1;
        TensorCoord coordOrig(dim()), coordFlip(dim());
        TensorShape shapeOrig = shape();
        _TensorBase_t<T> flipTensor(shapeOrig);
        do {
            for (int i = 0; i < dim(); i++) {
                if (axesNeedFlip[i])
                    coordFlip[i] = shapeOrig[i] - 1 - coordOrig[i];
                else
                    coordFlip[i] = coordOrig[i];
            }
            flipTensor[coordFlip] = (*this)[coordOrig];
        } while (coordOrig.next(shapeOrig));
        return flipTensor;
    }
    _TensorBase_t<T> flip(const char* axes) const {
        /* obtain squeeze axes */
        Array<int> _axes;
        char s[64] = { 0 }, w[16] = { 0 };
        strRemCh(axes, s, 64, ' ');
        int n = 0;
        strSplit(s, w, 16, ",", &n);
        while (n) {
            strSplit(s, w, 16, ",", NULL);
            _axes.append(atoi(w));
            n--;
        }
        return flip(_axes);
    }
    _TensorBase_t<T> flip() const {
        /* flip all axes */
        Array<int> axesArr;
        for (int i = 1; i <= dim(); i++)
            axesArr.append(i);
        return flip(axesArr);
    }

    /* Padding a tensor using value "fill". Padding amount of each axis  */
    /* is indicated by "padAmount".                                      */
    /* Format is:                                                        */
    /*           [ ... , [left, right], [left, right], [left, right]]    */
    /*             ...    <= dim 3 =>    <= dim 2 =>    <= dim 1 =>      */
    /* The padding style is actually the same with PyTorch pad(). Here   */
    /* are some examples of padding a tensor when different "padAmount"  */
    /* are given:                                                        */
    /*           "4,4"     => padAmount: [2,2]     => "4,8"              */
    /*           "4,4"     => padAmount: [1,0,2,2] => "5,8"              */
    /*           "2,1,3,4" => padAmount: [1,1]     => "2,1,3,6"          */
    /*           "2,1,3,4" => padAmount: [1,0]     => "2,1,3,5"          */
    /*           "3,2,5,1" => padAmount: [1,1,2,2] => "3,2,7,5"          */
    _TensorBase_t<T> pad(const Array<int>& padAmount, const T& fill = T(0.0)) const {
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (isScalar())
            T_error("scalar does not support padding");
        if (padAmount.size() % 2 != 0)
            T_error("invalid pad amount.");
        if (dim() * 2 < padAmount.size())
            T_error("too many pad parameters given.");
#endif
        TensorShape newShape = shape();
        Array<int> _padAmount;
        int _nE = dim() - padAmount.size() / 2;
        for (int i = 0; i < dim(); i++) {
            if (i < _nE) {
                _padAmount.append(0);
                _padAmount.append(0);
            }
            else {
                _padAmount.append(padAmount[(i - _nE) * 2]);
                _padAmount.append(padAmount[(i - _nE) * 2 + 1]);
            }
        }
        for (int i = 0; i < dim(); i++) {
            newShape[i] += _padAmount[i * 2];
            newShape[i] += _padAmount[i * 2 + 1];
        }

        TensorCoord coordLow(dim()), coordHigh(dim());
        for (int i = 0; i < dim(); i++) {
            coordLow[i] = _padAmount[i * 2];
            coordHigh[i] = newShape[i] - _padAmount[i * 2 + 1];
        }

        _TensorBase_t<T> padTensor(newShape);

        /* naive implementation */
        //TensorCoord coord(dim());
        //TensorShape thisShape = shape();
        //int I = 0, J = 0;
        //do {
        //    bool oob = false; /* out of bound */
        //    for (int i = dim() - 1; i >= 0; i--) {
        //        if (coord[i] >= coordLow[i] && coord[i] < coordHigh[i])
        //            continue;
        //        else {
        //            oob = true;
        //            break;
        //        }
        //    }
        //    if (oob)
        //        padTensor[I++] = fill;
        //    else
        //        padTensor[I++] = (*this)[J++];
        //} while (coord.next(newShape));

        /* optimized for speed, ~2.6x speed up */
        padTensor.fill_(fill);
        TensorCoord coord = coordLow;
        int I = 0;
        do {
            padTensor[coord] = (*this)[I++];
        } while (coord.next(coordLow, coordHigh));

        return padTensor;
    }

    /* Bed-of-nail padding, used in back propagation                     */
    /* example:                                                          */
    /*             1   2   3           1 0 0 2 0 0 3 0 0                 */
    /*                                 0 0 0 0 0 0 0 0 0                 */
    /*                                 0 0 0 0 0 0 0 0 0                 */
    /*             4   5   6    =>     4 0 0 5 0 0 6 0 0                 */
    /*                                 0 0 0 0 0 0 0 0 0                 */
    /*                                 0 0 0 0 0 0 0 0 0                 */
    /*             7   8   9           7 0 0 8 0 0 9 0 0                 */
    /*                                 0 0 0 0 0 0 0 0 0                 */
    /*                                 0 0 0 0 0 0 0 0 0                 */
    /*                                                                   */
    /* tightEdge: true/false, the difference of them are shown below:    */
    /* tightEdge = true:     x x x -> x 0 0 x 0 0 x                      */
    /* tightEdge = false:    x x x -> x 0 0 x 0 0 x 0 0                  */
    _TensorBase_t<T> bonpad(const Array<int>& padAmount, const bool& tightEdge = false, const T& fill = T(0.0)) const {
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (padAmount.size() == 0)
            T_error("Invalid pad amount");
        for (int i = 0; i < padAmount.size(); i++) {
            if (padAmount[i] < 1)
                T_error("Invalid step size. Step size must >= 1, got %d.", padAmount[i]);
        }
        if (dim() < padAmount.size())
            T_error("Cannot pad a tensor with dimension (%d) less than padAmount.size() (%d).", dim(),
                padAmount.size());
#endif
        int D = dim(), S = padAmount.size();
        TensorShape bonShp(D);
        for (int i = 0; i < D; i++) {
            if (i < D - S) bonShp[i] = shape()[i];
            else {
                int dimStride = padAmount[i - D + S];
                if (tightEdge == false)
                    bonShp[i] = shape()[i] * dimStride;
                else
                    bonShp[i] = (shape()[i] - 1) * dimStride + 1;
            }
        }
        Array<int> steps;
        for (int i = 0; i < D; i++) {
            if (i < D - S) steps.append(1);
            else steps.append(padAmount[i - D + S]);
        }
        _TensorBase_t<T> newTensor(bonShp);
        newTensor.fill_(fill);
        TensorCoord coord(D);
        int I = 0;
        do {
            newTensor[coord] = (*this)[I++];
        } while (coord.next(bonShp, steps));

        return newTensor;
    }

    /* shrink: inverse operation of padding. The format of "shrinkAmount"*/
    /* is the same with pad(). */
    _TensorBase_t<T> shrink(const Array<int>& shrinkAmount) {
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (isScalar())
            T_error("scalar does not support shrinking");
        if (shrinkAmount.size() % 2 != 0)
            T_error("invalid pad amount.");
        if (dim() * 2 < shrinkAmount.size())
            T_error("too many pad parameters given.");
#endif
        TensorShape newShape = shape();
        Array<int> _shrinkAmount;
        int _nE = dim() - shrinkAmount.size() / 2;
        for (int i = 0; i < dim(); i++) {
            if (i < _nE) {
                _shrinkAmount.append(0);
                _shrinkAmount.append(0);
            }
            else {
                _shrinkAmount.append(shrinkAmount[(i - _nE) * 2]);
                _shrinkAmount.append(shrinkAmount[(i - _nE) * 2 + 1]);
            }
        }
        for (int i = 0; i < dim(); i++) {
            newShape[i] -= _shrinkAmount[i * 2];
            newShape[i] -= _shrinkAmount[i * 2 + 1];
#ifdef NN_ENABLE_RUNTIME_CHECKING
            if (newShape[i] <= 0)
                T_error("cannot shrink tensor because the shrink amount is "
                    "too big that causes the result tensor shape in axis "
                    "%d smaller than 1, try using a smaller shrink value.",
                    i + 1);
#endif
        }

        TensorCoord coordLow(dim()), coordHigh(dim());
        for (int i = 0; i < dim(); i++) {
            coordLow[i] = _shrinkAmount[i * 2];
            coordHigh[i] = shape()[i] - _shrinkAmount[i * 2 + 1];
        }

        _TensorBase_t<T> shrinkTensor(newShape);
        /* optimized for speed, ~2.6x speed up */
        TensorCoord coord = coordLow;
        int I = 0;
        do {
            shrinkTensor[I++] = (*this)[coord];
        } while (coord.next(coordLow, coordHigh));

        return shrinkTensor;
    }

    /* window crop                                                       */
    /* For example, imagine a 3D window is placed at the origin (0,0,0), */
    /* the window size is (3,3,3), and a 3D tensor is placed at the      */
    /* location specified by "offset" (say (-10,-10,-10)), then after    */
    /* window cropping, the tensor block at location "10:13,10:13,10:13" */
    /* is extracted and returned. All out of range locations will be     */
    /* filled with value 0.0.                                            */
    /* parameters:                                                       */
    /* =>  offset: tensor offset                                         */
    /* =>    size: window size, the dimension must equal to tensor dim.  */
    _TensorBase_t<T> window(const TensorCoord& offset, const TensorShape& size) const {
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (isScalar())
            T_error("scalar does not support window cropping.");
        if (size.isShapeValid() == false)
            T_error("invalid window size setting.");
        if (size.dim() != this->dim())
            T_error("window dimension should be equal to tensor dimension.");
        if (offset.dim() != size.dim())
            T_error("invalid offset setting.");
        if (size.numel() == 0)
            T_error("window size is 0.");
#endif
        _TensorBase_t<T> resultTensor(size);
        TensorCoord sampleStart = -offset;
        TensorCoord sampleEnd = -offset + size;
        TensorCoord srcCoord = sampleStart, dstCoord;
        T defVal = T(0.0);
        do {
            dstCoord = srcCoord + offset;
            resultTensor[dstCoord] = this->sample(srcCoord, defVal);
        } while (srcCoord.next(sampleStart, sampleEnd));
        return resultTensor;
    }
    _TensorBase_t<T> window(const char* offset, const char* size) const {
        return window(TensorCoord(offset), TensorShape(size));
    }


    /* tensor broadcasting, expand a small tensor to a large tensor      */
    /* >> The current implementation of tensor broadcasting is slow, so  */
    /*    please avoid using it as much as possible.                     */
    /* For example, broadcast a tensor with shape (4,5) to a tensor with */
    /* shape (4,6,5), call like this:                                    */
    /* this->broadcast(TensorShape("4,6,5"),T_buildArray<2,int>({1,3})); */
    /* bcastShape: shape after broadcast.                                */
    /* axesPos: axes position mapping. For the above example, since the  */
    /* first axis remains the same after broadcast and the second axis   */
    /* becomes the third axis in the new tensor, we pass an array with   */
    /* value {1,3} to the function. Here is a more complicated example:  */
    /* >> broadcast a tensor with shape (4,5,6) to a new tensor with     */
    /*    shape (1,2,4,3,5,7,6).                                         */
    /*               ^   ^   ^   note the new axes positions             */
    /*    pos:   1 2 3 4 5 6 7                                           */
    /* Here is the function call:                                        */
    /* >> this->broadcast(TensorShape("1,2,4,3,5,7,6"),                  */
    /*                    T_buildArray<3,int>({3,5,7}));                 */
    /* Note that you cannot alter the axis order during broadcasting     */
    _TensorBase_t<T> broadcast(const TensorShape& bcastShape, const Array<int> axesPos) {
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (isScalar())
            T_error("scalar does not support broadcasting.");
        if (bcastShape.isShapeValid() == false)
            T_error("invalid target shape.");
        if (dim() != axesPos.size())
            T_error("invalid axes mapping when calling broadcast(), dimension not equal.");
        if (bcastShape.dim() > 32 || dim() > 32)
            T_error("tensor dimension too high! try reduce dimension to be less than 32.");
        for (int i = 0; i < axesPos.size(); i++) {
            if (axesPos[i] < 1 || axesPos[i] > bcastShape.dim())
                T_error("axes position out of range, accepted range is [%d~%d].", 1, bcastShape.dim());
            if (i > 0 && axesPos[i] <= axesPos[i - 1])
                T_error("invalid axes position setting, cannot alter/copy axis order during broadcasting.");
            if (shape()[i] != bcastShape[axesPos[i] - 1])
                T_error("invalid axes position setting, tried to map axis %d (size=%d) to "
                    "axis %d (size=%d), these two sizes are not equal.",
                    i + 1, shape()[i], axesPos[i], bcastShape[axesPos[i] - 1]);
        }
#endif
        UINT mask = 0;
        for (UINT ax = UINT(bcastShape.dim()), s = 0x1u; ax > 0; ax--, s <<= 1) {
            bool record = false;
            for (int t = 0; t < axesPos.size(); t++)
                if (axesPos[t] == ax)
                    record = true;
            if (record)
                mask |= s;
        }

        _TensorBase_t bcastTensor(bcastShape);
        TensorCoord c(bcastShape.dim());
        int I = 0;
        do {
            bcastTensor[I++] = (*this)[c.toAddr(bcastShape, mask)];
        } while (c.next(bcastShape));

        return bcastTensor;
    }
    
public:
    /* memory management */
    void purge() {
        if (this->_dptr) {
            free(this->_dptr);
            _dptr = NULL;
        }
        _shape.fromStr("");
    }
    void moveStorage(_TensorBase_t& tensor) {
        /* move storage to another tensor without allocating memory to save time */
        tensor.purge();
        if (this->isNull()) {
            return;
        }
        else {
            tensor._dptr = this->_dptr;
            tensor._shape = this->_shape;
            this->_dptr = NULL;
            this->_shape.clear();
        }
    }

protected:
    /* internal memory management */
    void _create(const TensorShape shp) {
        purge();
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (shp.isShapeValid() == false) {
            char buf[256];
            shp.toStr(buf);
            T_error("error, invalid shape is given (\"%s\") when creating tensor object.\n", buf);
            return;
        }
#endif
        int _numel = 1;
        for (int i = 0; i < shp.dim(); i++)
            _numel *= shp[i];
        int bytes = sizeof(T) * _numel;
        /* alloc space */
        this->_dptr = (T*)malloc(bytes);
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (!(this->_dptr)) {
            char buf[256];
            _shape.toStr(buf);
            int mb = int(bytes / 1024.0 / 1024.0);
            T_error("Cannot create tensor object with shape (%s). Failed to "
                "allocate %d bytes of memory (%dMB).", buf, bytes, bytes, mb);
            purge();
        }
#endif
        this->_shape = shp;
    }
    void _copy(const _TensorBase_t& that) {
        if (this == &that)
            return;
        if (that._dptr == NULL) {
            purge();
            return;
        }
        if (!(this->isShapeEq(that))) {
            purge();
            _create(that._shape);
        }
        /* now the two tensors have the same shape, direct copy data */
        int bytes = this->numel() * sizeof(T);
        memcpy(this->_dptr, that._dptr, bytes); /* direct memory copy to save huge amount of time */
    }

protected:
    /* misc */
    bool _parseSlicing(const char* slicing, 
        TensorCoord& coordStart, TensorCoord& coordEnd) const
    {
        char s[128] = { 0 }, /* whole slicing "1:2,3:6" */
            w[64] = { 0 },   /*  word "3:6" */
            b[32] = { 0 },   /* begin "3"   */
            e[32] = { 0 };   /*   end "6"   */
        TensorShape tensorShape = shape();

        /* remove space character */
        char _slicing[128] = { 0 };
        strRemCh(slicing, _slicing, 128, ' ');

        strncpy(s, _slicing, 127);
        int n = 0;
        strSplit(s, w, 64, ",", &n);
        if (isScalar()) {
            if (n != 1)
                return false;
            strncpy(s, _slicing, 127);
            coordStart = TensorCoord(1);
            coordEnd = TensorCoord(1);
            /* parse scalar slicing here */
            strSplit(s, b, 32, ":", NULL);
            strSplit(s, e, 32, ":", NULL);
            int ib, ie;
            /* parse ib */
            if (strlen(b) == 0) ib = 0;
            else ib = atoi(b);
            /* parse ie */
            if (strlen(e) == 0) ie = 1;
            else ie = atoi(e);
            /* check */
            if (ib != 0 || ie != 1)
                return false;
            /* fill_ coord */
            coordStart[0] = ib;
            coordEnd[0] = ie;
        }
        else {
            if (n != tensorShape.dim())
                return false;
            strncpy(s, _slicing, 127);
            coordStart = TensorCoord(tensorShape.dim());
            coordEnd = TensorCoord(tensorShape.dim());
            int i = 0;
            while (n) {
                int ib, ie;
                strSplit(s, w, 64, ",", NULL);
                if (chInStr(':', w) == false) {
                    strSplit(w, b, 32, ":", NULL);
                    /* parse ib */
                    if (strlen(b) == 0) ib = 0;
                    else {
                        ib = atoi(b);
                        if (ib < 0 || ib >= tensorShape[i])
                            return false;
                    }
                    ie = ib + 1;
                    if (ie < 0 || ie > tensorShape[i])
                        return false;
                }
                else {
                    strSplit(w, b, 32, ":", NULL);
                    strSplit(w, e, 32, ":", NULL);
                    /* parse ib */
                    if (strlen(b) == 0) ib = 0;
                    else {
                        ib = atoi(b);
                        if (ib < 0 || ib >= tensorShape[i])
                            return false;
                    }
                    /* parse ie */
                    if (strlen(e) == 0) ie = tensorShape[i];
                    else {
                        ie = atoi(e);
                        if (ie < 0 || ie > tensorShape[i])
                            return false;
                    }
                }
                /* check */
                if (ib >= ie)
                    return false;
                /* fill_ coord */
                coordStart[i] = ib;
                coordEnd[i] = ie;
                /* advance */
                n--;
                i++;
            }
        }
        return true;
    }
    bool _checkCoordRange(const TensorCoord& coordStart, 
        const TensorCoord& coordEnd) const {
        /* range checking */
        if (coordStart.dim() > 30) {
#ifdef NN_ENABLE_RUNTIME_CHECKING
            T_error("Dimensions too large (>30)!\n");
#endif
            return false;
        }
        if (!_shape.rangeCheckAt(coordStart)) {
#ifdef NN_ENABLE_RUNTIME_CHECKING
            char p[256], s[256];
            coordStart.toStr(p);
            _shape.toStr(s);
            T_error("Start position \"%s\" is out of range. Tensor shape is \"%s\".\n", p, s);
#endif
            return false;
        }
        if (!_shape.rangeCheckEnd(coordEnd)) {
#ifdef NN_ENABLE_RUNTIME_CHECKING
            char p[256], s[256];
            coordEnd.toStr(p);
            _shape.toStr(s);
            T_error("End position \"%s\" is out of range. Tensor shape is \"%s\".\n", p, s);
#endif
            return false;
        }
        for (int t = 0; t < coordStart.dim(); t++) {
            if (coordStart[t] >= coordEnd[t]) {
#ifdef NN_ENABLE_RUNTIME_CHECKING
                char s[256], e[256];
                coordStart.toStr(s);
                coordEnd.toStr(e);
                T_error("Invalid position (empty dimension). Start: %s, end: %s.\n", s, e);
#endif
                return false;
            }
        }
        return true;
    }

public:

    /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
    /* here i defined the basic math operations for _TensorBase_t class  */
    /*       for simplicity i omit the typename "<T>" from now on        */
    /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

    /* (+) tensor add, support broadcasting */
    _TensorBase_t add(const _TensorBase_t & that) const {
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (T_shapeEqBcast(this->shape(), that.shape()) == false) {
            char a[64], b[64];
            this->shape().toStr(a);
            that.shape().toStr(b);
            T_error("cannot add tensors with different sizes (\"%s\", \"%s\").", a, b);
        }
#endif
        if (this->shape() == that.shape()) {
            /* element-wise adding */
            int _numel = numel();
            _TensorBase_t newTensor(shape());
            for (int i = 0; i < _numel; i++)
                newTensor._dptr[i] = this->_dptr[i] + that._dptr[i];
            return newTensor;
        }
        else {
            /* broadcasting */
            UINT mask1 = _T_buildMaskFromShape(this->shape());
            UINT mask2 = _T_buildMaskFromShape(that.shape());
            _TensorBase_t newTensor(_T_maxShape(this->shape(), that.shape()));
            TensorCoord dstCoord(newTensor.dim());
            TensorShape s = newTensor.shape();
            int I = 0;
            do {
                newTensor[I++] = (*this)[dstCoord.toAddr(s, mask1)] + that[dstCoord.toAddr(s, mask2)];
            } while (dstCoord.next(s));
            return newTensor;
        }
    }
    void add_(const _TensorBase_t & that) /* inplace add does not (and should not) support broadcasting */
    {
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (this->isShapeEq(that) == false)
            T_error("cannot add tensors with different sizes.");
#endif
        int _numel = numel();
        for (int i = 0; i < _numel; i++)
            this->_dptr[i] += that._dptr[i];
    }
    _TensorBase_t add(const T& value) const {
        int _numel = numel();
        _TensorBase_t newTensor(shape());
        for (int i = 0; i < _numel; i++)
            newTensor._dptr[i] = this->_dptr[i] + value;
        return newTensor;
    }
    void add_(const T& value)
    {
        int _numel = numel();
        for (int i = 0; i < _numel; i++)
            this->_dptr[i] += value;
    }
    _TensorBase_t operator+(const _TensorBase_t& that) const { return add(that); }
    void operator+=(const _TensorBase_t & that)
    {
        return this->add_(that);
    }
    _TensorBase_t operator+(const T& value) const { return add(value); }
    void operator+=(const T& value)
    {
        return this->add_(value);
    }
    friend _TensorBase_t operator+(const T& value, const _TensorBase_t& tensor) {
        return tensor.add(value);
    }

    /* - */
    _TensorBase_t sub(const _TensorBase_t & that) const {
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (this->isShapeEq(that) == false)
            T_error("cannot sub tensors with different sizes.");
#endif
        int _numel = numel();
        _TensorBase_t newTensor(shape());
        for (int i = 0; i < _numel; i++)
            newTensor._dptr[i] = this->_dptr[i] - that._dptr[i];
        return newTensor;
    }
    void sub_(const _TensorBase_t & that)
    {
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (this->isShapeEq(that) == false)
            T_error("cannot sub tensors with different sizes.");
#endif
        int _numel = numel();
        for (int i = 0; i < _numel; i++)
            this->_dptr[i] -= that._dptr[i];
    }
    _TensorBase_t sub(const T& value) const {
        int _numel = numel();
        _TensorBase_t newTensor(shape());
        for (int i = 0; i < _numel; i++)
            newTensor._dptr[i] = this->_dptr[i] - value;
        return newTensor;
    }
    void sub_(const T& value)
    {
        int _numel = numel();
        for (int i = 0; i < _numel; i++)
            this->_dptr[i] -= value;
    }
    _TensorBase_t operator-(const _TensorBase_t& that) const { return sub(that); }
    void operator-=(const _TensorBase_t & that)
    {
        return this->sub_(that);
    }
    _TensorBase_t operator-(const T& value) const { return sub(value); }
    void operator-=(const T& value)
    {
        return this->sub_(value);
    }
    friend _TensorBase_t operator-(const T& value, const _TensorBase_t& tensor) {
        return (-tensor) + value;
    }
    _TensorBase_t operator-() const {
        int _numel = numel();
        _TensorBase_t newTensor(shape());
        for (int i = 0; i < _numel; i++)
            newTensor._dptr[i] = -this->_dptr[i];
        return newTensor;
    }

    /* * */
    _TensorBase_t hprod(const _TensorBase_t & that) const {
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (this->isShapeEq(that) == false)
            T_error("cannot multiply tensors element-wise with different sizes.");
#endif
        int _numel = numel();
        _TensorBase_t newTensor(shape());
        for (int i = 0; i < _numel; i++)
            newTensor._dptr[i] = this->_dptr[i] * that._dptr[i];
        return newTensor;
    }
    _TensorBase_t mul(const _TensorBase_t & that) const {
        return hprod(that);
    }
    void mul_(const _TensorBase_t & that)
    {
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (this->isShapeEq(that) == false)
            T_error("cannot multiply tensors element-wise with different sizes.");
#endif
        int _numel = numel();
        for (int i = 0; i < _numel; i++)
            this->_dptr[i] *= that._dptr[i];
    }
    _TensorBase_t mul(const T& value) const
    {
        int _numel = numel();
        _TensorBase_t newTensor(shape());
        for (int i = 0; i < _numel; i++)
            newTensor._dptr[i] = this->_dptr[i] * value;
        return newTensor;
    }
    void mul_(const T& value)
    {
        int _numel = numel();
        for (int i = 0; i < _numel; i++)
            this->_dptr[i] *= value;
    }
    _TensorBase_t operator*(const _TensorBase_t& that) const {
        return hprod(that);
    }
    void operator*=(const _TensorBase_t & that)
    {
        this->mul_(that);
    }
    _TensorBase_t operator*(const T& value) const {
        int _numel = numel();
        _TensorBase_t newTensor(shape());
        for (int i = 0; i < _numel; i++)
            newTensor._dptr[i] = this->_dptr[i] * value;
        return newTensor;
    }
    void operator*=(const T& value)
    {
        this->mul_(value);
    }
    friend _TensorBase_t operator*(const T& value, const _TensorBase_t& tensor) {
        return tensor * value;
    }

    /* / */
    _TensorBase_t div(const T& value) const {
        int _numel = numel();
        _TensorBase_t newTensor(shape());
        T inv_value = T(1.0) / value;
        for (int i = 0; i < _numel; i++)
            newTensor._dptr[i] = this->_dptr[i] * inv_value;
        return newTensor;
    }
    void div_(const T& value) {
        int _numel = numel();
        T inv_value = T(1.0) / value;
        for (int i = 0; i < _numel; i++)
            this->_dptr[i] *= inv_value;
    }
    _TensorBase_t operator/(const T& value) const {
        return div(value);
    }
    void operator/=(const T& value) {
        div_(value);
    }

    /* >, >= */
    _TensorBase_t operator>(const T& value) {
        _TensorBase_t newTensor = *this;
        int _numel = numel();
        for (int i = 0; i < _numel; i++)
            newTensor[i] = ((*this)[i] > value) ? T(1.0) : T(0.0);
        return newTensor;
    }
    _TensorBase_t operator>=(const T& value) {
        _TensorBase_t newTensor = *this;
        int _numel = numel();
        for (int i = 0; i < _numel; i++)
            newTensor[i] = ((*this)[i] >= value) ? T(1.0) : T(0.0);
        return newTensor;
    }
    /* > */
    _TensorBase_t operator<(const T& value) {
        _TensorBase_t newTensor = *this;
        int _numel = numel();
        for (int i = 0; i < _numel; i++)
            newTensor[i] = ((*this)[i] < value) ? T(1.0) : T(0.0);
        return newTensor;
    }
    _TensorBase_t operator<=(const T& value) {
        _TensorBase_t newTensor = *this;
        int _numel = numel();
        for (int i = 0; i < _numel; i++)
            newTensor[i] = ((*this)[i] <= value) ? T(1.0) : T(0.0);
        return newTensor;
    }

    T sum() const {
        T _sum = T(0.0);
        int _numel = numel();
        for (int i = 0; i < _numel; i++)
            _sum += _dptr[i];
        return _sum;
    }
    _TensorBase_t sum(const Array<int>& axes) const {
#ifdef NN_ENABLE_RUNTIME_CHECKING
        for (int i = 0; i < axes.size(); i++) {
            int axis = axes[i];
            if (isScalar() || axis < 1 || axis > dim())
                T_error("Invalid axis setting. Check if this tensor is a scalar or "
                    "axis value is out of range.");
            for (int j = i + 1; j < axes.size(); j++) {
                if (axes[j] == axis)
                    T_error("Invalid axis setting. Cannot sum the same axis twice.");
            }
        }
#endif
        if (dim() == axes.size())
            return _TensorBase_t(sum());

        /* build mask */
        UINT m = 0;
        for (UINT i = 0, s = 0x1u; i < UINT(dim()); i++, s <<= 1) {
            bool reduce = false;
            for (int j = 0; j < axes.size(); j++)
                if (dim() - axes[j] == i)
                    reduce = true;
            if (!reduce)
                m |= s;
        }

        TensorShape resultShape(dim() - axes.size());
        for (int i = 0, j = 0; i < dim(); i++) {
            bool reduce = false;
            for (int t = 0; t < axes.size(); t++)
                if (axes[t] - 1 == i)
                    reduce = true;
            if (!reduce)
                resultShape[j++] = shape()[i];
        }

        _TensorBase_t sumTensor(resultShape);
        sumTensor.fill_(T(0.0));

        TensorCoord c(dim());
        TensorShape s = shape();

        int I = 0;
        do {
            sumTensor[c.toAddr(s, m)] += (*this)[I++];
        } while (c.next(s));

        return sumTensor;
    }
    _TensorBase_t sum(const int& axis) const {
        return sum(T_buildArray<1, int>({ axis }));        
    }
    T max() const {
        int _numel = numel();
        T _max = _dptr[0];
        for (int i = 1; i < _numel; i++) {
            if (_max < _dptr[i])
                _max = _dptr[i];
        }
        return _max;
    }
    T min() const {
        int _numel = numel();
        T _min = _dptr[0];
        for (int i = 1; i < _numel; i++) {
            if (_min > _dptr[i])
                _min = _dptr[i];
        }
        return _min;
    }
    
    T mean() const {
        T _sum = T(0.0);
        int _numel = numel();
        for (int i = 0; i < _numel; i++)
            _sum += _dptr[i];
        return T(_sum / _numel);
    }
    _TensorBase_t mean(const int& axis) const {
        return mean(T_buildArray<1, int>({ axis }));
    }
    _TensorBase_t mean(const Array<int>& axes) const {
#ifdef NN_ENABLE_RUNTIME_CHECKING
        for (int i = 0; i < axes.size(); i++) {
            int axis = axes[i];
            if (isScalar() || axis < 1 || axis > dim())
                T_error("Invalid axis setting. Check if this tensor is a scalar or "
                    "axis value is out of range.");
            for (int j = i + 1; j < axes.size(); j++) {
                if (axes[j] == axis)
                    T_error("Invalid axis setting. Axis can only occur once.");
            }
        }
#endif
        int sum_numel = 1;
        for (int ax = 0; ax < axes.size(); ax++)
            sum_numel *= shape()[axes[ax] - 1];
        return sum(axes) / T(sum_numel);
    }

    /* variance */
    T var(const bool& unbiased = true) const {
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (numel() < 2) {
            T_error("cannot calculate variance for tensor as number of elements is "
                "less than 2.");
        }
#endif
        int n = (unbiased ? (numel() - 1) : numel());
        _TensorBase_t d = (*this) - mean();
        return d.hprod(d).sum() / T(n);
    }
    _TensorBase_t var(const Array<int>& axes, const bool& unbiased = true) const {
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (numel() < 2) {
            T_error("cannot calculate variance for tensor as number of elements is "
                "less than 2.");
        }
        for (int i = 0; i < axes.size(); i++) {
            int axis = axes[i];
            if (isScalar() || axis < 1 || axis > dim())
                T_error("Invalid axis setting. Check if this tensor is a scalar or "
                    "axis value is out of range.");
            for (int j = i + 1; j < axes.size(); j++) {
                if (axes[j] == axis)
                    T_error("Invalid axis setting. Axis can only occur once.");
            }
        }
#endif
        int var_numel = 1;
        for (int ax = 0; ax < axes.size(); ax++)
            var_numel *= shape()[axes[ax] - 1];
        if (unbiased)
            var_numel--;
        //..................................
    }


public: /* tensor quick build helper */
    static _TensorBase_t zero(const TensorShape& shape) {
        _TensorBase_t tensor(shape);
        tensor.fill_(T(0.0));
        return tensor;
    }
    static _TensorBase_t one(const TensorShape& shape) {
        _TensorBase_t tensor(shape);
        tensor.fill_(T(1.0));
        return tensor;
    }

public:
    /*
    tensor saving format:
    +-------------------------------------------------------------------------+
    | magic string | description | element size | tensor shape | element data |
    +-------------------------------------------------------------------------+
    magic string : const char [16]  = "@TensorSaveData"
    description  : const char [TENSOR_DESCRIPTION_LENGTH] = <tensor descriptions>
    element size : (const int) 4, 8
    tensor shape : const int [16] (unused=-1, for example, 5,6,7,-1,...,-1)
    element data : void*
    */
    bool save(const char* file, const char* tensorDesc) const {
        FILE* fp = NULL;
        if ((fp = fopen(file, "wb")) == NULL)
            return false;

        char buffer[TENSOR_DESCRIPTION_LENGTH];
        const char* magicString = "@TensorSaveData";
        int elemSize = sizeof(T);
        int elemCount = numel();
        TensorShape tensorShape = shape();

        if (tensorDesc == NULL || strlen(tensorDesc) > TENSOR_DESCRIPTION_LENGTH-1)
            goto save_failed;
        if (dim() > 16)
            goto save_failed;

        /* write magic string */
        memset(buffer, 0, TENSOR_DESCRIPTION_LENGTH);
        memcpy(buffer, magicString, strlen(magicString));
        if (fwrite(buffer, 1, 16, fp) != 16)
            goto save_failed;

        /* write description */
        memset(buffer, 0, TENSOR_DESCRIPTION_LENGTH);
        memcpy(buffer, tensorDesc, strlen(tensorDesc));
        if (fwrite(buffer, 1, TENSOR_DESCRIPTION_LENGTH, fp) != TENSOR_DESCRIPTION_LENGTH)
            goto save_failed;

        /* write element size and count */
        fwrite(&elemSize, sizeof(int), 1, fp);
        int s = tensorShape[0];
        if (fwrite(&s, sizeof(int), 1, fp) != 1)
            goto save_failed;
        for (int d = 1; d < 16; d++) {
            s = -1;
            if (d < dim()) 
                s = tensorShape[d];
            if (fwrite(&s, sizeof(int), 1, fp) != 1)
                goto save_failed;
        }

        /* write data */
        for (int i = 0; i < elemCount; i++) {
            T value = (*this)[i];
            if (fwrite(&value, elemSize, 1, fp) != 1)
                goto save_failed;
        }

        fclose(fp);
        return true;

    save_failed:
        fclose(fp);
        return false;
    }
    bool save(Array<BYTE>& byteArray, const char* tensorDesc) const {

        char buffer[TENSOR_DESCRIPTION_LENGTH];
        const char* magicString = "@TensorSaveData";
        int elemSize = sizeof(T);
        int elemCount = numel();
        TensorShape tensorShape = shape();

        if (tensorDesc == NULL || strlen(tensorDesc) > TENSOR_DESCRIPTION_LENGTH - 1)
            return false;
        if (dim() > 16)
            return false;

        /* write magic string */
        memset(buffer, 0, TENSOR_DESCRIPTION_LENGTH);
        memcpy(buffer, magicString, strlen(magicString));
        for (int i = 0; i < 16; i++)
            byteArray.append(buffer[i]);

        /* write description */
        memset(buffer, 0, TENSOR_DESCRIPTION_LENGTH);
        memcpy(buffer, tensorDesc, strlen(tensorDesc));
        for (int i = 0; i < TENSOR_DESCRIPTION_LENGTH; i++)
            byteArray.append(buffer[i]);

        /* write element size and count */
        byteArray.append(valueAsByteArray(elemSize));

        int s = tensorShape[0];
        byteArray.append(valueAsByteArray(s));

        for (int d = 1; d < 16; d++) {
            s = -1;
            if (d < dim())
                s = tensorShape[d];
            byteArray.append(valueAsByteArray(s));
        }

        /* write data */
        for (int i = 0; i < elemCount; i++) {
            T value = (*this)[i];
            byteArray.append(valueAsByteArray(value));
        }

        return true;

    }
    bool load(const char* file) {

        FILE* fp = NULL;
        if ((fp = fopen(file, "rb")) == NULL)
            return false;

        Array<int> shapeArray;
        TensorShape tensorShape;

        char buffer[TENSOR_DESCRIPTION_LENGTH];
        memset(buffer, 0, TENSOR_DESCRIPTION_LENGTH);
        if (fread(buffer, 1, 16, fp) != 16) goto load_failed;
        if (strcmp(buffer, "@TensorSaveData") != 0)
            goto load_failed;

        memset(buffer, 0, TENSOR_DESCRIPTION_LENGTH);
        if (fread(buffer, 1, TENSOR_DESCRIPTION_LENGTH, fp) != TENSOR_DESCRIPTION_LENGTH) goto load_failed;

        int elemSize;
        if (fread(&elemSize, 4, 1, fp) != 1) goto load_failed;
        if (elemSize != 4 && elemSize != 8)
            goto load_failed;
        if (sizeof(T) != elemSize)
            goto load_failed;

        for (int i = 0; i < 16; i++) {
            int s;
            if (fread(&s, 4, 1, fp) != 1) goto load_failed;
            if (s >= 0)
                shapeArray.append(s);
        }
        tensorShape = TensorShape(shapeArray);
        if (tensorShape.isShapeValid() == false)
            goto load_failed;

        this->_create(tensorShape);
        int _numel = this->numel();
        for (int i = 0; i < _numel; i++) {
            T value;
            if (fread(&value, sizeof(T), 1, fp) != 1)
                goto load_failed;
            else
                (*this)[i] = value;
        }
        
        fclose(fp);
        return true;

    load_failed:
        fclose(fp);
        return false;

    }
    bool load(
        const Array<BYTE>& byteArray, 
        const int offset = 0,
        Array<BYTE>* tensorDesc=NULL,
        int* nextOffset=NULL) {

        Array<int> shapeArray;
        TensorShape tensorShape;
        int _off = offset;

        char buffer[TENSOR_DESCRIPTION_LENGTH];

        /* read magic string */
        memset(buffer, 0, TENSOR_DESCRIPTION_LENGTH);
        for (int i = 0; i < 16; i++, _off++) {
            buffer[i] = byteArray[_off];
            if (_off >= byteArray.size())
                return false;
        }
        if (strcmp(buffer, "@TensorSaveData") != 0)
            return false;

        /* read description */
        for (int i = 0; i < TENSOR_DESCRIPTION_LENGTH; i++, _off++) {
            if (tensorDesc)
                tensorDesc->append(byteArray[_off]);
            if (_off >= byteArray.size())
                return false;
        }
        
        int elemSize;
        memcpy(&elemSize, &(byteArray[_off]), 4);
        _off += 4;
        if (_off >= byteArray.size())
            return false;
        if ( (elemSize != 4 && elemSize != 8) || (sizeof(T) != elemSize) )
            return false;


        for (int i = 0; i < 16; i++, _off+=4) {
            int s;
            memcpy(&s, &(byteArray[_off]), 4);
            if (_off >= byteArray.size())
                return false;
            if (s >= 0)
                shapeArray.append(s);
        }
        tensorShape = TensorShape(shapeArray);
        if (tensorShape.isShapeValid() == false)
            return false;

        this->_create(tensorShape);
        int _numel = this->numel();
        for (int i = 0; i < _numel; i++, _off += sizeof(T)) {
            T value;
            memcpy(&value, &(byteArray[_off]), sizeof(T));
            if (_off >= byteArray.size())
                return false;
            else
                (*this)[i] = value;
        }

        if (nextOffset)
            *nextOffset = _off;

        return true;

    }

public: /* functions for quick accessing */
    /* access _dptr and fill in data quickly */
    void _dptrWrite(const int& iStart, const int& count, const void* data) {
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (iStart <0 || iStart >= numel() ||
            count <= 0 || iStart + count > numel()) {
            T_error("_dptrWrite() error: dst index out of range!");
            return;
        }
#endif
        int bytes = sizeof(T) * count;
        T* dstAddr = &(_dptr[iStart]);
        const T* srcAddr = (const T*)data;
#ifdef NN_ENABLE_OPENMP
#pragma omp critical(Tensor_dptrWrite) /* strongly recommend using "named" critical */
#endif
        memcpy(dstAddr, srcAddr, bytes);
    }
    void _dptrWrite(const int& iStart, const int& count, const _TensorBase_t& tensor, const int& iOffset = 0) {
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (iStart <0 || iStart >= numel() ||
            count <= 0 || iStart + count > numel()) {
            T_error("_dptrWrite() error: dst index out of range! Trying to write into index %d with length %d but total length is %d.", iStart, count, numel());
            return;
        }
        if (iOffset <0 || iOffset >= tensor.numel() || iOffset + count > tensor.numel()) {
            T_error("_dptrWrite() error: src index out of range!");
            return;
        }
#endif
        int bytes = sizeof(T) * count;
        T* dstAddr = &(_dptr[iStart]);
        T* srcAddr = &(tensor._dptr[iOffset]);
#ifdef NN_ENABLE_OPENMP
#pragma omp critical(Tensor_dptrWrite)
#endif
        memcpy(dstAddr, srcAddr, bytes);
    }
    /* checking if the content of two tensors are exactly the same */
    bool _memcmp(const _TensorBase_t& tensor) {
        if (numel() != tensor.numel())
            return false;
        int bytes = sizeof(T) * numel();
        if (memcmp(_dptr, tensor._dptr, bytes) != 0)
            return false;
        else
            return true;
    }
    const T* dptr() const { return _dptr; }
    unsigned long long nbytes() const {
        return sizeof(T) * numel();
    }
};

/* create a tensor using range */
template <typename T>
_TensorBase_t<T> T_buildTensor(const T& start, const T& step, const char* shape) {
    TensorShape shp(shape);
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (shp.isShapeValid() == false || shp.dim() == 0)
        T_error("dimension is too small or target tensor shape is not vald.");
#endif
    _TensorBase_t<T> tensor(shp);
    TensorCoord I(tensor.dim());
    T i = start;
    do {
        tensor[I] = i;
        i += step;
    } while (I.next(shp));
    return tensor;
}

template <typename T>
_TensorBase_t<T> T_hprod(const _TensorBase_t<T> & tsr1, const _TensorBase_t<T> & tsr2){
    return tsr1.hprod(tsr2);
}
template <typename T>
T T_sum(const _TensorBase_t<T> & tsr) {
    return tsr.sum();
}

template <typename T>
_TensorBase_t<T> T_stack(const _TensorBase_t<T> & tsr, const int n)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (tsr.isScalar())
        T_error("cannot stack a scalar, please convert it to a tensor with "
            "dimension be at least 1");
    if (n <= 0)
        T_error("invalid parameter \"n\" when calling T_stack()");
#endif
    TensorShape stackShape(tsr.dim() + 1);
    stackShape[0] = n;
    for (int i = 0; i < tsr.dim(); i++)
        stackShape[i + 1] = tsr.shape()[i];
    _TensorBase_t<T> stackedTensor(stackShape);
    int _numel = tsr.numel();
    for (int i = 0; i < n; i++)
        stackedTensor._dptrWrite(i*_numel, _numel, tsr);
    return stackedTensor;
}

template <typename T>
void T_reverse(T* start, const int & n)
{
    int i = 0, j = n - 1;
    while (i < j) {
        T t = start[i];
        start[i] = start[j];
        start[j] = t;
        i++; j--;
    }
}

template <typename T>
_TensorBase_t<T> T_swapBatchChannel(const _TensorBase_t<T>& tsr) {
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (tsr.dim() < 2)
        T_error("T_swapBatchChannel() expects a tensor with dimension >= 2, got %d.", tsr.dim());
#endif

    if (tsr.dim() == 2) {
        /* matrix */
        return tsr.transpose();
    }
    else {
        int batch = tsr.shape()[0];
        int channels = tsr.shape()[1];
        int stride = 1;
        Array<int> newShape;
        newShape.append(tsr.shape()[1]);
        newShape.append(tsr.shape()[0]);
        for (int d = 2; d < tsr.dim(); d++) {
            stride *= tsr.shape()[d];
            newShape.append(tsr.shape()[d]);
        }
        _TensorBase_t<T> newTensor = _TensorBase_t<T>(TensorShape(newShape));
        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < channels; c++) {
                int iSrcOffset = stride * (b * channels + c);
                int iDestStart = stride * (c * batch + b);
                newTensor._dptrWrite(iDestStart, stride, tsr, iSrcOffset);
            }
        }
        return newTensor;
    }
}

template <typename T>
class SerializedTensors {
protected:
    Array<BYTE> rawBytes;
    Array<Array<BYTE>> tensorDescriptions;
    Array<int> tensorOffsets;
    int currentOffset;
public:
    bool load(const char* file) {
        /* load file */
        Array<BYTE> byteArray = loadAsByteArray(file);
        if (byteArray.size() == 0) {
            T_error("NN error: SerializedTensors::load() returned empty byte array.");
            return false;
        }
        int offset = rawBytes.size(), nextOffset;
        for (int i = 0; i < byteArray.size(); i++)
            rawBytes.append(byteArray[i]);
        /* parse newly added tensors */
        while (true) {
            _TensorBase_t<T> tensor;
            Array<BYTE> tensorDesc;
            if (tensor.load(rawBytes, offset, &tensorDesc, &nextOffset) == false)
                break;
            else {
                tensorDescriptions.append(tensorDesc);
                tensorOffsets.append(offset);
            }
            offset = nextOffset;
        }
        return true;
    }
    bool save(const char* file) const {
        /* save serialized tensors to disk */
        if (saveByteArray(rawBytes, file) == false) {
            T_error("Error, cannot save serialized tensor(s) to file \"%s\".", file);
            return false;
        }
        return true;
    }
    bool append(const SerializedTensors<T>& ser) {
        int baseOffset = rawBytes.size();
        rawBytes.append(ser.rawBytes);
        tensorDescriptions.append(ser.tensorDescriptions);
        for (int i = 0; i < ser.tensorOffsets.size(); i++) {
            tensorOffsets.append(ser.tensorOffsets[i] + baseOffset);
        }
        return true;
    }
    bool serializeTensor(const _TensorBase_t<T>& tensor, const char* tensorDesc) { 
        Array<BYTE> byteArray;
        if (tensor.save(byteArray, tensorDesc) == false) {
            T_error("Error, cannot serialize tensor.");
            return false;
        }
        tensorOffsets.append(rawBytes.size());
        Array<BYTE> desc;
        char _desc[TENSOR_DESCRIPTION_LENGTH] = { 0 };
        memcpy(_desc, tensorDesc, strlen(tensorDesc));
        for (int i = 0; i < TENSOR_DESCRIPTION_LENGTH; i++)
            desc.append(_desc[i]);
        tensorDescriptions.append(desc);
        rawBytes.append(byteArray);
        return true;
    }
    bool serializeTensor(const _TensorBase_t<T>& tensor) {
        return serializeTensor(tensor, "");
    }
    _TensorBase_t<T> deserializeTensor(const char* tensorDesc) const {
        char szTensorDesc[TENSOR_DESCRIPTION_LENGTH];
        int tensorStorageId = -1;
        _TensorBase_t<T> resultTensor;
        for (int I = 0; I < tensorOffsets.size(); I++) {
            for (int i = 0; i < TENSOR_DESCRIPTION_LENGTH; i++)
                szTensorDesc[i] = tensorDescriptions[I][i];
            if (strcmp(szTensorDesc, tensorDesc) == 0) {
                tensorStorageId = I;
                break;
            }
        }
        if (tensorStorageId < 0)
            T_error("Error, cannot load tensor with description \"%s\".", tensorDesc);
        else
            resultTensor.load(rawBytes, tensorOffsets[tensorStorageId]);
        return resultTensor;
    }
    _TensorBase_t<T> deserializeTensor() {
        _TensorBase_t<T> resultTensor;
        if (currentOffset < rawBytes.size()) {
            int nextOffset;
            resultTensor.load(rawBytes, currentOffset, NULL, &nextOffset);
            currentOffset = nextOffset;
        }
        else
            T_error("Error, cannot deserialize any tensor because byte stream ended.");
        return resultTensor;
    }
public:
    SerializedTensors() { currentOffset = 0; }
    void purge() {
        currentOffset = 0;
        rawBytes.clear();
        tensorOffsets.clear();
        tensorDescriptions.clear();
    }
    void resetOffset() { currentOffset = 0; }
};

template<typename T>
bool T_save(const SerializedTensors<T>& ser, const char* file) {
    return ser.save(file);
}

template<typename T>
SerializedTensors<T> T_load(const char* file) {
    SerializedTensors<T> ser;
    ser.load(file);
    return ser;
}


