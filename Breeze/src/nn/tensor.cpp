#include "nn/tensor.h"

void T_error(const char * format, ...)
{
    char fmsg[512] = { 0 };
    va_list args;
    va_start(args, format);
    vsprintf(fmsg, format, args);
    va_end(args);
    /* now the string is formatted, we can choose a method to let */
    /* the user know the information */
    printf("%s", fmsg);
#ifdef NN_DEBUG_BREAK_WHEN_ERROR
    DEBUG_BREAK;
#endif
}

/* check if the two tensors have the same shape */
bool T_shapeEq(const TensorShape & shp1, const TensorShape & shp2) {
    return shp1 == shp2;
}

/* check if two shapes can be the same using broadcasting */
bool T_shapeEqBcast(const TensorShape & shp1, const TensorShape & shp2) {
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (shp1.isShapeValid() == false || shp2.isShapeValid() == false)
        T_error("invalid shape.");
#endif
    if (shp1.dim() != shp2.dim())
        return false;
    for (int i = 0; i < shp1.dim(); i++) {
        if (shp1[i] != shp2[i]) {
            /* at least one size must be 1 */
            if (shp1[i] != 1 && shp2[i] != 1)
                return false;
        }
    }
    return true;
}
UINT _T_buildMaskFromShape(const TensorShape & shp) {
    /* shp:  [5,1,4,2] */
    /* mask:  1 0 1 1  */
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (shp.isShapeValid() == false)
        T_error("shape is invalid");
#endif
    UINT mask = 0;
    int bit = 0;
    while (bit < shp.dim()) {
        if (shp[shp.dim() - 1 - bit] != 1)
            mask |= (1 << bit);
        bit++;
    }
    return mask;
}
TensorShape _T_maxShape(const TensorShape & shp1, const TensorShape & shp2) {
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (shp1.isShapeValid() == false || shp2.isShapeValid() == false)
        T_error("invalid shape");
    if (shp1.dim() != shp2.dim())
        T_error("dimension not equal.");
#endif
    TensorShape mShape(shp1.dim());
    for (int i = 0; i < shp1.dim(); i++) {
        mShape[i] = (shp1[i] > shp2[i] ? shp1[i] : shp2[i]);
    }
    return mShape;
}

_TensorDim_t::_TensorDim_t(){}
_TensorDim_t::_TensorDim_t(const char * init)
{
    this->fromStr(init);
}
_TensorDim_t::_TensorDim_t(const int & nZ)
{
    zeros(nZ);
}
_TensorDim_t::_TensorDim_t(const Array<int>& init)
{
    clear();
    for (int i = 0; i < init.size(); i++)
        _append(init[i]);
}
_TensorDim_t::_TensorDim_t(const _TensorDim_t & that) {
    for (int i = 0; i < that._data.size(); i++) {
        _data.append(that[i]);
    }
}
_TensorDim_t::_TensorDim_t(const _TensorDim_t & start, const _TensorDim_t & end)
{
    for (int i = 0; i < start._data.size(); i++) {
        _data.append(end._data[i] - start._data[i]);
    }
}
int& _TensorDim_t::operator[](const int& index)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (index < 0 || index >= (dim() == 0 ? 1 : dim()))
        T_error("out of range.\n");
#endif
    return _data[index];
}
const int & _TensorDim_t::operator[](const int & index) const
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (index < 0 || index >= (dim() == 0 ? 1 : dim()))
        T_error("out of range.\n");
#endif
    return _data[index];
}
int _TensorDim_t::dim() const
{
    if (_data.size() > 1)
        return _data.size();
    else if (_data.size() == 1 && _data[0] == 0)
        return 0;
    else 
        return 1;
}
bool _TensorDim_t::isShapeValid() const
{
    if (_data.size() == 0)
        return false;
    else if (_data.size() == 1) {
        if (_data[0] < 0)
            return false;
    }
    else {
        int nZ = 0;
        for (int i = 0; i < _data.size(); i++) {
            if (_data[i] < 0)
                return false;
            else if (_data[i] == 0)
                nZ++;
        }
        if (nZ > 0)
            return false;
    }
    return true;
}
bool _TensorDim_t::isEqual(const _TensorDim_t & that) const
{
    if (this->dim() != that.dim() || 
        this->_data.size() == 0 || that._data.size() == 0)
        return false;
    for (int i = 0; i < dim(); i++) {
        if (this->_data[i] != that._data[i])
            return false;
    }
    return true;
}
bool _TensorDim_t::operator==(const _TensorDim_t & that) const
{
    return isEqual(that);
}
bool _TensorDim_t::operator!=(const _TensorDim_t & that) const
{
    return !isEqual(that);
}
void _TensorDim_t::fromStr(const char * shp)
{
    const int slen = 256, buflen = 64;
    clear();
    char s[slen], buf[buflen];
    strRemCh(shp, s, slen, ' ');
    //strncpy(s, shp, slen - 1);
    while (strGetWord(s, buf, buflen - 1, ",", '#')) {
        int sz = atoi(buf);
        _append(sz);
    }
}
void _TensorDim_t::toStr(char * shp) const
{
    int _dim = dim();
    if (_dim > 0) {
        /* not a scalar */
        int t = 0;
        for (int i = 0; i < dim(); i++) {
            int dimsz = _data[i];
            char value[32] = { 0 };
            itoa(dimsz, value, 10);
            for (UINT j = 0; j < UINT(strlen(value)); j++)
                shp[t++] = value[j];
            if (i != dim() - 1)
                shp[t++] = ',';
        }
        shp[t++] = '\0';
    }
    else if(_dim==0){
        /* scalar, return "0" */
        shp[0] = '0'; shp[1] = '\0';
        return;

    }
    else {
        /* _dim<0, invalid, return "" */
        shp[0] = '\0';
        return;
    }
}
void _TensorDim_t::print()
{
    char buf[64];
    toStr(buf);
    printf("%s", buf);
}
void _TensorDim_t::scalar()
{
    _data.clear();
    _data.append(0);
}
int _TensorDim_t::numel() const
{
    int _numel = 1;
    for (int i = 0; i < dim(); i++)
        _numel *= _data[i];
    return _numel;
}
//UINT _TensorDim_t::maskAddr(UINT addr, UINT m) const
//{
//    int s = dim()-1;
//    int step=1;
//    UINT maddr = 0;
//    while (s >= 0) {
//        int Q = (*this)[s];
//        int r = addr % Q;
//        addr /= Q;
//        if (m & 1) {
//            maddr += r * step;
//            step *= Q;
//        }
//        s--;
//        m >>= 1;
//    }
//    return maddr;
//}
_TensorDim_t _TensorDim_t::operator+(const _TensorDim_t & that) const
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dim() != that.dim())
        T_error("cannot add two coordinates together due to different dimensions.");
    if (dim() == 0)
        T_error("scalar does not support coordinate adding.");
#endif
    _TensorDim_t newCoord;
    for (int i = 0; i < dim(); i++)
        newCoord._append((*this)[i] + that[i]);
    return newCoord;
}
_TensorDim_t _TensorDim_t::operator-(const _TensorDim_t & that) const
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dim() != that.dim())
        T_error("cannot subtract two coordinates due to different dimensions.");
    if (dim() == 0)
        T_error("scalar does not support coordinate subtraction.");
#endif
    _TensorDim_t newCoord;
    for (int i = 0; i < dim(); i++)
        newCoord._append((*this)[i] - that[i]);
    return newCoord;
}
_TensorDim_t _TensorDim_t::operator-() const
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dim() == 0)
        T_error("scalar does not support coordinate negation.");
#endif
    _TensorDim_t newCoord;
    for (int i = 0; i < dim(); i++)
        newCoord._append(-((*this)[i]));
    return newCoord;
}
void _TensorDim_t::zeros(const int & nZ)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (nZ <= 0)
        T_error("cannot build coordinate, nZ<=0.\n");
#endif
    clear();
    for (int i = 0; i < nZ; i++)
        _append(0);
}
bool _TensorDim_t::next(const _TensorDim_t & shp)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (shp.isShapeValid() == false || this->_data.size() != shp._data.size())
        T_error("cannot advance to next position due to invalid shape or index.\n");
#endif
    int carry = 1;
    int t = shp._data.size() - 1;
    while (carry && t>=0){
        this->_data[t]++;
        if (this->_data[t] >= shp._data[t]) {
            this->_data[t] = 0; /* min */
            carry = 1;
        }
        else
            carry = 0;
        t--;
    }
    if (carry)
        return false;
    else
        return true;

/*

    UINT i = toAddr(shp);
    i++;
    int t = shp.dim() - 1;
    while (t >= 0) {
        (*this)[t] = i % shp[t];
        i /= shp[t];
        t--;
    }
    if (i > 0) return false;
    else return true;*/
}
bool _TensorDim_t::next(const _TensorDim_t & start, const _TensorDim_t & end)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (this->_data.size() != start._data.size() || this->_data.size() != end._data.size())
        T_error("cannot advance to next position due to invalid shape or index.\n");
#endif
    int carry = 1;
    int t = start._data.size() - 1;
    while (carry && t >= 0) {
        this->_data[t]++;
        if (this->_data[t] >= end._data[t]) {
            this->_data[t] = start._data[t]; /* min */
            carry = 1;
        }
        else
            carry = 0;
        t--;
    }
    if (carry)
        return false;
    else
        return true;
}
bool _TensorDim_t::next(const _TensorDim_t & shp, const Array<int>& steps)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (shp.isShapeValid() == false || this->_data.size() != shp._data.size())
        T_error("cannot advance to next position due to invalid shape or index.\n");
    if (steps.size() != shp.dim())
        T_error("cannot advance to next position due to invalid steps length.\n");
    for (int i = 0; i < steps.size(); i++) {
        if (steps[i] < 1)
            T_error("invalid step size (%d).", steps[i]);
    }
#endif
    int carry = 1;
    int t = shp._data.size() - 1;
    while (carry && t >= 0) {
        this->_data[t] += steps[t];
        if (this->_data[t] >= shp._data[t]) {
            this->_data[t] = 0; /* min */
            carry = 1;
        }
        else
            carry = 0;
        t--;
    }
    if (carry)
        return false;
    else
        return true;
}
bool _TensorDim_t::next(const _TensorDim_t & start, const _TensorDim_t & end, const Array<int>& steps)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (this->_data.size() != start._data.size() || this->_data.size() != end._data.size())
        T_error("cannot advance to next position due to invalid shape or index.\n");
    if (steps.size() != start.dim())
        T_error("cannot advance to next position due to invalid steps length.\n");
    for (int i = 0; i < steps.size(); i++) {
        if (steps[i] < 1)
            T_error("invalid step size (%d).", steps[i]);
    }
#endif
    int carry = 1;
    int t = start._data.size() - 1;
    while (carry && t >= 0) {
        this->_data[t] += steps[t];
        if (this->_data[t] >= end._data[t]) {
            this->_data[t] = start._data[t]; /* min */
            carry = 1;
        }
        else
            carry = 0;
        t--;
    }
    if (carry)
        return false;
    else
        return true;
}
bool _TensorDim_t::prev(const _TensorDim_t & shp)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (shp.isShapeValid() == false || this->_data.size() != shp._data.size())
        T_error("cannot advance to next position due to invalid shape or index.\n");
#endif
    int borrow = 1;
    int t = shp._data.size() - 1;
    while (borrow && t >= 0) {
        this->_data[t]--;
        if (this->_data[t] < 0) {
            this->_data[t] = shp._data[t]-1; /* max */
            borrow = 1;
        }
        else
            borrow = 0;
        t--;
    }
    if (borrow)
        return false;
    else
        return true;
}
void _TensorDim_t::_append(int d)
{
    _data.append(d);
}
void _TensorDim_t::clear()
{
    _data.clear();
}
bool _TensorDim_t::prev(const _TensorDim_t & start, const _TensorDim_t & end)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (this->_data.size() != start._data.size() || this->_data.size() != end._data.size())
        T_error("cannot advance to next position due to invalid shape or index.\n");
#endif
    int borrow = 1;
    int t = start._data.size() - 1;
    while (borrow && t >= 0) {
        this->_data[t]--;
        if (this->_data[t] < start._data[t]) {
            this->_data[t] = end._data[t] - 1; /* max */
            borrow = 1;
        }
        else
            borrow = 0;
        t--;
    }
    if (borrow)
        return false;
    else
        return true;
}
UINT _TensorDim_t::toAddr(const _TensorDim_t & shp) const
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (shp.isShapeValid() == false || shp.rangeCheckAt(*this) == false)
        T_error("cannot resolve absolute address for invalid "
            "shape/index descriptor.\n");
#endif
    /* resolve absolute address */
    unsigned int i = 0;
    unsigned int step = 1;
    for (int t = shp._data.size() - 1; t >= 0; t--) {
        i += _data[t] * step;
        step *= shp[t];
    }
    return i;
}
UINT _TensorDim_t::toAddr(const _TensorDim_t & shp, const UINT & mask) const
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (shp.isShapeValid() == false || shp.rangeCheckAt(*this) == false)
        T_error("cannot resolve absolute address for invalid "
            "shape/index descriptor.\n");
    if (shp.dim() > 32)
        T_error("dimension too high (>32)");
#endif
    /* resolve absolute address */
    UINT i = 0;
    UINT step = 1;
    UINT _m = mask;
    for (int t = shp._data.size() - 1; t >= 0; t--, _m>>=1) {
        if (_m & 1) {
            i += _data[t] * step;
            step *= shp[t];
        }
    }
    return i;
}

_TensorDim_t _TensorDim_t::toPos(const UINT & addr) const
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (this->isShapeValid() == false)
        T_error("invalid shape, cannot convert address to pos.\n");
    if (addr >= UINT(this->numel()))
        T_error("cannot resolve position for absolute addr %u, "
            "addr out of range (0~%d accepted).", addr, numel()-1);
#endif
    _TensorDim_t pos;
    for (int t = 0; t < _data.size(); t++)
        pos._append(0);
    UINT i = addr;
    for (int t = _data.size() - 1; t >= 0; t--) {
        pos[t] = i % _data[t];
        i /= _data[t];
    }
    return pos;
}
bool _TensorDim_t::rangeCheckAt(const _TensorDim_t & pos) const
{
    if (_data.size() == 0) {
#ifdef NN_ENABLE_RUNTIME_CHECKING
        T_error("invalid shape \"\".\n");
#endif
        return false;
    }
    if (this->dim() == 0) {
        /* scalar */
        if ((pos._data.size() != 1) || (pos._data.size()==1 && pos[0] != 0) ) {
#ifdef NN_ENABLE_RUNTIME_CHECKING
            char p[256];
            pos.toStr(p);
            T_error("cannot retrieve tensor element at position %s "
                "because the tensor is a scalar.\n", p);
#endif
            return false;
        }
        return true;
    }
    else {
        if (pos._data.size() != this->_data.size()) {
#ifdef NN_ENABLE_RUNTIME_CHECKING
            char p[256], s[256];
            pos.toStr(p);
            this->toStr(s);
            T_error("cannot retrieve tensor element at position \"%s\" "
                "while tensor shape is \"%s\".\n", p, s);
#endif
            return false;
        }
        for (int i = 0; i < _data.size(); i++) {
            if (pos[i] >= _data[i] || pos[i] < 0) {
#ifdef NN_ENABLE_RUNTIME_CHECKING
                char p[256], s[256];
                pos.toStr(p);
                this->toStr(s);
                T_error("cannot retrieve tensor element at position %s "
                    "while tensor shape is %s, index out of range.\n", p, s);
#endif
                return false;
            }
        }
    }
    return true;
}
bool _TensorDim_t::rangeCheckEnd(const _TensorDim_t & pos) const
{
    if (_data.size() == 0) {
#ifdef NN_ENABLE_RUNTIME_CHECKING
        T_error("invalid shape \"\".\n");
#endif
        return false;
    }
    if (this->dim() == 0) {
        /* scalar */
        if ((pos._data.size() != 1) || (pos._data.size() == 1 && pos[0] != 1)) {
#ifdef NN_ENABLE_RUNTIME_CHECKING
            char p[256];
            pos.toStr(p);
            T_error("cannot retrieve tensor element till position %s "
                "because the tensor is a scalar.\n", p);
#endif
            return false;
        }
        return true;
    }
    else {
        if (pos._data.size() != this->_data.size()) {
#ifdef NN_ENABLE_RUNTIME_CHECKING
            char p[256], s[256];
            pos.toStr(p);
            this->toStr(s);
            T_error("cannot retrieve tensor element at position \"%s\" "
                "while tensor shape is \"%s\".\n", p, s);
#endif
            return false;
        }
        for (int i = 0; i < _data.size(); i++) {
            if (pos[i] > _data[i] || pos[i] < 0) {
#ifdef NN_ENABLE_RUNTIME_CHECKING
                char p[256], s[256];
                pos.toStr(p);
                this->toStr(s);
                T_error("cannot retrieve tensor element till position %s "
                    "while tensor shape is %s, index out of range.\n", p, s);
#endif
                return false;
            }
        }
    }
    return true;
}
