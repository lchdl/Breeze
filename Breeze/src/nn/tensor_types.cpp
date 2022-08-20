#include "nn/tensor_types.h"

FloatTensor::FloatTensor() { _dptr = NULL; setPrintOptions(6, 2); }
FloatTensor::FloatTensor(const char * shp) : _TensorBase_t<float>::_TensorBase_t(shp) { setPrintOptions(6, 2); }
FloatTensor::FloatTensor(const TensorShape & tshp) : _TensorBase_t<float>::_TensorBase_t(tshp) { setPrintOptions(6, 2); }
FloatTensor::FloatTensor(const FloatTensor & that) : _TensorBase_t<float>::_TensorBase_t(that) { setPrintOptions(6, 2); }
FloatTensor::FloatTensor(const _TensorBase_t<float>& that) : _TensorBase_t<float>::_TensorBase_t(that) { setPrintOptions(6, 2); }
FloatTensor::FloatTensor(const float & value) : _TensorBase_t<float>::_TensorBase_t(value) { setPrintOptions(6, 2); }
FloatTensor::~FloatTensor() { purge(); }
void FloatTensor::print() const
{
    printf("\n");
    char shpstr[64];
    shape().toStr(shpstr);
    char buf[64];
    if (isScalar())
        strcpy(buf, "scalar");
    else if (dim() == 1)
        strcpy(buf, "vector");
    else if (dim() == 2)
        strcpy(buf, "matrix");
    else
        sprintf(buf, "%dD array", dim());
    printf("Tensor: %s, dim=%d, shape=(%s)\n", buf, dim(), shpstr);
    if (isScalar()) { /* scalar */
        sprintf(buf, printFormatter, _dptr[0]);
        printf("data={%s}\n", buf);
    }
    else if (dim() == 1) { /* vector */
        TensorCoord coord(1);
        printf("data={ ");
        do {
            float v = this->at(coord);
            sprintf(buf, printFormatter, v);
            printf("%s ", buf);
        } while (coord.next(shape()));
        printf("}\n");
    }
    else if (dim() == 2) { /* matrix */
        TensorCoord coord(2);
        printf("data={\n");
        int p = 0;
        TensorShape shp = shape();
        do {
            float v = this->at(coord);
            sprintf(buf, printFormatter, v);
            printf("%s ", buf);
            p++;
            if (p == shp[1]) {
                printf("\n");
                p %= shp[1];
            }
        } while (coord.next(shp));
        printf("}\n");
    }
    else {
        TensorCoord coordSlice(dim() - 2);
        TensorShape shapeSlice(dim() - 2), realShape = shape();
        for (int i = 0; i < dim() - 2; i++)
            shapeSlice[i] = realShape[i];

        char slicing[64];
        int printNum = 0;
        const int maxPrintNum = 32;
        do {
            /* print matrix */
            coordSlice.toStr(slicing);
            strcat(slicing, ",:,:");
            printf("slice[%s]={\n", slicing);
            FloatTensor sliceTensor = (*this)[slicing];
            TensorCoord coordInternal(2);
            TensorShape shapeInternal = sliceTensor.shape();
            int p = 0;
            do {
                float v = sliceTensor.at(coordInternal);
                sprintf(buf, printFormatter, v);
                printf("%s ", buf);
                p++;
                if (p == shapeInternal[1]) {
                    printf("\n");
                    p %= shapeInternal[1];
                }
            } while (coordInternal.next(shapeInternal));
            printf("}\n");
            printNum++;
            if (printNum == maxPrintNum) {
                if (shapeSlice.numel() > maxPrintNum) {
                    printf("<%d slice(s) remains>\n...\n",
                        shapeSlice.numel() - maxPrintNum);
                }
                break;
            }
        } while (coordSlice.next(shapeSlice));
    }
    printf("\n");

}
void FloatTensor::setPrintOptions(int all, int right)
{
    if (all > 60) all = 60;
    if (right > 30) right = 30;
    sprintf(this->printFormatter, "%%%d.%df", all, right);
}

DoubleTensor::DoubleTensor() { _dptr = NULL; setPrintOptions(6, 2); }
DoubleTensor::DoubleTensor(const char * shp) : _TensorBase_t<double>::_TensorBase_t(shp) { setPrintOptions(6, 2); }
DoubleTensor::DoubleTensor(const TensorShape & tshp) : _TensorBase_t<double>::_TensorBase_t(tshp) { setPrintOptions(6, 2); }
DoubleTensor::DoubleTensor(const DoubleTensor & that) : _TensorBase_t<double>::_TensorBase_t(that) { setPrintOptions(6, 2); }
DoubleTensor::DoubleTensor(const _TensorBase_t<double>& that) : _TensorBase_t<double>::_TensorBase_t(that) { setPrintOptions(6, 2); }
DoubleTensor::DoubleTensor(const double & value) : _TensorBase_t<double>::_TensorBase_t(value) { setPrintOptions(6, 2); }
DoubleTensor::~DoubleTensor() { purge(); }
void DoubleTensor::print() const
{
    printf("\n");
    char shpstr[64];
    shape().toStr(shpstr);
    char buf[64];
    if (isScalar())
        strcpy(buf, "scalar");
    else if (dim() == 1)
        strcpy(buf, "vector");
    else if (dim() == 2)
        strcpy(buf, "matrix");
    else
        sprintf(buf, "%dD array", dim());
    printf("Tensor: %s, dim=%d, shape=(%s)\n", buf, dim(), shpstr);
    if (isScalar()) { /* scalar */
        sprintf(buf, printFormatter, _dptr[0]);
        printf("data={%s}\n", buf);
    }
    else if (dim() == 1) { /* vector */
        TensorCoord coord(1);
        printf("data={ ");
        do {
            double v = this->at(coord);
            sprintf(buf, printFormatter, v);
            printf("%s ", buf);
        } while (coord.next(shape()));
        printf("}\n");
    }
    else if (dim() == 2) { /* matrix */
        TensorCoord coord(2);
        printf("data={\n");
        int p = 0;
        TensorShape shp = shape();
        do {
            double v = this->at(coord);
            sprintf(buf, printFormatter, v);
            printf("%s ", buf);
            p++;
            if (p == shp[1]) {
                printf("\n");
                p %= shp[1];
            }
        } while (coord.next(shp));
        printf("}\n");
    }
    else {
        TensorCoord coordSlice(dim() - 2);
        TensorShape shapeSlice(dim() - 2), realShape = shape();
        for (int i = 0; i < dim() - 2; i++)
            shapeSlice[i] = realShape[i];

        char slicing[64];
        int printNum = 0;
        const int maxPrintNum = 32;
        do {
            /* print matrix */
            coordSlice.toStr(slicing);
            strcat(slicing, ",:,:");
            printf("slice[%s]={\n", slicing);
            DoubleTensor sliceTensor = (*this)[slicing];
            TensorCoord coordInternal(2);
            TensorShape shapeInternal = sliceTensor.shape();
            int p = 0;
            do {
                double v = sliceTensor.at(coordInternal);
                sprintf(buf, printFormatter, v);
                printf("%s ", buf);
                p++;
                if (p == shapeInternal[1]) {
                    printf("\n");
                    p %= shapeInternal[1];
                }
            } while (coordInternal.next(shapeInternal));
            printf("}\n");
            printNum++;
            if (printNum == maxPrintNum) {
                if (shapeSlice.numel() > maxPrintNum) {
                    printf("<%d slice(s) remains>\n...\n",
                        shapeSlice.numel() - maxPrintNum);
                }
                break;
            }
        } while (coordSlice.next(shapeSlice));
    }
    printf("\n");
}
void DoubleTensor::setPrintOptions(int all, int right)
{
    if (all > 60) all = 60;
    if (right > 30) right = 30;
    sprintf(this->printFormatter, "%%%d.%dlf", all, right);
}

IntTensor::IntTensor() { _dptr = NULL; setPrintOptions(6); }
IntTensor::IntTensor(const char * shp) : _TensorBase_t<int>::_TensorBase_t(shp) { setPrintOptions(6); }
IntTensor::IntTensor(const TensorShape & tshp) : _TensorBase_t<int>::_TensorBase_t(tshp) { setPrintOptions(6); }
IntTensor::IntTensor(const IntTensor & that) : _TensorBase_t<int>::_TensorBase_t(that) { setPrintOptions(6); }
IntTensor::IntTensor(const _TensorBase_t<int>& that) : _TensorBase_t<int>::_TensorBase_t(that) { setPrintOptions(6); }
IntTensor::IntTensor(const int & value) : _TensorBase_t<int>::_TensorBase_t(value) { setPrintOptions(6); }
IntTensor::~IntTensor() { purge(); }
void IntTensor::print() const
{
    printf("\n");
    char shpstr[64];
    shape().toStr(shpstr);
    char buf[64];
    if (isScalar())
        strcpy(buf, "scalar");
    else if (dim() == 1)
        strcpy(buf, "vector");
    else if (dim() == 2)
        strcpy(buf, "matrix");
    else
        sprintf(buf, "%dD array", dim());
    printf("Tensor: %s, dim=%d, shape=(%s)\n", buf, dim(), shpstr);
    if (isScalar()) { /* scalar */
        sprintf(buf, printFormatter, _dptr[0]);
        printf("data={%s}\n", buf);
    }
    else if (dim() == 1) { /* vector */
        TensorCoord coord(1);
        printf("data={ ");
        do {
            int v = this->at(coord);
            sprintf(buf, printFormatter, v);
            printf("%s ", buf);
        } while (coord.next(shape()));
        printf("}\n");
    }
    else if (dim() == 2) { /* matrix */
        TensorCoord coord(2);
        printf("data={\n");
        int p = 0;
        TensorShape shp = shape();
        do {
            int v = this->at(coord);
            sprintf(buf, printFormatter, v);
            printf("%s ", buf);
            p++;
            if (p == shp[1]) {
                printf("\n");
                p %= shp[1];
            }
        } while (coord.next(shp));
        printf("}\n");
    }
    else {
        TensorCoord coordSlice(dim() - 2);
        TensorShape shapeSlice(dim() - 2), realShape = shape();
        for (int i = 0; i < dim() - 2; i++)
            shapeSlice[i] = realShape[i];

        char slicing[64];
        int printNum = 0;
        const int maxPrintNum = 32;
        do {
            /* print matrix */
            coordSlice.toStr(slicing);
            strcat(slicing, ",:,:");
            printf("slice[%s]={\n", slicing);
            IntTensor sliceTensor = (*this)[slicing];
            TensorCoord coordInternal(2);
            TensorShape shapeInternal = sliceTensor.shape();
            int p = 0;
            do {
                int v = sliceTensor.at(coordInternal);
                sprintf(buf, printFormatter, v);
                printf("%s ", buf);
                p++;
                if (p == shapeInternal[1]) {
                    printf("\n");
                    p %= shapeInternal[1];
                }
            } while (coordInternal.next(shapeInternal));
            printf("}\n");
            printNum++;
            if (printNum == maxPrintNum) {
                if (shapeSlice.numel() > maxPrintNum) {
                    printf("<%d slice(s) remains>\n...\n",
                        shapeSlice.numel() - maxPrintNum);
                }
                break;
            }
        } while (coordSlice.next(shapeSlice));
    }
    printf("\n");
}
void IntTensor::setPrintOptions(int all)
{
    if (all > 60) all = 60;
    sprintf(this->printFormatter, "%%%dd", all);
}

FloatTensor FloatTensor::mm(const FloatTensor & that) const
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    T_mm_check(*this, that);
#endif
    TensorShape lshape = shape(), rshape = that.shape();
    TensorShape newShape(2);
    newShape[0] = lshape[0];
    newShape[1] = rshape[1];
    FloatTensor resultTensor(newShape);
    FloatTensor thatT = that.transpose();
    int row0 = this->_shape[0], row1 = thatT._shape[0];
    int col = this->_shape[1];
    for (int i = 0; i < row0; i++) {
        for (int j = 0; j < row1; j++) {
#ifdef NN_ENABLE_AVX_INTRINSICS
            resultTensor._dptr[i*row1 + j] = _avx_float32_dot(&(this->_dptr[i*col]),
                &(thatT._dptr[j*col]), col);
#else
            float sum = 0.0f;
            for (int k = 0; k < col; k++) {
                sum += this->_dptr[i*col + k] * thatT._dptr[j*col + k];
            }
            resultTensor._dptr[i*row1 + j] = sum;
#endif
        }
    }
    return resultTensor;
}
FloatTensor FloatTensor::softmax(const int & axis) const
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dim() < 2)
        T_error("NN error: cannot taking softmax with tensor dimension "
            "less than 2.");
    if (axis <1 || axis>dim())
        T_error("NN error: invalid axis setting (%d).", axis);
#endif
    /* for speed, i will transpose axis to the last axis */
    Array<int> axesTpose;
    for (int i = 1; i <= dim(); i++) axesTpose.append(i);
    int tempAxis = axesTpose[dim() - 1];
    axesTpose[dim() - 1] = axesTpose[axis-1];
    axesTpose[axis-1] = tempAxis;
    FloatTensor Xt = transpose(axesTpose);
    /* now calculate softmax value from logits */
    int channels = Xt.shape()[Xt.dim() - 1];
    int numVecs = Xt.numel() / channels;
    for (int i = 0; i < numVecs; i++) {
        FloatTensor x(T_buildShape<1>({ channels }));
        FloatTensor s(T_buildShape<1>({ channels }));
        x._dptrWrite(0, channels, Xt, i*channels);
        s.fill_(0.0f);
        x -= x.max(); /* avoid too large value */
        for (int i = 0; i < channels; i++) s[i] = expf(x[i]);
        float den = 0.0f;
        for (int i = 0; i < channels; i++) den += s[i];
        for (int i = 0; i < channels; i++) s[i] /= den;
        Xt._dptrWrite(i*channels, channels, s, 0);
    }
    /* now transpose back and return */
    return Xt.transpose(axesTpose);
}
FloatTensor FloatTensor::mm_t(const FloatTensor & that) const
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    FloatTensor thatT = that.transpose();
    T_mm_check(*this, thatT);
#endif
    TensorShape newShape = T_buildShape<2>({ this->shape()[0], that.shape()[0] });
    FloatTensor resultTensor(newShape);
    int row0 = this->_shape[0], row1 = that._shape[0];
    int col = this->_shape[1];
    for (int i = 0; i < row0; i++) {
        for (int j = 0; j < row1; j++) {
#ifdef NN_ENABLE_AVX_INTRINSICS
            resultTensor._dptr[i*row1 + j] = _avx_float32_dot(&(this->_dptr[i*col]),
                &(that._dptr[j*col]), col);
#else
            float sum = 0.0f;
            for (int k = 0; k < col; k++) {
                sum += this->_dptr[i*col + k] * that._dptr[j*col + k];
            }
            resultTensor._dptr[i*row1 + j] = sum;
#endif
        }
    }
    return resultTensor;
}
FloatTensor FloatTensor::conv2d(const FloatTensor & kernel) const
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dim() != 2 || kernel.dim() != 2) {
        char a[64], b[64];
        shape().toStr(a);
        kernel.shape().toStr(b);
        T_error("Invalid input/kernel dimension. Both input and kernel should be "
            "2D in naive convolution. Input shape is (%s), kernel shape is (%s).",
            a, b);
    }
#endif
    int imageHeight = shape()[0], imageWidth = shape()[1];
    int kernelHeight = kernel.shape()[0], kernelWidth = kernel.shape()[1];
    int resultHeight = imageHeight + 1 - kernelHeight;
    int resultWidth = imageWidth + 1 - kernelWidth;
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (resultWidth < 1 || resultHeight < 1) {
        char a[64], b[64];
        shape().toStr(a);
        kernel.shape().toStr(b);
        T_error("Cannot perform 2D convolution due to invalid result image "
            "size (<1). Maybe the kernel size is too large. Input size "
            "is (%s), while kernel size is (%s).", a, b);
    }
#endif
    FloatTensor result(T_buildShape<1>({ resultHeight*resultWidth }));
    /* start convolution */
    int I = 0;
    for (int ih = 0; ih < resultHeight; ih++) {
        for (int iw = 0; iw < resultWidth; iw++) {
            FloatTensor::type sum = FloatTensor::type(0.0);
            for (int iy = 0; iy < kernelHeight; iy++) {
#ifdef NN_ENABLE_AVX_INTRINSICS
                int y = iy + ih;
                sum += _avx_float32_dot(&((*this)[y * imageWidth + iw]), &kernel[iy * kernelWidth], kernelWidth);
#else
                for (int ix = 0; ix < kernelWidth; ix++) {
                    int x = ix + iw, y = iy + ih;
                    sum += (*this)[y*imageWidth + x] * kernel[iy*kernelWidth + ix];
                }
#endif            
            }
            result[I++] = sum;
        }
    }
    return result.reshape(T_buildShape<2>({ resultHeight, resultWidth }));
}

DoubleTensor DoubleTensor::mm(const DoubleTensor & that) const
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    T_mm_check(*this, that);
#endif
    TensorShape lshape = shape(), rshape = that.shape();
    TensorShape newShape(2);
    newShape[0] = lshape[0];
    newShape[1] = rshape[1];
    DoubleTensor resultTensor(newShape);
    DoubleTensor thatT = that.transpose();
    int row0 = this->_shape[0], row1 = thatT._shape[0];
    int col = this->_shape[1];
    for (int i = 0; i < row0; i++) {
        for (int j = 0; j < row1; j++) {
#ifdef NN_ENABLE_AVX_INTRINSICS
            resultTensor._dptr[i*row1 + j] = _avx_float64_dot(&(this->_dptr[i*col]),
                &(thatT._dptr[j*col]), col);
#else
            double sum = 0.0f;
            for (int k = 0; k < col; k++) {
                sum += this->_dptr[i*col + k] * thatT._dptr[j*col + k];
            }
            resultTensor._dptr[i*row1 + j] = sum;
#endif
        }
    }
    return resultTensor;
}
DoubleTensor DoubleTensor::softmax(const int & axis) const
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dim() <= 2)
        T_error("NN error: cannot taking softmax with tensor dimension "
            "less than 2.");
    if (axis <1 || axis>dim())
        T_error("NN error: invalid axis setting (%d).", axis);
#endif
    /* for speed, i will transpose axis to the last axis */
    Array<int> axesTpose;
    for (int i = 1; i <= dim(); i++) axesTpose.append(i);
    int tempAxis = axesTpose[dim() - 1];
    axesTpose[dim() - 1] = axesTpose[axis - 1];
    axesTpose[axis - 1] = tempAxis;
    DoubleTensor Xt = transpose(axesTpose);
    /* now calculate softmax value from logits */
    int channels = Xt.shape()[Xt.dim() - 1];
    int numVecs = Xt.numel() / channels;
    for (int i = 0; i < numVecs; i++) {
        DoubleTensor x(T_buildShape<1>({ numVecs }));
        DoubleTensor s(T_buildShape<1>({ numVecs }));
        x._dptrWrite(0, channels, Xt, i*channels);
        s.fill_(0.0);
        x -= x.max(); /* avoid too large value */
        for (int i = 0; i < channels; i++) s[i] = exp(x[i]);
        double den = 0.0;
        for (int i = 0; i < channels; i++) den += s[i];
        for (int i = 0; i < channels; i++) s[i] /= den;
        Xt._dptrWrite(i*channels, channels, s, 0);
    }
    /* now transpose back and return */
    return Xt.transpose(axesTpose);
}
DoubleTensor DoubleTensor::mm_t(const DoubleTensor & that) const
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    DoubleTensor thatT = that.transpose();
    T_mm_check(*this, thatT);
#endif
    TensorShape newShape = T_buildShape<2>({ this->shape()[0], that.shape()[0] });
    DoubleTensor resultTensor(newShape);
    int row0 = this->_shape[0], row1 = that._shape[0];
    int col = this->_shape[1];
    for (int i = 0; i < row0; i++) {
        for (int j = 0; j < row1; j++) {
#ifdef NN_ENABLE_AVX_INTRINSICS
            resultTensor._dptr[i*row1 + j] = _avx_float64_dot(&(this->_dptr[i*col]),
                &(that._dptr[j*col]), col);
#else
            double sum = 0.0;
            for (int k = 0; k < col; k++) {
                sum += this->_dptr[i*col + k] * that._dptr[j*col + k];
            }
            resultTensor._dptr[i*row1 + j] = sum;
#endif
        }
    }
    return resultTensor;
}
DoubleTensor DoubleTensor::conv2d(const DoubleTensor & kernel) const
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dim() != 2 || kernel.dim() != 2) {
        char a[64], b[64];
        shape().toStr(a);
        kernel.shape().toStr(b);
        T_error("Invalid input/kernel dimension. Both input and kernel should be "
            "2D in naive convolution. Input shape is (%s), kernel shape is (%s).",
            a, b);
    }
#endif
    int imageHeight = shape()[0], imageWidth = shape()[1];
    int kernelHeight = kernel.shape()[0], kernelWidth = kernel.shape()[1];
    int resultHeight = imageHeight + 1 - kernelHeight;
    int resultWidth = imageWidth + 1 - kernelWidth;
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (resultWidth < 1 || resultHeight < 1) {
        char a[64], b[64];
        shape().toStr(a);
        kernel.shape().toStr(b);
        T_error("Cannot perform 2D convolution due to invalid result image "
            "size (<1). Maybe the kernel size is too large. Input size "
            "is (%s), while kernel size is (%s).", a, b);
    }
#endif
    DoubleTensor result(T_buildShape<1>({ resultHeight*resultWidth }));
    /* start convolution */
    int I = 0;
    for (int ih = 0; ih < resultHeight; ih++) {
        for (int iw = 0; iw < resultWidth; iw++) {
            DoubleTensor::type sum = DoubleTensor::type(0.0);
            for (int iy = 0; iy < kernelHeight; iy++) {
#ifdef NN_ENABLE_AVX_INTRINSICS
                int y = iy + ih;
                sum += _avx_float64_dot(&((*this)[y * imageWidth + iw]), &kernel[iy * kernelWidth], kernelWidth);
#else
                for (int ix = 0; ix < kernelWidth; ix++) {
                    int x = ix + iw, y = iy + ih;
                    sum += (*this)[y*imageWidth + x] * kernel[iy*kernelWidth + ix];
                }
#endif
            }
            result[I++] = sum;
        }
    }
    return result.reshape(T_buildShape<2>({ resultHeight, resultWidth }));
}

IntTensor IntTensor::mm(const IntTensor & that) const
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    T_mm_check(*this, that);
#endif
    TensorShape lshape = shape(), rshape = that.shape();
    TensorShape newShape(2);
    newShape[0] = lshape[0];
    newShape[1] = rshape[1];
    IntTensor resultTensor(newShape);
    IntTensor thatT = that.transpose();
    int row0 = this->_shape[0], row1 = thatT._shape[0];
    int col = this->_shape[1];
    for (int i = 0; i < row0; i++) {
        for (int j = 0; j < row1; j++) {
#ifdef NN_ENABLE_AVX_INTRINSICS
            resultTensor._dptr[i*row1 + j] = _avx_int32_dot(&(this->_dptr[i*col]),
                &(thatT._dptr[j*col]), col);
#else
            int sum = 0;
            for (int k = 0; k < col; k++) {
                sum += this->_dptr[i*col + k] * thatT._dptr[j*col + k];
            }
            resultTensor._dptr[i*row1 + j] = sum;
#endif
        }
    }
    return resultTensor;
}
IntTensor IntTensor::softmax(const int & axis) const
{
    T_error("NN error: IntTensor does not support softmax()");
    return IntTensor();
}
IntTensor IntTensor::mm_t(const IntTensor & that) const
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    IntTensor thatT = that.transpose();
    T_mm_check(*this, thatT);
#endif
    TensorShape newShape = T_buildShape<2>({ this->shape()[0], that.shape()[0] });
    IntTensor resultTensor(newShape);
    int row0 = this->_shape[0], row1 = that._shape[0];
    int col = this->_shape[1];
    for (int i = 0; i < row0; i++) {
        for (int j = 0; j < row1; j++) {
#ifdef NN_ENABLE_AVX_INTRINSICS
            resultTensor._dptr[i*row1 + j] = _avx_int32_dot(&(this->_dptr[i*col]),
                &(that._dptr[j*col]), col);
#else
            int sum = 0;
            for (int k = 0; k < col; k++) {
                sum += this->_dptr[i*col + k] * that._dptr[j*col + k];
            }
            resultTensor._dptr[i*row1 + j] = sum;
#endif
        }
    }
    return resultTensor;
}
IntTensor IntTensor::conv2d(const IntTensor & kernel) const
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dim() != 2 || kernel.dim() != 2) {
        char a[64], b[64];
        shape().toStr(a);
        kernel.shape().toStr(b);
        T_error("Invalid input/kernel dimension. Both input and kernel should be "
            "2D in naive convolution. Input shape is (%s), kernel shape is (%s).",
            a, b);
    }
#endif
    int imageHeight = shape()[0], imageWidth = shape()[1];
    int kernelHeight = kernel.shape()[0], kernelWidth = kernel.shape()[1];
    int resultHeight = imageHeight + 1 - kernelHeight;
    int resultWidth = imageWidth + 1 - kernelWidth;
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (resultWidth < 1 || resultHeight < 1) {
        char a[64], b[64];
        shape().toStr(a);
        kernel.shape().toStr(b);
        T_error("Cannot perform 2D convolution due to invalid result image "
            "size (<1). Maybe the kernel size is too large. Input size "
            "is (%s), while kernel size is (%s).", a, b);
    }
#endif
    IntTensor result(T_buildShape<1>({ resultHeight*resultWidth }));
    /* start convolution */
    int I = 0;
    for (int ih = 0; ih < resultHeight; ih++) {
        for (int iw = 0; iw < resultWidth; iw++) {
            IntTensor::type sum = IntTensor::type(0.0);
            for (int iy = 0; iy < kernelHeight; iy++) {
#ifdef NN_ENABLE_AVX_INTRINSICS
                int y = iy + ih;
                sum += _avx_int32_dot(&((*this)[y * imageWidth + iw]), &kernel[iy * kernelWidth], kernelWidth);
#else
                for (int ix = 0; ix < kernelWidth; ix++) {
                    int x = ix + iw, y = iy + ih;
                    sum += (*this)[y*imageWidth + x] * kernel[iy*kernelWidth + ix];
                }
#endif
            }
            result[I++] = sum;
        }
    }
    return result.reshape(T_buildShape<2>({ resultHeight, resultWidth }));
}

