#include "nn/op_convolution.h"

OpConv2d::OpConv2d() { nextOp = prevOp = NULL; }
void OpConv2d::createOp(const int _inChannels, const int _outChannels, 
    const char * _kernel, const char* _stride, const char* _padding){

    this->inChannels = _inChannels;
    this->outChannels = _outChannels;
    this->kernelSize = T_buildArray<int>(_kernel);
    this->strideSize = T_buildArray<int>(_stride);
    this->padSize = T_buildArray<int>(_padding);
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_inChannels < 1)
        T_error("NN error: invalid conv2d input channel setting: %d.", _inChannels);
    if (_outChannels < 1)
        T_error("NN error: invalid conv2d output channel setting: %d.", _outChannels);
    if (this->kernelSize.size() != 2 || this->kernelSize[0] < 1 || this->kernelSize[1] < 1)
        T_error("NN error: invalid conv2d kernel size setting (%s).", _kernel);
    if (this->strideSize.size() != 2 || this->strideSize[0] < 1 || this->strideSize[1] < 1)
        T_error("NN error: invalid conv2d stride size setting (%s).", _stride);
    if (this->padSize.size() != 2 || this->padSize[0] < 0 || this->padSize[1] < 0)
        T_error("NN error: invalid conv2d padding setting (%s).", _padding);
#endif
    /* create and initialize kernel weights and biases */
    this->K = RealTensor(T_buildShape<4>({ _outChannels, _inChannels, this->kernelSize[0], this->kernelSize[1] }));
    this->b = RealTensor(T_buildShape<1>({ _outChannels }));
    /* initialize */
    REAL n = REAL(_inChannels * this->kernelSize[0] * this->kernelSize[1]);
    REAL std = REAL(1.0) / Sqrt(n);
    for (int i = 0; i < this->K.numel(); i++)
        this->K[i] = uniform_distribution(-std, +std);
    for (int i = 0; i < this->b.numel(); i++)
        this->b[i] = uniform_distribution(-std, +std);
    dLdK = RealTensor(K.shape());
    dLdb = RealTensor(b.shape());
}
const char * OpConv2d::getOpClassName() const { return "OpConv2d"; }
void OpConv2d::forward(RealTensor * _input)
{
    if (_input != NULL) {
        /* input comes from external source */
        X = *_input;
    }
    else { /* input comes from internal data flow */
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (prevOp == NULL) {
            T_error("NN error: cannot feed forward because no data has given.\n");
            return;
        }
#endif
        /* request input data from previous layer */
        prevOp->_passInput(this, &(this->X));
    }
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (X.dim() != 4) {
        char s[64];
        X.shape().toStr(s);
        T_error("NN error: conv2d expects a 4D tensor input with shape (B, C, H, W), "
            "got tensor with shape (%s).", s);
    }
#endif
    /* dynamically create input and output because the batch size */
    /* can be only determined when sending data into op */
    int _batchSize = X.shape()[0];
    int _inChannels = X.shape()[1];
    int _imageHeight = X.shape()[2], _imageWidth = X.shape()[3];
    this->inShape = T_buildShape<4>({ _batchSize, _inChannels, _imageHeight, _imageWidth });
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (this->inChannels != _inChannels) {
        char s[64];
        X.shape().toStr(s);
        T_error("NN error: conv2d expects input to have %d channels, "
            "but %d was given. Tensor shape is (%s).", this->inChannels, _inChannels, s);
    }
#endif

    Xpad = X.pad(T_buildArray<4, int>({ padSize[0],padSize[0],padSize[1],padSize[1] }));
    X.purge();
    this->padShape = Xpad.shape();
/*
    printf("value for T_conv2d:\n");
    Xpad.print();
    K.print();
    printf("stride size: %d %d", this->strideSize[0], this->strideSize[1]);
*/
    Y = T_conv2d(Xpad, K, this->strideSize);

    //Y.print();

    this->outShape = Y.shape();
    /* adding biases */
    int _outChannels = Y.shape()[1];
    int _outHeight = Y.shape()[2], _outWidth = Y.shape()[3]; /* output height and width without padding */
    for (int bi = 0; bi < _batchSize; bi++) {
        int _step1 = _outChannels * _outHeight * _outWidth;
        int _step2 = _outHeight * _outWidth;
        for (int c = 0; c < _outChannels; c++) {
            for (int i = 0; i < _outHeight*_outWidth; i++) {
                Y[bi * _step1 + c * _step2 + i] += this->b[c];
            }
        }
    }
}
void OpConv2d::backward(GraphOp * _op)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_op == NULL && nextOp == NULL) {
        T_error("NN error: backward op set to NULL, which means the gradient flows from "
            "next op, but next op is NULL.");
        return;
    }
#endif
    /* * * * * * * * * * */
    /* step 2: get dL/dY */
    /* * * * * * * * * * */
    if (_op == this) {
        dLdY = RealTensor(outShape);
        /* gradients will be back propagated from this  */
        /* layer so setting output gradients to all one */
        dLdY.fill_(RealTensor::type(1.0));
    }
    else {
        /* passing gradients of the next layer to current layer */
        nextOp->_passGrad(this, &(this->dLdY));
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (outShape.isEqual(dLdY.shape()) == false) {
            char es[64], as[64];
            outShape.toStr(es);
            dLdY.shape().toStr(as);
            T_error("NN error: out shape changed after backward(). "
                "Expected shape: (%s), actual shape: (%s).", es, as);
        }
#endif
    }

    /* compute gradients */
    _T_conv2d_backward(dLdY, Xpad, K, this->strideSize, dLdK, dLdb, dLdX);

#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdK.shape() != this->K.shape())
        T_error("NN error: kernel shape error when backprop gradients.");
    if (dLdX.shape() != this->padShape) {
        char a[64], b[64];
        dLdX.shape().toStr(a);
        padShape.toStr(b);
        T_error("NN error: input shape error when backprop gradients. "
            "dLdX (%s), padShape (%s).", a, b);
    }
    if (dLdb.shape() != this->b.shape())
        T_error("NN error: bias shape error when backprop gradients.");
#endif
    /* shrink */
    dLdX = dLdX.shrink(T_buildArray<4, int>({ padSize[0],padSize[0],padSize[1],padSize[1] }));
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdX.shape() != this->inShape) {
        char a[64], b[64];
        dLdX.shape().toStr(a);
        inShape.toStr(b);
        T_error("NN error: input shape error when backprop gradients. "
            "dLdX (%s), inShape (%s).", a, b);
    }
#endif
    dLdY.purge();
    Xpad.purge();
    X.purge(); // is this correct? please double check

}
void OpConv2d::linkTo(GraphOp * _op)
{
    this->nextOp = _op;
    if (_op != NULL)
        _op->_linkFrom(this);
}
void OpConv2d::params(Array<RealTensor*>* _param, Array<RealTensor*>* _grad)
{
    _param->append(&K);
    _param->append(&b);
    _grad->append(&dLdK);
    _grad->append(&dLdb);
}
SerializedTensors<REAL> OpConv2d::state()
{
    SerializedTensors<REAL> ser;
    ser.serializeTensor(this->K);
    ser.serializeTensor(this->b);
    return ser;
}
bool OpConv2d::loadState(SerializedTensors<REAL>& ser)
{
    TensorShape shape_K = K.shape(), shape_b = b.shape();
    K = ser.deserializeTensor();
    b = ser.deserializeTensor();
    if (shape_K != K.shape()) {
        T_error("NN error: tensor shape changed after loading state. Before (%s), after (%s).", shape_K, K.shape());
        return false;
    }
    if (shape_b != b.shape()) {
        T_error("NN error: tensor shape changed after loading state. Before (%s), after (%s).", shape_b, b.shape());
        return false;
    }
    return true;
}
RealTensor OpConv2d::input()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (X.isNull())
        T_error("NN error: cannot retrieve input tensor as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return X;
}
RealTensor OpConv2d::output()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (Y.isNull())
        T_error("NN error: cannot retrieve output tensor as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return Y;
}
RealTensor OpConv2d::inputGrad()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdX.isNull())
        T_error("NN error: cannot retrieve input gradient as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return dLdX;
}
RealTensor OpConv2d::outputGrad()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdY.isNull())
        T_error("NN error: cannot retrieve output gradient as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return dLdY;
}
void OpConv2d::_linkFrom(GraphOp * _op)
{
    this->prevOp = _op;
}
void OpConv2d::_passGrad(GraphOp * _prevOp, RealTensor * _grad)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_prevOp != prevOp)
        T_error("NN error: specified layer is not linked to the input slot of this layer.\n");
#endif
    /* passing gradients from current layer to previous layer */
    this->dLdX.moveStorage(*_grad);
}
void OpConv2d::_passInput(GraphOp * _nextOp, RealTensor * _input)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_nextOp != nextOp)
        T_error("NN error: specified layer is not linked to the output slot of this layer.\n");
#endif
    /* set next layer input to be the output of this layer */
    this->Y.moveStorage(*_input);
}

OpConvTranspose2d::OpConvTranspose2d() { nextOp = prevOp = NULL; }
void OpConvTranspose2d::createOp(const int _inChannels, 
    const int _outChannels, const char * _kernel, const char * _stride)
{
    this->inChannels = _inChannels;
    this->outChannels = _outChannels;
    this->kernelSize = T_buildArray<int>(_kernel);
    this->strideSize = T_buildArray<int>(_stride);
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_inChannels < 1)
        T_error("NN error: invalid convTranspose2d input channel setting: %d.", _inChannels);
    if (_outChannels < 1)
        T_error("NN error: invalid convTranspose2d output channel setting: %d.", _outChannels);
    if (this->kernelSize.size() != 2 || this->kernelSize[0] < 1 || this->kernelSize[1] < 1)
        T_error("NN error: invalid convTranspose2d kernel size setting (%s).", _kernel);
    if (this->strideSize.size() != 2 || this->strideSize[0] < 1 || this->strideSize[1] < 1)
        T_error("NN error: invalid convTranspose2d stride size setting (%s).", _stride);
#endif
    /* create and initialize kernel weights and biases */
    this->K = RealTensor(T_buildShape<4>({ _outChannels, _inChannels, this->kernelSize[0], this->kernelSize[1] }));
    this->b = RealTensor(T_buildShape<1>({ _outChannels }));
    /* initialize */
    REAL n = REAL(_inChannels * this->kernelSize[0] * this->kernelSize[1]);
    REAL std = REAL(1.0) / Sqrt(n);
    for (int i = 0; i < this->K.numel(); i++)
        this->K[i] = uniform_distribution(-std, +std);
    for (int i = 0; i < this->b.numel(); i++)
        this->b[i] = uniform_distribution(-std, +std);
    dLdK = RealTensor(K.shape());
    dLdb = RealTensor(b.shape());
}
const char * OpConvTranspose2d::getOpClassName() const {return "OpConvTranspose2d";}
void OpConvTranspose2d::forward(RealTensor * _input)
{
    if (_input != NULL) {
        /* input comes from external source */
        X = *_input;
    }
    else { /* input comes from internal data flow */
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (prevOp == NULL) {
            T_error("NN error: cannot feed forward because no data has given.\n");
            return;
        }
#endif
        /* request input data from previous layer */
        prevOp->_passInput(this, &(this->X));
    }
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (X.dim() != 4) {
        char s[64];
        X.shape().toStr(s);
        T_error("NN error: OpConvTranspose2d expects a 4D tensor input with shape (B, C, H, W), "
            "got tensor with shape (%s).", s);
    }
#endif
    int _batchSize = X.shape()[0];
    int _inChannels = X.shape()[1];
    int _imageHeight = X.shape()[2], _imageWidth = X.shape()[3];
    this->inShape = T_buildShape<4>({ _batchSize, _inChannels, _imageHeight, _imageWidth });
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (this->inChannels != _inChannels) {
        char s[64];
        X.shape().toStr(s);
        T_error("NN error: OpConvTranspose2d expects input to have %d channels, "
            "but %d was given. Tensor shape is (%s).", this->inChannels, _inChannels, s);
    }
#endif
    Y = T_convTranspose2d(X, K, this->strideSize);
    this->outShape = Y.shape();
    /* adding biases */
    int _outChannels = Y.shape()[1];
    int _outHeight = Y.shape()[2], _outWidth = Y.shape()[3]; /* output height and width without padding */
    for (int bi = 0; bi < _batchSize; bi++) {
        int _step1 = _outChannels * _outHeight * _outWidth;
        int _step2 = _outHeight * _outWidth;
        for (int c = 0; c < _outChannels; c++) {
            for (int i = 0; i < _outHeight*_outWidth; i++) {
                Y[bi * _step1 + c * _step2 + i] += this->b[c];
            }
        }
    }
}
void OpConvTranspose2d::backward(GraphOp * _op)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_op == NULL && nextOp == NULL) {
        T_error("NN error: backward op set to NULL, which means the gradient flows from "
            "next op, but next op is NULL.");
        return;
    }
#endif
    /* * * * * * * * * * */
    /* step 2: get dL/dY */
    /* * * * * * * * * * */
    if (_op == this) {
        dLdY = RealTensor(outShape);
        /* gradients will be back propagated from this  */
        /* layer so setting output gradients to all one */
        dLdY.fill_(RealTensor::type(1.0));
    }
    else {
        /* passing gradients of the next layer to current layer */
        nextOp->_passGrad(this, &(this->dLdY));
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (outShape.isEqual(dLdY.shape()) == false) {
            char es[64], as[64];
            outShape.toStr(es);
            dLdY.shape().toStr(as);
            T_error("NN error: out shape changed after backward(). "
                "Expected shape: (%s), actual shape: (%s).", es, as);
        }
#endif
    }
    ///////////////////////////
    //dLdY.print("%.4f");

    _T_convTranspose2d_backward(this->inShape, dLdY, X, K, this->strideSize, dLdK, dLdb, dLdX);
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdX.shape() != this->inShape) {
        char a[64], b[64];
        dLdX.shape().toStr(a);
        inShape.toStr(b);
        T_error("NN error: input shape error when backprop gradients. "
            "dLdX (%s), inShape (%s).", a, b);
    }
#endif
    dLdY.purge();
    X.purge();

    ///////////////////////////
    //dLdX.print("%.4f");
}
void OpConvTranspose2d::linkTo(GraphOp * _op)
{
    this->nextOp = _op;
    if (_op != NULL)
        _op->_linkFrom(this);
}
void OpConvTranspose2d::params(Array<RealTensor*>* _param, Array<RealTensor*>* _grad)
{
    _param->append(&K);
    _param->append(&b);
    _grad->append(&dLdK);
    _grad->append(&dLdb);
}
SerializedTensors<REAL> OpConvTranspose2d::state()
{
    SerializedTensors<REAL> ser;
    ser.serializeTensor(this->K);
    ser.serializeTensor(this->b);
    return ser;
}
bool OpConvTranspose2d::loadState(SerializedTensors<REAL>& ser)
{
    TensorShape shape_K = K.shape(), shape_b = b.shape();
    K = ser.deserializeTensor();
    b = ser.deserializeTensor();
    if (shape_K != K.shape()) {
        T_error("NN error: tensor shape changed after loading state. Before (%s), after (%s).", shape_K, K.shape());
        return false;
    }
    if (shape_b != b.shape()) {
        T_error("NN error: tensor shape changed after loading state. Before (%s), after (%s).", shape_b, b.shape());
        return false;
    }
    return true;
}
RealTensor OpConvTranspose2d::input()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (X.isNull())
        T_error("NN error: cannot retrieve input tensor as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return X;
}
RealTensor OpConvTranspose2d::output()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (Y.isNull())
        T_error("NN error: cannot retrieve output tensor as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return Y;
}
RealTensor OpConvTranspose2d::inputGrad()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdX.isNull())
        T_error("NN error: cannot retrieve input gradient as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return dLdX;
}
RealTensor OpConvTranspose2d::outputGrad()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdY.isNull())
        T_error("NN error: cannot retrieve output gradient as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return dLdY;
}
void OpConvTranspose2d::_linkFrom(GraphOp * _op)
{
    this->prevOp = _op;
}
void OpConvTranspose2d::_passGrad(GraphOp * _prevOp, RealTensor * _grad)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_prevOp != prevOp)
        T_error("NN error: specified layer is not linked to the input slot of this layer.\n");
#endif
    /* passing gradients from current layer to previous layer */
    this->dLdX.moveStorage(*_grad);
}
void OpConvTranspose2d::_passInput(GraphOp * _nextOp, RealTensor * _input)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_nextOp != nextOp)
        T_error("NN error: specified layer is not linked to the output slot of this layer.\n");
#endif
    /* set next layer input to be the output of this layer */
    this->Y.moveStorage(*_input);
}

OpConv3d::OpConv3d() { nextOp = prevOp = NULL; }
void OpConv3d::createOp(const int _inChannels, const int _outChannels, const char * _kernel, const char * _stride, const char * _padding)
{
    this->inChannels = _inChannels;
    this->outChannels = _outChannels;
    this->kernelSize = T_buildArray<int>(_kernel);
    this->strideSize = T_buildArray<int>(_stride);
    this->padSize = T_buildArray<int>(_padding);
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_inChannels < 1)
        T_error("NN error: invalid conv3d input channel setting: %d.", _inChannels);
    if (_outChannels < 1)
        T_error("NN error: invalid conv3d output channel setting: %d.", _outChannels);
    if (this->kernelSize.size() != 3 || 
        this->kernelSize[0] < 1 || this->kernelSize[1] < 1 || this->kernelSize[2] < 1)
        T_error("NN error: invalid conv3d kernel size setting (%s).", _kernel);
    if (this->strideSize.size() != 3 || 
        this->strideSize[0] < 1 || this->strideSize[1] < 1 || this->strideSize[2] < 1)
        T_error("NN error: invalid conv3d stride size setting (%s).", _stride);
    if (this->padSize.size() != 3 || 
        this->padSize[0] < 0 || this->padSize[1] < 0 || this->padSize[2] < 0)
        T_error("NN error: invalid conv3d padding setting (%s).", _padding);
#endif
    /* create and initialize kernel weights and biases */
    this->K = RealTensor(T_buildShape<5>({ _outChannels, _inChannels, this->kernelSize[0], this->kernelSize[1], this->kernelSize[2] }));
    this->b = RealTensor(T_buildShape<1>({ _outChannels }));
    /* initialize */
    REAL n = REAL(_inChannels * this->kernelSize[0] * this->kernelSize[1] * this->kernelSize[2]);
    REAL std = REAL(1.0) / Sqrt(n);
    for (int i = 0; i < this->K.numel(); i++)
        this->K[i] = uniform_distribution(-std, +std);
    for (int i = 0; i < this->b.numel(); i++)
        this->b[i] = uniform_distribution(-std, +std);
    dLdK = RealTensor(K.shape());
    dLdb = RealTensor(b.shape());
}
const char * OpConv3d::getOpClassName() const { return "OpConv3d"; }
void OpConv3d::forward(RealTensor * _input)
{
    if (_input != NULL) {
        /* input comes from external source */
        X = *_input;
    }
    else { /* input comes from internal data flow */
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (prevOp == NULL) {
            T_error("NN error: cannot feed forward because no data has given.\n");
            return;
        }
#endif
        /* request input data from previous layer */
        prevOp->_passInput(this, &(this->X));
    }
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (X.dim() != 5) {
        char s[64];
        X.shape().toStr(s);
        T_error("NN error: conv3d expects a 5D tensor input with shape (B, C, X, Y, Z), "
            "got tensor with shape (%s).", s);
    }
#endif
    /* dynamically create input and output because the batch size */
    /* can be only determined when sending data into op */
    int _batchSize = X.shape()[0];
    int _inChannels = X.shape()[1];
    int _inputX = X.shape()[2], _inputY = X.shape()[3], _inputZ = X.shape()[4];
    this->inShape = T_buildShape<5>({ _batchSize, _inChannels, _inputX, _inputY, _inputZ });
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (this->inChannels != _inChannels) {
        char s[64];
        X.shape().toStr(s);
        T_error("NN error: conv3d expects input to have %d channels, "
            "but %d was given. Tensor shape is (%s).", this->inChannels, _inChannels, s);
    }
#endif

    Xpad = X.pad(T_buildArray<6, int>({ padSize[0],padSize[0],padSize[1],padSize[1], padSize[2],padSize[2] }));
    X.purge();
    this->padShape = Xpad.shape();
    Y = T_conv3d(Xpad, K, this->strideSize);
    this->outShape = Y.shape();
    /* adding biases */
    int _outChannels = Y.shape()[1];
    int _outputX = Y.shape()[2], _outputY = Y.shape()[3], _outputZ = Y.shape()[4]; 
    int _step1 = _outChannels * _outputX * _outputY * _outputZ;
    int _step2 = _outputX * _outputY * _outputZ;
    for (int bi = 0; bi < _batchSize; bi++) {
        for (int c = 0; c < _outChannels; c++) {
            for (int i = 0; i < _step2; i++) {
                Y[bi * _step1 + c * _step2 + i] += this->b[c];
            }
        }
    }
}
void OpConv3d::backward(GraphOp * _op)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_op == NULL && nextOp == NULL) {
        T_error("NN error: backward op set to NULL, which means the gradient flows from "
            "next op, but next op is NULL.");
        return;
    }
#endif
    /* * * * * * * * * * */
    /* step 2: get dL/dY */
    /* * * * * * * * * * */
    if (_op == this) {
        dLdY = RealTensor(outShape);
        /* gradients will be back propagated from this  */
        /* layer so setting output gradients to all one */
        dLdY.fill_(RealTensor::type(1.0));
    }
    else {
        /* passing gradients of the next layer to current layer */
        nextOp->_passGrad(this, &(this->dLdY));
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (outShape.isEqual(dLdY.shape()) == false) {
            char es[64], as[64];
            outShape.toStr(es);
            dLdY.shape().toStr(as);
            T_error("NN error: out shape changed after backward(). "
                "Expected shape: (%s), actual shape: (%s).", es, as);
        }
#endif
    }

    /* compute gradients */
    _T_conv3d_backward(dLdY, Xpad, K, this->strideSize, dLdK, dLdb, dLdX);

#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdK.shape() != this->K.shape())
        T_error("NN error: kernel shape error when backprop gradients.");
    if (dLdX.shape() != this->padShape) {
        char a[64], b[64];
        dLdX.shape().toStr(a);
        padShape.toStr(b);
        T_error("NN error: input shape error when backprop gradients. "
            "dLdX (%s), padShape (%s).", a, b);
    }
    if (dLdb.shape() != this->b.shape())
        T_error("NN error: bias shape error when backprop gradients.");
#endif
    /* shrink */
    dLdX = dLdX.shrink(T_buildArray<6, int>({ padSize[0],padSize[0],padSize[1],padSize[1], padSize[2], padSize[2] }));
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdX.shape() != this->inShape) {
        char a[64], b[64];
        dLdX.shape().toStr(a);
        inShape.toStr(b);
        T_error("NN error: input shape error when backprop gradients. "
            "dLdX (%s), inShape (%s).", a, b);
    }
#endif
    dLdY.purge();
    Xpad.purge();
    X.purge(); // is this correct? please double check
}
void OpConv3d::linkTo(GraphOp * _op)
{
    this->nextOp = _op;
    if (_op != NULL)
        _op->_linkFrom(this);
}
void OpConv3d::params(Array<RealTensor*>* _param, Array<RealTensor*>* _grad)
{
    _param->append(&K);
    _param->append(&b);
    _grad->append(&dLdK);
    _grad->append(&dLdb);
}
SerializedTensors<REAL> OpConv3d::state()
{
    SerializedTensors<REAL> ser;
    ser.serializeTensor(this->K);
    ser.serializeTensor(this->b);
    return ser;
}
bool OpConv3d::loadState(SerializedTensors<REAL>& ser)
{
    TensorShape shape_K = K.shape(), shape_b = b.shape();
    K = ser.deserializeTensor();
    b = ser.deserializeTensor();
    if (shape_K != K.shape()) {
        T_error("NN error: tensor shape changed after loading state. Before (%s), after (%s).", shape_K, K.shape());
        return false;
    }
    if (shape_b != b.shape()) {
        T_error("NN error: tensor shape changed after loading state. Before (%s), after (%s).", shape_b, b.shape());
        return false;
    }
    return true;
}
RealTensor OpConv3d::input()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (X.isNull())
        T_error("NN error: cannot retrieve input tensor as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return X;
}
RealTensor OpConv3d::output()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (Y.isNull())
        T_error("NN error: cannot retrieve output tensor as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return Y;
}
RealTensor OpConv3d::inputGrad()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdX.isNull())
        T_error("NN error: cannot retrieve input gradient as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return dLdX;
}
RealTensor OpConv3d::outputGrad()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdY.isNull())
        T_error("NN error: cannot retrieve output gradient as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return dLdY;
}
void OpConv3d::_linkFrom(GraphOp * _op)
{
    this->prevOp = _op;
}
void OpConv3d::_passGrad(GraphOp * _prevOp, RealTensor * _grad)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_prevOp != prevOp)
        T_error("NN error: specified layer is not linked to the input slot of this layer.\n");
#endif
    /* passing gradients from current layer to previous layer */
    this->dLdX.moveStorage(*_grad);
}
void OpConv3d::_passInput(GraphOp * _nextOp, RealTensor * _input)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_nextOp != nextOp)
        T_error("NN error: specified layer is not linked to the output slot of this layer.\n");
#endif
    /* set next layer input to be the output of this layer */
    this->Y.moveStorage(*_input);
}

OpConvTranspose3d::OpConvTranspose3d() { nextOp = prevOp = NULL; }
void OpConvTranspose3d::createOp(const int _inChannels, const int _outChannels, const char * _kernel, const char * _stride)
{
    this->inChannels = _inChannels;
    this->outChannels = _outChannels;
    this->kernelSize = T_buildArray<int>(_kernel);
    this->strideSize = T_buildArray<int>(_stride);
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_inChannels < 1)
        T_error("NN error: invalid convTranspose3d input channel setting: %d.", _inChannels);
    if (_outChannels < 1)
        T_error("NN error: invalid convTranspose3d output channel setting: %d.", _outChannels);
    if (this->kernelSize.size() != 3 || 
        this->kernelSize[0] < 1 || this->kernelSize[1] < 1 || this->kernelSize[2] < 1)
        T_error("NN error: invalid convTranspose3d kernel size setting (%s).", _kernel);
    if (this->strideSize.size() != 3 || 
        this->strideSize[0] < 1 || this->strideSize[1] < 1 || this->strideSize[2] < 1)
        T_error("NN error: invalid convTranspose3d stride size setting (%s).", _stride);
#endif
    /* create and initialize kernel weights and biases */
    this->K = RealTensor(T_buildShape<5>({ _outChannels, _inChannels, this->kernelSize[0], this->kernelSize[1], this->kernelSize[2] }));
    this->b = RealTensor(T_buildShape<1>({ _outChannels }));
    /* initialize */
    REAL n = REAL(_inChannels * this->kernelSize[0] * this->kernelSize[1] * this->kernelSize[2]);
    REAL std = REAL(1.0) / Sqrt(n);
    for (int i = 0; i < this->K.numel(); i++)
        this->K[i] = uniform_distribution(-std, +std);
    for (int i = 0; i < this->b.numel(); i++)
        this->b[i] = uniform_distribution(-std, +std);
    dLdK = RealTensor(K.shape());
    dLdb = RealTensor(b.shape());
}
const char * OpConvTranspose3d::getOpClassName() const { return "OpConvTranspose3d"; }
void OpConvTranspose3d::forward(RealTensor * _input)
{
    if (_input != NULL) {
        /* input comes from external source */
        X = *_input;
    }
    else { /* input comes from internal data flow */
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (prevOp == NULL) {
            T_error("NN error: cannot feed forward because no data has given.\n");
            return;
        }
#endif
        /* request input data from previous layer */
        prevOp->_passInput(this, &(this->X));
    }
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (X.dim() != 5) {
        char s[64];
        X.shape().toStr(s);
        T_error("NN error: OpConvTranspose3d expects a 5D tensor input with shape (B, C, X, Y, Z), "
            "got tensor with shape (%s).", s);
    }
#endif
    int _batchSize = X.shape()[0];
    int _inChannels = X.shape()[1];
    int _inputX = X.shape()[2], _inputY = X.shape()[3], _inputZ = X.shape()[4];
    this->inShape = T_buildShape<5>({ _batchSize, _inChannels, _inputX, _inputY, _inputZ });
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (this->inChannels != _inChannels) {
        char s[64];
        X.shape().toStr(s);
        T_error("NN error: OpConvTranspose3d expects input to have %d channels, "
            "but %d was given. Tensor shape is (%s).", this->inChannels, _inChannels, s);
    }
#endif
    Y = T_convTranspose3d(X, K, this->strideSize);
    this->outShape = Y.shape();
    /* adding biases */
    int _outChannels = Y.shape()[1];
    /* output size without padding */
    int _outX = Y.shape()[2], _outY = Y.shape()[3], _outZ = Y.shape()[4];
    int _step1 = _outChannels * _outX * _outY * _outZ;
    int _step2 = _outX * _outY * _outZ;
    for (int bi = 0; bi < _batchSize; bi++) {
        for (int c = 0; c < _outChannels; c++) {
            for (int i = 0; i < _outX*_outY*_outZ; i++) {
                Y[bi * _step1 + c * _step2 + i] += this->b[c];
            }
        }
    }
}
void OpConvTranspose3d::backward(GraphOp * _op)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_op == NULL && nextOp == NULL) {
        T_error("NN error: backward op set to NULL, which means the gradient flows from "
            "next op, but next op is NULL.");
        return;
    }
#endif
    /* * * * * * * * * * */
    /* step 2: get dL/dY */
    /* * * * * * * * * * */
    if (_op == this) {
        dLdY = RealTensor(outShape);
        /* gradients will be back propagated from this  */
        /* layer so setting output gradients to all one */
        dLdY.fill_(RealTensor::type(1.0));
    }
    else {
        /* passing gradients of the next layer to current layer */
        nextOp->_passGrad(this, &(this->dLdY));
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (outShape.isEqual(dLdY.shape()) == false) {
            char es[64], as[64];
            outShape.toStr(es);
            dLdY.shape().toStr(as);
            T_error("NN error: out shape changed after backward(). "
                "Expected shape: (%s), actual shape: (%s).", es, as);
        }
#endif
    }
    _T_convTranspose3d_backward(inShape, dLdY, X, K, this->strideSize, dLdK, dLdb, dLdX);
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdX.shape() != this->inShape) {
        char a[64], b[64];
        dLdX.shape().toStr(a);
        inShape.toStr(b);
        T_error("NN error: input shape error when backprop gradients. "
            "dLdX (%s), inShape (%s).", a, b);
    }
#endif
    dLdY.purge();
    X.purge();
}
void OpConvTranspose3d::linkTo(GraphOp * _op)
{
    this->nextOp = _op;
    if (_op != NULL)
        _op->_linkFrom(this);
}
void OpConvTranspose3d::params(Array<RealTensor*>* _param, Array<RealTensor*>* _grad)
{
    _param->append(&K);
    _param->append(&b);
    _grad->append(&dLdK);
    _grad->append(&dLdb);
}
SerializedTensors<REAL> OpConvTranspose3d::state()
{
    SerializedTensors<REAL> ser;
    ser.serializeTensor(this->K);
    ser.serializeTensor(this->b);
    return ser;
}
bool OpConvTranspose3d::loadState(SerializedTensors<REAL>& ser)
{
    TensorShape shape_K = K.shape(), shape_b = b.shape();
    K = ser.deserializeTensor();
    b = ser.deserializeTensor();
    if (shape_K != K.shape()) {
        T_error("NN error: tensor shape changed after loading state. Before (%s), after (%s).", shape_K, K.shape());
        return false;
    }
    if (shape_b != b.shape()) {
        T_error("NN error: tensor shape changed after loading state. Before (%s), after (%s).", shape_b, b.shape());
        return false;
    }
    return true;
}
RealTensor OpConvTranspose3d::input()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (X.isNull())
        T_error("NN error: cannot retrieve input tensor as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return X;
}
RealTensor OpConvTranspose3d::output()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (Y.isNull())
        T_error("NN error: cannot retrieve output tensor as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return Y;
}
RealTensor OpConvTranspose3d::inputGrad()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdX.isNull())
        T_error("NN error: cannot retrieve input gradient as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return dLdX;
}
RealTensor OpConvTranspose3d::outputGrad()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdY.isNull())
        T_error("NN error: cannot retrieve output gradient as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return dLdY;
}
void OpConvTranspose3d::_linkFrom(GraphOp * _op)
{
    this->prevOp = _op;
}
void OpConvTranspose3d::_passGrad(GraphOp * _prevOp, RealTensor * _grad)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_prevOp != prevOp)
        T_error("NN error: specified layer is not linked to the input slot of this layer.\n");
#endif
    /* passing gradients from current layer to previous layer */
    this->dLdX.moveStorage(*_grad);
}
void OpConvTranspose3d::_passInput(GraphOp * _nextOp, RealTensor * _input)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_nextOp != nextOp)
        T_error("NN error: specified layer is not linked to the output slot of this layer.\n");
#endif
    /* set next layer input to be the output of this layer */
    this->Y.moveStorage(*_input);
}

