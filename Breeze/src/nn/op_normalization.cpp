#include "nn/op_normalization.h"

/* inplace batch normalization (1d/2d/3d) */
/*
X: input data
running_mu: running mean
running_sigma: running standard deviation (std),
sample_mu: mean of the input sample
sample_sigma: std of the input sample
momentum: exponential moving average coefficient
gamma: weights applied to z-scored input
beta: biases applied to z-scored input
*/
void batchNorm(RealTensor & X, RealTensor & running_mu, RealTensor & running_sigma, 
    const REAL & momentum, RealTensor& sample_mu, RealTensor& sample_sigma,
    RealTensor& gamma, RealTensor& beta, bool train)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (X.dim() != 3 && X.dim() != 4 && X.dim() != 5)
        T_error("NN error: batchNorm expects input to be 3/4/5 dimensional but got %d.", X.dim());
#endif
    X = T_swapBatchChannel(X);
    int channels = X.shape()[0];
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (running_mu.dim() != 1 || (channels != running_mu.numel()))
        T_error("NN error: invalid tensor shape in batchNorm().");
    if (running_sigma.dim() != 1 || (channels != running_sigma.numel()))
        T_error("NN error: invalid tensor shape in batchNorm().");
#endif
    int stride = 1;
    const REAL eps = REAL(1e-6);
    for (int d = 1; d < X.dim(); d++)
        stride *= X.shape()[d];
#ifdef NN_ENABLE_OPENMP
#pragma omp parallel for schedule (static)
#endif
    for (int c = 0; c < channels; c++) {
        RealTensor A(T_buildShape({ stride }));
        A._dptrWrite(0, stride, X, c*stride);
        /*A.print();*/
        REAL _mu = A.mean();
        REAL _sigma = Sqrt(A.var(false));
        if (train) {
            A = (A - _mu) / (_sigma + eps); /* z-score transform */
            running_mu[c] = (1 - momentum) * running_mu[c] + momentum * _mu;
            running_sigma[c] = (1 - momentum) * running_sigma[c] + momentum * _sigma;
        }
        else {
            A = (A - running_mu[c]) / (running_sigma[c] + eps); /* z-score transform */
        }
        A *= gamma[c]; A += beta[c]; /* apply linear transform */
        X._dptrWrite(c*stride, stride, A, 0);
        sample_mu[c] = _mu;
        sample_sigma[c] = _sigma;
    }
    X = T_swapBatchChannel(X);

    /*X.print();*/
}

OpBatchNorm::OpBatchNorm(){nextOp = prevOp = NULL;}
void OpBatchNorm::createOp(const int inChannels, const REAL momentum)
{
    this->inChannels = inChannels;
    this->momentum = momentum;
    this->W = RealTensor(T_buildShape({ inChannels }));
    this->B = RealTensor(T_buildShape({ inChannels }));
    this->running_mu = RealTensor(T_buildShape({ inChannels }));
    this->running_sigma = RealTensor(T_buildShape({ inChannels }));
    this->W.fill_(1);
    this->B.fill_(0);
    this->running_mu.fill_(0);
    this->running_sigma.fill_(1);
    this->dLdW = RealTensor(T_buildShape({ inChannels }));
    this->dLdB = RealTensor(T_buildShape({ inChannels }));
    this->sample_mu = RealTensor(T_buildShape({ inChannels }));
    this->sample_sigma = RealTensor(T_buildShape({ inChannels }));
}
const char * OpBatchNorm::getOpClassName() const{return "OpBatchNorm2d";}
void OpBatchNorm::forward(RealTensor * _input)
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

    /* parameter check */
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (X.dim() != 3 && X.dim() != 4 && X.dim() != 5) {
        char s[64];
        X.shape().toStr(s);
        T_error("NN error: invalid input tensor in batchnorm. "
            "Tensor can be a 3D tensor with shape (B, C, N), or "
            "a 4D tensor with shape (B, C, H ,W), or "
            "a 5D tensor with shape (B, C, X, Y, Z). "
            "got a %dD tensor with shape (%s).", X.dim(), s);
    }
    if (X.shape()[1] != inChannels) {
        char s[64];
        X.shape().toStr(s);
        T_error("NN error: batchnorm expects an tensor with %d input channel(s), "
            "but %d was given. Tensor shape is (%s).", this->inChannels, X.shape()[1], s);
    }
#endif
    /* forward pass */
    this->inShape = X.shape();
    this->outShape = this->inShape;
    this->Y = X;
/*
    Y.print();
*/
    batchNorm(Y, this->running_mu, this->running_sigma, this->momentum,
        this->sample_mu, this->sample_sigma, this->W, this->B, this->_training);

    //Y.print();
}
void OpBatchNorm::backward(GraphOp * _op)
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
    /////////////////////////////////////////
    //dLdY.print("%.4f");

    dLdY = T_swapBatchChannel(dLdY); /* c,b,h,w */
    X = T_swapBatchChannel(X); 
    dLdX = RealTensor(dLdY.shape());
    int channels = dLdY.shape()[0];
    int stride = 1;
    for (int d = 1; d < dLdY.dim(); d++)
        stride *= dLdY.shape()[d];
    
#ifdef NN_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int c = 0; c < channels; c++) {
        const REAL eps = REAL(1e-6);
        RealTensor dLdY_channel(T_buildShape({ stride }));
        RealTensor X_channel(T_buildShape({ stride }));
        dLdY_channel._dptrWrite(0, stride, dLdY, c*stride);
        X_channel._dptrWrite(0, stride, X, c*stride);
        /* dLdX */
        RealTensor B = dLdY_channel * (W[c] / (sample_sigma[c] + eps));
        //B.print("%.4f");
        dLdX._dptrWrite(c*stride, stride, B, 0);
        /* dLdW */
        dLdW[c] = (dLdY_channel * (X_channel - sample_mu[c]) / (sample_sigma[c] + eps)).sum();
        /* dLdB */
        dLdB[c] = dLdY_channel.sum();
    }
    dLdX = T_swapBatchChannel(dLdX);
    X.purge();
    dLdY.purge();

    //this->W.print();
    //this->B.print();

    /////////////////////////////////////////
    //dLdX.print("%.4f");
}
void OpBatchNorm::linkTo(GraphOp * _op)
{
    this->nextOp = _op;
    if (_op != NULL)
        _op->_linkFrom(this);
}
void OpBatchNorm::_linkFrom(GraphOp * _op)
{
    this->prevOp = _op;
}
void OpBatchNorm::params(Array<RealTensor*>* _param, Array<RealTensor*>* _grad)
{
    _param->append(&W);
    _param->append(&B);
    _grad->append(&dLdW);
    _grad->append(&dLdB);
}
SerializedTensors<REAL> OpBatchNorm::state()
{
    SerializedTensors<REAL> ser;
    ser.serializeTensor(W);
    ser.serializeTensor(B);
    return ser;
}
bool OpBatchNorm::loadState(SerializedTensors<REAL>& ser)
{
    TensorShape shape_W = W.shape(), shape_B = B.shape();
    W = ser.deserializeTensor();
    B = ser.deserializeTensor();
    if (shape_W != W.shape()){
        T_error("NN error: tensor shape changed after loading state. Before (%s), after (%s).", shape_W, W.shape());
        return false;
    }
    if (shape_B != B.shape()) {
        T_error("NN error: tensor shape changed after loading state. Before (%s), after (%s).", shape_B, B.shape());
        return false;
    }
    return true;
}
RealTensor OpBatchNorm::input()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (X.isNull())
        T_error("NN error: cannot retrieve input tensor as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return X;
}
RealTensor OpBatchNorm::output()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (Y.isNull())
        T_error("NN error: cannot retrieve output tensor as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return Y;
}
RealTensor OpBatchNorm::inputGrad()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdX.isNull())
        T_error("NN error: cannot retrieve input gradient as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return dLdX;
}
RealTensor OpBatchNorm::outputGrad()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdY.isNull())
        T_error("NN error: cannot retrieve output gradient as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return dLdY;
}
void OpBatchNorm::_passGrad(GraphOp * _prevOp, RealTensor * _grad)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_prevOp != prevOp)
        T_error("NN error: specified layer is not linked to the input slot of this layer.\n");
#endif
    /* passing gradients from current layer to previous layer */
    this->dLdX.moveStorage(*_grad);
}
void OpBatchNorm::_passInput(GraphOp * _nextOp, RealTensor * _input)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_nextOp != nextOp)
        T_error("NN error: specified layer is not linked to the output slot of this layer.\n");
#endif
    /* set next layer input to be the output of this layer */
    this->Y.moveStorage(*_input);
}
