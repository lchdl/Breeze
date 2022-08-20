#include "nn/op_linear.h"

OpLinear::OpLinear() { nextOp = prevOp = NULL; }
const char * OpLinear::getOpClassName() const { return "OpLinear"; }
void OpLinear::forward(RealTensor * _input)
{
    /* * * * * * * * * * * * * * * * * */
    /* step 1: setting input properly  */
    /* * * * * * * * * * * * * * * * * */
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
    if (X.dim() < 2) {
        char s[64];
        X.shape().toStr(s);
        T_error("NN error, input dimension < 2 (%s).", s);
    }
#endif
    /* dynamically create input and output because the batch size */
    /* can be only determined when sending data into op */
    int _batchSize = X.shape()[0];
    this->inShape = T_buildShape<2>({ _batchSize, this->inFeatures });
    this->outShape = T_buildShape<2>({ _batchSize, this->outFeatures });
    Y = RealTensor::zero(outShape);
    /* * * * * * * * * * * * * * * * * * * * * * * * * */
    /* step 2: compute output based on requested input */
    /* * * * * * * * * * * * * * * * * * * * * * * * * */
    X.unsqueeze_(X.dim()); /* convert X to column vector */
    
    Y = T_mmul(W, X) + b.item();
    
    X.squeeze_(X.dim()); /* restore X and Y to original shape */
    Y.squeeze_(Y.dim());

}
void OpLinear::backward(GraphOp * _op)
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
    /* * * * * * * * * * * * * * */
    /* step 2: compute gradients */
    /* * * * * * * * * * * * * * */
    dLdX = RealTensor(inShape);
    int batchSize = X.shape()[0];
    int inNeurons = W.shape()[1];
    int outNeurons = W.shape()[0];
#ifdef NN_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int b = 0; b < batchSize; b++) {
        /* accumulate gradients in each sample */
        RealTensor dLdW_sample = RealTensor::zero(dLdW.shape());
        RealTensor dLdb_sample = RealTensor::zero(dLdb.shape());
        RealTensor X_sample(T_buildShape<1>({ inNeurons }));
        RealTensor dLdY_sample(T_buildShape<1>({ outNeurons }));
        X_sample._dptrWrite(0, X_sample.numel(), X, b*X_sample.numel());
        dLdY_sample._dptrWrite(0, dLdY_sample.numel(), dLdY, b*dLdY_sample.numel());
        /* compute dLdW, dLdb for this sample */
        for (int i = 0; i < outNeurons; i++) {
            for (int j = 0; j < inNeurons; j++) {
                RealTensor::type dydw = X_sample[j];    /* dy/dw */
                RealTensor::type dLdy = dLdY_sample[i]; /* dL/dy */
                dLdW_sample[i*inNeurons + j] += dLdy * dydw;
            }
            dLdb_sample[0] += dLdY_sample[i];
        }
        /* compute dLdX for this sample */
        RealTensor dLdX_sample = RealTensor::zero(T_buildShape<1>({ inNeurons }));
        for (int i = 0; i < inNeurons; i++) {
            for (int j = 0; j < outNeurons; j++)
                dLdX_sample[i] += dLdY_sample[j] * W[j*inNeurons + i];
        }
        /* update to total (may require omp critical) */
#ifdef NN_ENABLE_OPENMP
#pragma omp critical(oplinear_backward)
        {
#endif
            dLdW += dLdW_sample;
            dLdb += dLdb_sample;
#ifdef NN_ENABLE_OPENMP
        }
#endif
        dLdX._dptrWrite(b*dLdX_sample.numel(), dLdX_sample.numel(), dLdX_sample, 0);
    }
    dLdY.purge();
}
void OpLinear::linkTo(GraphOp * _op)
{
    this->nextOp = _op;
    if (_op != NULL)
        _op->_linkFrom(this);
}
void OpLinear::_linkFrom(GraphOp * _op)
{
    this->prevOp = _op;
}
void OpLinear::_passGrad(GraphOp * _prevOp, RealTensor * _grad)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_prevOp != prevOp)
        T_error("NN error: specified layer is not linked to the input slot of this layer.\n");
#endif
    /* passing gradients from current layer to previous layer */
    this->dLdX.moveStorage(*_grad);
}

void OpLinear::_passInput(GraphOp * _nextOp, RealTensor * _input)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_nextOp != nextOp)
        T_error("NN error: specified layer is not linked to the output slot of this layer.\n");
#endif
    /* set next layer input to be the output of this layer */
    this->Y.moveStorage(*_input);
}

void OpLinear::params(Array<RealTensor*>* _param, Array<RealTensor*>* _grad)
{
    _param->append(&(this->W));
    _param->append(&(this->b));
    _grad->append(&(this->dLdW));
    _grad->append(&(this->dLdb));
}
SerializedTensors<REAL> OpLinear::state()
{
    SerializedTensors<REAL> ser;
    ser.serializeTensor(W);
    ser.serializeTensor(b);
    return ser;
}
bool OpLinear::loadState(SerializedTensors<REAL>& ser)
{
    TensorShape shape_W = W.shape(), shape_b = b.shape();
    W = ser.deserializeTensor();
    b = ser.deserializeTensor();
    if (shape_W != W.shape()) {
        T_error("NN error: tensor shape changed after loading state. Before (%s), after (%s).", shape_W, W.shape());
        return false;
    }
    if (shape_b != b.shape()) {
        T_error("NN error: tensor shape changed after loading state. Before (%s), after (%s).", shape_b, b.shape());
        return false;
    }
    return true;
}
RealTensor OpLinear::input()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (X.isNull())
        T_error("NN error: cannot retrieve input tensor as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return X;
}
RealTensor OpLinear::output()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (Y.isNull())
        T_error("NN error: cannot retrieve output tensor as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return Y;
}
RealTensor OpLinear::inputGrad()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdX.isNull())
        T_error("NN error: cannot retrieve input gradient as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return dLdX;
}
RealTensor OpLinear::outputGrad()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdY.isNull())
        T_error("NN error: cannot retrieve output gradient as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return dLdY;
}
void OpLinear::createOp(int _inFeatures, int _outFeatures)
{
    this->inFeatures = _inFeatures;
    this->outFeatures = _outFeatures;
    W = RealTensor::zero(T_buildShape<2>({ _outFeatures, _inFeatures }));
    b = RealTensor::zero(TensorShape("0"));
    dLdW = RealTensor::zero(T_buildShape<2>({ _outFeatures, _inFeatures }));
    dLdb = RealTensor::zero(TensorShape("0"));
    nextOp = prevOp = NULL;

    /* initialize weights and biases */
    RealTensor::type u = RealTensor::type(1.0 / sqrt(W.shape()[0]));
    for (int i = 0; i < W.numel(); i++)
        W[i] = RealTensor::type(uniform_distribution(-u, u));
    b[0] = RealTensor::type(uniform_distribution(-u, u));
}
