#include "nn/op_activation.h"

OpLeakyReLU::OpLeakyReLU() { nSlope = REAL(0.1); nextOp = prevOp = NULL; }
const char * OpLeakyReLU::getOpClassName() const
{
    return "OpLeakyReLU";
}
void OpLeakyReLU::forward(RealTensor * _input)
{
    /* * * * * * * * * * * * * * * * * */
    /* step 1: setting input properly  */
    /* * * * * * * * * * * * * * * * * */
    if (_input != NULL) /* input comes from external source */
        X = *_input;
    else { /* input comes from internal data flow */
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (prevOp == NULL) {
            printf("NN error: cannot feed forward because no data has given.\n");
            return;
        }
#endif
        /* request input data from previous layer */
        prevOp->_passInput(this, &(this->X));
    }
    /* get input shape and dynamic initialize remaining tensors */
    Y = RealTensor(X.shape());
    inShape = X.shape();
    outShape = Y.shape();

    /* * * * * * * * * * * * * * * * * * * * * * * * * */
    /* step 2: compute output based on requested input */
    /* * * * * * * * * * * * * * * * * * * * * * * * * */
    for (int i = 0; i < X.numel(); i++) {
        REAL inval = X[i];
        Y[i] = (inval > REAL(0) ? inval : (this->nSlope * inval));
    }

}
void OpLeakyReLU::backward(GraphOp * _op)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_op == NULL && nextOp == NULL) {
        T_error("NN error: backward op == NULL, which means the gradient "
            "flows from next op, but next op is NULL.");
        return;
    }
#endif
    /* * * * * * * * * * */
    /* step 1: set dL/dY */
    /* * * * * * * * * * */
    if (_op == this) {                        
        dLdY = RealTensor(outShape);       /* gradients will be back propagated from this  */
        dLdY.fill_(RealTensor::type(1.0));  /* layer so setting output gradients to all one */
    }
    else {
        /* passing gradients of the next layer to current layer */
        nextOp->_passGrad(this, &(this->dLdY));
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (outShape != dLdY.shape()) {
            char os[64], ns[64];
            outShape.toStr(os);
            dLdY.shape().toStr(ns);
            T_error("NN error: shape changed after backward(). "
                "Old shape: (%s), new shape (%s).", os, ns);
        }
#endif
    }
    ///////////////////////////////
    //dLdY.print("%.4f");

    /* * * * * * * * * * * * * * * * * * * * * * * * * * * * */
    /* step 2: compute partial derivatives and form BP chain */
    /* * * * * * * * * * * * * * * * * * * * * * * * * * * * */
    dLdX = RealTensor(inShape);
    dLdX.fill_(REAL(0.0));
    for (int i = 0; i < inShape.numel(); i++) {
        REAL x = X[i];
        REAL dydx = (x > REAL(0) ? REAL(1.0) : this->nSlope); /* partial derivative */
        dLdX[i] += dLdY[i] * dydx; /* make chain */
    }
    dLdY.purge();


    ///////////////////////////////
    //dLdX.print("%.4f");

}
void OpLeakyReLU::linkTo(GraphOp * _op)
{
    this->nextOp = _op;
    if (_op != NULL)
        _op->_linkFrom(this);
}
void OpLeakyReLU::_linkFrom(GraphOp * _op)
{
    this->prevOp = _op;
}
void OpLeakyReLU::_passGrad(GraphOp * _prevOp, RealTensor * _grad)
{   
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_prevOp != prevOp)
        T_error("NN error: specified layer is not linked to the input slot of this layer.\n");
#endif
    /* passing gradients from current layer to previous layer */
    /* here i used moveStorage to save memory space */
    this->dLdX.moveStorage(*_grad);
}
void OpLeakyReLU::_passInput(GraphOp * _nextOp, RealTensor * _input)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_nextOp != nextOp)
        T_error("NN error: specified layer is not linked to the output slot of this layer.\n");
#endif
    /* passing output of this layer to next layer */
    /* here i used moveStorage to save memory space */
    this->Y.moveStorage(*_input);
}
void OpLeakyReLU::params(Array<RealTensor*>* _param, Array<RealTensor*>* _grad) {}
SerializedTensors<REAL> OpLeakyReLU::state() {return SerializedTensors<REAL>();}
bool OpLeakyReLU::loadState(SerializedTensors<REAL>& ser) { return true; }
RealTensor OpLeakyReLU::input()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (X.isNull())
        T_error("NN error: cannot retrieve input tensor as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return X;
}
RealTensor OpLeakyReLU::output()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (Y.isNull())
        T_error("NN error: cannot retrieve output tensor as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return Y;
}
RealTensor OpLeakyReLU::inputGrad()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdX.isNull())
        T_error("NN error: cannot retrieve input gradient as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return dLdX;
}
RealTensor OpLeakyReLU::outputGrad()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdY.isNull())
        T_error("NN error: cannot retrieve output gradient as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return dLdY;
}
void OpLeakyReLU::createOp(const REAL& nSlope)
{
    this->nSlope = nSlope;
    nextOp = prevOp = NULL;
}

OpSoftmax::OpSoftmax()
{
    nextOp = prevOp = NULL;
}
void OpSoftmax::createOp()
{
}
const char * OpSoftmax::getOpClassName() const
{
    return "OpSoftmax";
}
void OpSoftmax::forward(RealTensor * _input)
{
    /* * * * * * * * * * * * * * * * * */
    /* step 1: setting input properly  */
    /* * * * * * * * * * * * * * * * * */
    if (_input != NULL) /* input comes from external source */
        X = *_input;
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
    /* * * * * * * * * * * * * * * * * * * * */
    /* step 2: compute output based on input */
    /* * * * * * * * * * * * * * * * * * * * */
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (X.dim() < 2)
        T_error("NN error: input must have at least 2 dimensions.");
#endif
    Y = X.softmax(2); /* taking softmax in the channel axis, remember axis count from 1 */

    inShape = X.shape();
    outShape = Y.shape();

    //Y.print();

    /* backward is not relevant to input, so we can delete it */
    X.purge();

}
void OpSoftmax::backward(GraphOp * _op)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_op == NULL && nextOp == NULL) {
        T_error("NN error: backward op set to NULL, which means "
            "the gradient flows from next op, but next op is NULL.");
        return;
    }
#endif
    typedef RealTensor::type ElemType;
    /* * * * * * * * */
    /* step 2: dL/dY */
    /* * * * * * * * */
    if (_op == this) {              
        dLdY = RealTensor(outShape);/* gradients will be back propagated from this layer */
        dLdY.fill_(ElemType(1.0));  /* so setting output gradients to all one */
    }
    else {
        /* passing gradients of the next layer to current layer */
        nextOp->_passGrad(this, &(this->dLdY));
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (outShape != dLdY.shape()) {
            char os[64], ns[64];
            outShape.toStr(os);
            dLdY.shape().toStr(ns);
            T_error("NN error: shape changed after backward(). "
                "Old shape is: (%s), new shape is: (%s).", os, ns);
        }
#endif
    }
    dLdX = RealTensor(inShape);
    dLdX.fill_(ElemType(0.0));
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdX.shape() != dLdY.shape())
        T_error("NN error: input gradient shape should be equal to output gradient shape.");
#endif
    /* * * * * * * * * * * * */
    /* step 3: compute dL/dX */
    /* * * * * * * * * * * * */
    /* 1. build axes mapping */
    Array<int> axesTpose;
    for (int i = 1; i <= Y.dim(); i++) axesTpose.append(i);
    int tempAxis = axesTpose[Y.dim() - 1];
    axesTpose[Y.dim()-1] = axesTpose[1];
    axesTpose[1] = tempAxis;
    /* transpose */
    dLdY = dLdY.transpose(axesTpose);
    dLdX = dLdX.transpose(axesTpose);

    int channels = Y.shape()[1];
    int numVecs = Y.numel() / channels;

    Y = Y.transpose(axesTpose);

#ifdef NN_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int v = 0; v < numVecs; v++) {
        const RealTensor::type* ptrOut = &(Y.dptr()[v*channels]);
        const RealTensor::type* srcGrad = &(dLdY.dptr()[v*channels]);
        for(int j = 0; j < channels; j++) {
            REAL sum = REAL(0.0);
            for (int i = 0; i < channels; i++) {
                REAL delta = (i == j ? REAL(1.0) : REAL(0.0));
                REAL si = ptrOut[i], sj = ptrOut[j];
                REAL grad = srcGrad[i];
                sum += grad * si * (delta - sj);
            }
            dLdX[v*channels + j] = sum;
        }
    }
    dLdX = dLdX.transpose(axesTpose);
    Y = Y.transpose(axesTpose);
    dLdY.purge();

    //dLdX.print();
}
void OpSoftmax::linkTo(GraphOp * _op)
{
    this->nextOp = _op;
    if (_op != NULL)
        _op->_linkFrom(this);
}
void OpSoftmax::params(Array<RealTensor*>* _param, Array<RealTensor*>* _grad){}
SerializedTensors<REAL> OpSoftmax::state(){return SerializedTensors<REAL>();}
bool OpSoftmax::loadState(SerializedTensors<REAL>& ser){return false;}
RealTensor OpSoftmax::input()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (X.isNull())
        T_error("NN error: cannot retrieve input tensor as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return X;
}
RealTensor OpSoftmax::output()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (Y.isNull())
        T_error("NN error: cannot retrieve output tensor as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return Y;
}
RealTensor OpSoftmax::inputGrad()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdX.isNull())
        T_error("NN error: cannot retrieve input gradient as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return dLdX;
}
RealTensor OpSoftmax::outputGrad()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdY.isNull())
        T_error("NN error: cannot retrieve output gradient as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return dLdY;
}
void OpSoftmax::_linkFrom(GraphOp * _op)
{
    this->prevOp = _op;
}
void OpSoftmax::_passGrad(GraphOp * _prevOp, RealTensor * _grad)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_prevOp != prevOp)
        T_error("NN error: specified layer is not linked to the input slot of this layer.\n");
#endif
    /* passing gradients from current layer to previous layer */
    this->dLdX.moveStorage(*_grad);
}
void OpSoftmax::_passInput(GraphOp * _nextOp, RealTensor * _input)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_nextOp != nextOp)
        T_error("NN error: specified layer is not linked to the output slot of this layer.\n");
#endif
    /* set next layer input to be the output of this layer */
    /* softmax layer's gradient is relevant to its output, */
    /* so we cannot simply pass the output to the next layer */
    /* we need to keep it until backward is finished. */
    *_input = Y;
    //this->Y.moveStorage(*_input);
}

OpReLU::OpReLU() { nSlope = 0.0; nextOp = prevOp = NULL; }
void OpReLU::createOp(){}
const char * OpReLU::getOpClassName() const { return "OpReLU"; }



