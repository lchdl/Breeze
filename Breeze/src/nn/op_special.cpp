#include "nn/op_special.h"

OpReshape::OpReshape()
{
    nextOp = prevOp = NULL;
}
void OpReshape::createOp(const char * _inShape, const char * _outShape)
{
    TensorShape _inShapeNoBatch = TensorShape(_inShape);
    TensorShape _outShapeNoBatch = TensorShape(_outShape);
    this->inShape.zeros(_inShapeNoBatch.dim() + 1);
    this->outShape.zeros(_outShapeNoBatch.dim() + 1);
    for (int i = 1; i < this->inShape.dim(); i++)
        this->inShape[i] = _inShapeNoBatch[i - 1];
    for (int i = 1; i < this->outShape.dim(); i++)
        this->outShape[i] = _outShapeNoBatch[i - 1];
}
const char * OpReshape::getOpClassName() const
{
    return "OpReshape";
}
void OpReshape::forward(RealTensor * _input)
{
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
    /* fill in batch size */
    inShape[0] = X.shape()[0];
    outShape[0] = inShape[0];
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (inShape != X.shape()) {
        char a[64], b[64];
        inShape.toStr(a);
        X.shape().toStr(b);
        T_error("NN error: expected input have shape (%s), but actual shape is (%s).",
            a, b);
    }
    if (inShape.numel() != outShape.numel()) {
        char a[64], b[64];
        inShape.toStr(a);
        outShape.toStr(b);
        T_error("NN error: number of elements changed after reshape. Input shape is (%s), "
            "while output shape is (%s)", a, b);
    }
#endif
    /* for speed and saving memory i just invoke moveStorage() */
    X.moveStorage(Y);
    Y.reshape_(outShape);

    //Y.print("%4.2f");
}
void OpReshape::backward(GraphOp * _op)
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
    dLdY.moveStorage(dLdX);
    dLdX.reshape_(inShape);
}
void OpReshape::linkTo(GraphOp * _op)
{
    this->nextOp = _op;
    if (_op != NULL)
        _op->_linkFrom(this);
}
void OpReshape::params(Array<RealTensor*>* _param, Array<RealTensor*>* _grad){}
SerializedTensors<REAL> OpReshape::state(){return SerializedTensors<REAL>();}
bool OpReshape::loadState(SerializedTensors<REAL>& ser){return true;}
RealTensor OpReshape::input()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (X.isNull())
        T_error("NN error: cannot retrieve input tensor as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return X;
}
RealTensor OpReshape::output()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (Y.isNull())
        T_error("NN error: cannot retrieve output tensor as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return Y;
}
RealTensor OpReshape::inputGrad()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdX.isNull())
        T_error("NN error: cannot retrieve input gradient as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return dLdX;
}
RealTensor OpReshape::outputGrad()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdY.isNull())
        T_error("NN error: cannot retrieve output gradient as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return dLdY;
}
void OpReshape::_linkFrom(GraphOp * _op)
{
    this->prevOp = _op;
}
void OpReshape::_passGrad(GraphOp * _prevOp, RealTensor * _grad)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_prevOp != prevOp)
        T_error("NN error: specified layer is not linked to the input slot of this layer.\n");
#endif
    /* passing gradients from current layer to previous layer */
    /* here i used moveStorage to save memory space */
    this->dLdX.moveStorage(*_grad);
}
void OpReshape::_passInput(GraphOp * _nextOp, RealTensor * _input)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_nextOp != nextOp)
        T_error("NN error: specified layer is not linked to the output slot of this layer.\n");
#endif
    /* passing output of this layer to next layer */
    /* here i used moveStorage to save memory space */
    this->Y.moveStorage(*_input);
}

OpMultiOutput::OpMultiOutput(){}
void OpMultiOutput::createOp() { prevOp = NULL; }
const char * OpMultiOutput::getOpClassName() const { return "OpMultiOutput";}
void OpMultiOutput::forward(RealTensor * _input)
{
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
    this->inShape = X.shape();
    this->outShape = this->inShape;

    while (this->dLdYs.size() < this->nextOps.size())
        this->dLdYs.append(RealTensor());
    
    for (int i = 0; i < this->nextOps.size(); i++)
        this->dLdYs[i] = X;
}
void OpMultiOutput::backward(GraphOp * _op)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_op == NULL && nextOps.size() == 0) {
        T_error("NN error: backward op == NULL, which means the gradient "
            "flows from next ops, but no op is assigned.");
        return;
    }
#endif
    if (_op == this) {
        dLdX = RealTensor(outShape);       /* gradients will be back propagated from this  */
        dLdX.fill_(RealTensor::type(1.0)); /* layer so setting output gradients to all one */
    }
    else {
        /* passing gradients of the next layer to current layer */
        for (int i = 0; i < nextOps.size(); i++)
            nextOps[i]->_passGrad(this, &(this->dLdYs[i]));
#ifdef NN_ENABLE_RUNTIME_CHECKING
        for (int i = 0; i < nextOps.size(); i++) {
            if (outShape != dLdYs[i].shape()) {
                char os[64], ns[64];
                outShape.toStr(os);
                dLdYs[i].shape().toStr(ns);
                T_error("NN error: shape changed after backward(). "
                    "Old shape: (%s), new shape (%s).", os, ns);
            }
        }
#endif
        dLdX = RealTensor(this->inShape);
        dLdX.fill_(REAL(0.0));
        for (int i = 0; i < nextOps.size(); i++) {
            dLdX += dLdYs[i];
            dLdYs[i].purge();
        }
    }
}
void OpMultiOutput::linkTo(GraphOp * _op)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_op == NULL)
        T_error("NN error: op is NULL.");
    for (int i = 0; i < nextOps.size(); i++) {
        if (_op == nextOps[i]) {
            T_error("NN error: found duplicated op pointer in OpMultiOutput.");
        }
    }
#endif
    this->nextOps.append(_op);
    _op->_linkFrom(this);
}
void OpMultiOutput::params(Array<RealTensor*>* _param, Array<RealTensor*>* _grad){}
SerializedTensors<REAL> OpMultiOutput::state() { return SerializedTensors<REAL>(); }
bool OpMultiOutput::loadState(SerializedTensors<REAL>& ser) { return true; }
RealTensor OpMultiOutput::input()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (X.isNull())
        T_error("NN error: cannot retrieve input/output tensor as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return X;
}
RealTensor OpMultiOutput::output() { return this->input(); }
RealTensor OpMultiOutput::inputGrad()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdX.isNull())
        T_error("NN error: cannot retrieve input gradient as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return dLdX;
}
RealTensor OpMultiOutput::outputGrad()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    T_error("NN error: output gradient is undefined for OpMultiOutput, "
        "return null tensor instead.");
#endif
    return RealTensor();
}
void OpMultiOutput::_linkFrom(GraphOp * _op) { this->prevOp = _op; }
void OpMultiOutput::_passGrad(GraphOp * _prevOp, RealTensor * _grad)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_prevOp != prevOp)
        T_error("NN error: specified layer is not linked to the input slot of this layer.\n");
#endif
    /* passing gradients from current layer to previous layer */
    this->dLdX.moveStorage(*_grad);
}
void OpMultiOutput::_passInput(GraphOp * _nextOp, RealTensor * _input)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    bool found = false;
    for (int i = 0; i < nextOps.size(); i++) {
        if (_nextOp == nextOps[i])
            found = true;
    }
    if (!found)
        T_error("NN error: specified layer is not linked to the output slot of this layer.\n");
#endif
    /* set next layer input to be the output of this layer */
    *_input = X;
}

void T_channelSplit(RealTensor & tensor, const Array<int>& split, Array<RealTensor>& results) {
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (tensor.dim() < 2)
        T_error("Tensor dimension too low (<2).");
    int numChannels = tensor.shape()[1];
    int channelSum = 0;
    for (int i = 0; i < split.size(); i++) channelSum += split[i];
    if (channelSum != numChannels)
        T_error("Invalid parameter setting.");
#endif
    results.clear();
    int channelStart = 0;
    for (int n = 0; n < split.size(); n++) {
        TensorShape shape = tensor.shape();
        shape[1] = split[n];
        RealTensor newTensor(shape);
        int strideSrc0 = tensor.numel() / shape[0] / tensor.shape()[1];
        int strideSrc1 = tensor.numel() / shape[0];
        int strideDst0 = strideSrc0;
        int strideDst1 = newTensor.numel() / shape[0];
        for (int b = 0; b < tensor.shape()[0]; b++) {
            for (int ch = channelStart; ch < channelStart + split[n]; ch++)
                newTensor._dptrWrite(b*strideDst1 + (ch - channelStart)*strideDst0, strideDst0, tensor, b*strideSrc1 + ch * strideSrc0);
        }
        results.append(newTensor);
        channelStart += split[n];
    }
}

OpConcat::OpConcat()
{
    nextOp = prevOp1 = prevOp2 = NULL;
}
void OpConcat::createOp()
{
}
const char * OpConcat::getOpClassName() const
{
    return "OpConcat";
}
void OpConcat::forward(RealTensor * _input){ T_error("NN error: deprecated function.");}
void OpConcat::forward(RealTensor * _input1, RealTensor * _input2)
{
    if (_input1 != NULL && _input2 != NULL) { /* input comes from external source */
        X1 = *_input1;
        X2 = *_input2;
    }
    else if (_input1 == NULL && _input2 == NULL) { /* input comes from internal data flow */
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (prevOp1 == NULL || prevOp2 == NULL) {
            printf("NN error: cannot feed forward because not enough data has given.\n");
            return;
        }
#endif
        /* request input data from previous layer */
        prevOp1->_passInput(this, &(this->X1));
        prevOp2->_passInput(this, &(this->X2));
    }
    else
        T_error("invalid input.");

    this->inShape1 = X1.shape();
    this->inShape2 = X2.shape();
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (this->inShape1.isShapeValid() == false || this->inShape2.isShapeValid() == false)
        T_error("invalid shape.");
    if (this->inShape1.dim() < 2 || this->inShape2.dim() < 2)
        T_error("input dim too small (<2).");
    if (this->inShape1.dim() != this->inShape2.dim())
        T_error("dimension of the two input shapes are not equal.");
    for (int i = 0; i < this->inShape1.dim(); i++) {
        if (i == 1) continue;
        if (this->inShape1[i] != this->inShape2[i]) {
            T_error("shape not equal at dimension %d.", i + 1);
        }
    }
#endif
    this->outShape = this->inShape1;
    this->outShape[1] += this->inShape2[1];

    Y = T_channelConcat(X1, X2);
    X1.purge();
    X2.purge();
}
void OpConcat::backward(GraphOp * _op)
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
    Array<RealTensor> dLdXs;
    T_channelSplit(dLdY, T_buildArray<2, int>({ this->inShape1[1], this->inShape2[1] }), dLdXs);
    dLdXs[0].moveStorage(dLdX1);
    dLdXs[1].moveStorage(dLdX2);
}
void OpConcat::linkTo(GraphOp * _op)
{
    this->nextOp = _op;
    if (_op != NULL)
        _op->_linkFrom(this);
}
void OpConcat::params(Array<RealTensor*>* _param, Array<RealTensor*>* _grad){}
SerializedTensors<REAL> OpConcat::state(){return SerializedTensors<REAL>();}
bool OpConcat::loadState(SerializedTensors<REAL>& ser){ return true; }
RealTensor OpConcat::input()
{
    T_error("NN error: deprecated function.");
    return RealTensor();
}
RealTensor OpConcat::input1()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (X1.isNull())
        T_error("NN error: cannot retrieve input/output tensor as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return X1;
}
RealTensor OpConcat::input2()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (X2.isNull())
        T_error("NN error: cannot retrieve input/output tensor as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return X2;
}
RealTensor OpConcat::output()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (Y.isNull())
        T_error("NN error: cannot retrieve input/output tensor as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return Y;
}
RealTensor OpConcat::inputGrad()
{
    T_error("NN error: deprecated function.");
    return RealTensor();
}
RealTensor OpConcat::inputGrad1()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdX1.isNull())
        T_error("NN error: cannot retrieve input/output gradient as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return dLdX1;
}
RealTensor OpConcat::inputGrad2()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdX2.isNull())
        T_error("NN error: cannot retrieve input/output gradient as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return dLdX2;
}
RealTensor OpConcat::outputGrad()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdY.isNull())
        T_error("NN error: cannot retrieve input/output gradient as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return dLdY;
}
void OpConcat::_linkFrom(GraphOp * _op)
{
    if (this->prevOp1 == NULL)
        this->prevOp1 = _op;
    else if (this->prevOp2 == NULL)
        this->prevOp2 = _op;
    else
        T_error("cannot accept any input, input node is full.");
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (this->prevOp1 == this->prevOp2)
        T_error("NN error: two inputs are the same.");
#endif
}
void OpConcat::_passGrad(GraphOp * _prevOp, RealTensor * _grad)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_prevOp != prevOp1 && _prevOp != prevOp2)
        T_error("NN error: specified layer is not linked to the input slot of this layer.\n");
#endif
    /* passing gradients from current layer to previous layer */
    if (_prevOp == prevOp1)
        this->dLdX1.moveStorage(*_grad);
    if (_prevOp == prevOp2)
        this->dLdX2.moveStorage(*_grad);
}
void OpConcat::_passInput(GraphOp * _nextOp, RealTensor * _input)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_nextOp != nextOp)
        T_error("NN error: specified layer is not linked to the output slot of this layer.\n");
#endif
    /* passing output of this layer to next layer */
    /* here i used moveStorage to save memory space */
    this->Y.moveStorage(*_input);
}
