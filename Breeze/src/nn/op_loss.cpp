#include "nn/op_loss.h"

OpMeanSquareErrorLoss::OpMeanSquareErrorLoss() { nextOp = prevOp = NULL; }
const char * OpMeanSquareErrorLoss::getOpClassName() const
{
    return "OpMeanSquareErrorLoss";
}
void OpMeanSquareErrorLoss::forward(RealTensor * _input)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (T.isNull())
        T_error("NN error: target in the MSE layer is not set. "
            "Use OpMSE::setTarget(...) to set optimize target "
            "before calling OpMSE::forward(...). ");
    TensorShape targetShape = T.shape();
#endif
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
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (targetShape.isEqual(X.shape()) == false) {
            char ts[64], ns[64];
            targetShape.toStr(ts);
            X.shape().toStr(ns);
            T_error("NN error: input shape is different from target shape. "
                "Target shape is: (%s), input shape is: (%s).", ts, ns);
        }
#endif
    }
    inShape = X.shape();
    outShape.scalar();
    Y = RealTensor(outShape);
    /* * * * * * * * * * * * * * * * * * * * */
    /* step 2: compute output based on input */
    /* * * * * * * * * * * * * * * * * * * * */
    RealTensor D = X - T;
    D = T_hprod(D, D);
    Y[0] = T_sum(D) / X.numel();
/*
    system("cls");
    X.print();*/
}
void OpMeanSquareErrorLoss::backward(GraphOp * _op)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_op == NULL && nextOp == NULL) {
        T_error("NN error: backward op set to NULL, which means "
            "the gradient flows from next op, but next op is NULL.");
        return;
    }
#endif
    typedef RealTensor::type ElemType;
    /* * * * * * * * * * * * */
    /* step 2: compute dL/dY */
    /* * * * * * * * * * * * */
    if (_op == this)              /* gradients will be back propagated from this layer */
        dLdY[0] = ElemType(1.0);  /* so setting output gradients to all one */
    else{
        /* passing gradients of the next layer to current layer */
#ifdef NN_ENABLE_RUNTIME_CHECKING
        TensorShape oldShape = dLdY.shape();
#endif
        nextOp->_passGrad(this, &(this->dLdY));
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (oldShape.isEqual(dLdY.shape()) == false) {
            char os[64], ns[64];
            oldShape.toStr(os);
            dLdY.shape().toStr(ns);
            T_error("NN error: shape changed after backward(). "
                "Old shape is: (%s), new shape is: (%s).", os, ns);
        }
#endif
    }
    /* * * * * * * * * * * * */
    /* step 3: compute dL/dX */
    /* * * * * * * * * * * * */
    dLdX = RealTensor(inShape);
    dLdX.fill_(RealTensor::type(0.0));
    int _numel = X.numel();
    for (int i = 0; i < _numel; i++)
        dLdX[i] += dLdY[0] * ElemType(2.0) * (X[i] - T[i]) / _numel;
    dLdY.purge();
}
void OpMeanSquareErrorLoss::linkTo(GraphOp * _op)
{
    T_error("NN error: cannot link a loss op to any other op! "
        "A loss op is the end of a graph.");
}
void OpMeanSquareErrorLoss::_linkFrom(GraphOp * _op)
{
    this->prevOp = _op;
}
void OpMeanSquareErrorLoss::setTarget(const RealTensor & T)
{
    this->T = T;
    X = RealTensor(T.shape());
    Y = RealTensor("0"); /* scalar */
    dLdX = RealTensor(T.shape());
    dLdY = RealTensor("0");

}
RealTensor::type OpMeanSquareErrorLoss::item()
{
    return this->Y[0];
}
void OpMeanSquareErrorLoss::createOp()
{
    nextOp = prevOp = NULL;
}
void OpMeanSquareErrorLoss::_passGrad(GraphOp * _prevOp, RealTensor * _grad)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_prevOp != prevOp)
        T_error("NN error: specified layer is not linked to the input slot of this layer.\n");
#endif
    /* passing gradients from current layer to previous layer */
    this->dLdX.moveStorage(*_grad);
}
void OpMeanSquareErrorLoss::_passInput(GraphOp * _nextOp, RealTensor * _input)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_nextOp != nextOp)
        T_error("NN error: specified layer is not linked to the output slot of this layer.\n");
#endif
    /* set next layer input to be the output of this layer */
    this->Y.moveStorage(*_input);
}
void OpMeanSquareErrorLoss::params(Array<RealTensor*>* _param, Array<RealTensor*>* _grad) {}
SerializedTensors<REAL> OpMeanSquareErrorLoss::state(){return SerializedTensors<REAL>();}
bool OpMeanSquareErrorLoss::loadState(SerializedTensors<REAL>& ser){return false;}
RealTensor OpMeanSquareErrorLoss::input()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (X.isNull())
        T_error("NN error: cannot retrieve input tensor as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return X;
}
RealTensor OpMeanSquareErrorLoss::output()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (Y.isNull())
        T_error("NN error: cannot retrieve output tensor as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return Y;
}
RealTensor OpMeanSquareErrorLoss::inputGrad()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdX.isNull())
        T_error("NN error: cannot retrieve input gradient as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return dLdX;
}
RealTensor OpMeanSquareErrorLoss::outputGrad()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdY.isNull())
        T_error("NN error: cannot retrieve output gradient as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return dLdY;
}

OpCrossEntropyLoss::OpCrossEntropyLoss()
{
    nextOp = prevOp = NULL;
}
void OpCrossEntropyLoss::createOp()
{
    Y = RealTensor("0");
    dLdY = RealTensor("0");
}
void OpCrossEntropyLoss::setTarget(const IntTensor & L)
{
    this->L = L;
}
RealTensor::type OpCrossEntropyLoss::item()
{
    return Y[0];
}
const char * OpCrossEntropyLoss::getOpClassName() const
{
    return "OpCrossEntropyLoss";
}
void OpCrossEntropyLoss::forward(RealTensor * _input)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (L.isNull())
        T_error("NN error: label is not set. Use setTarget(...) to set label "
            "before calling forward(...). ");
#endif
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
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (X.dim() < 2)
            T_error("NN error: input should be at least two dimensional.");
        if (L.dim() < 1)
            T_error("NN error: label should be at least one dimensional.");
        if (X.shape()[0] != L.shape()[0]) {
            char xs[64], ls[64];
            X.shape().toStr(xs);
            L.shape().toStr(ls);
            T_error("NN error: batch size of input and label are not equal. "
                "Input shape: (%s), label shape: (%s).", xs, ls);
        }
        if (X.dim() - 1 != L.dim()) {
            T_error("NN error: input dimension is %d, expected label has "
                "dimension %d, but got %d.", X.dim(), X.dim() - 1, L.dim());
        }
        for (int i = 2; i < X.dim(); i++) {
            if (L.shape()[i - 1] != X.shape()[i]) {
                char xs[64], ls[64];
                X.shape().toStr(xs);
                L.shape().toStr(ls);
                T_error("NN error: input shape and label shape are not "
                    "compatible. Input shape: (%s), label shape: (%s).", xs, ls);
            }
        }
#endif
    }
    /* * * * * * * * * * * * * * * * * * * * */
    /* step 2: compute output based on input */
    /* * * * * * * * * * * * * * * * * * * * */
    inShape = X.shape();
    outShape.scalar();
    Y = RealTensor(outShape);
    dLdX = RealTensor(X.shape()); /* create gradient tensor from input */
    S = X.softmax(2); /* taking softmax in the channel axis, remember axis count from 1 */
    
    /* compute loss value */
    /* 1. build axes mapping */
    Array<int> axesTpose;
    for (int i = 1; i <= L.dim() + 1; i++) axesTpose.append(i);
    int tempAxis = axesTpose[L.dim()];
    axesTpose[L.dim()] = axesTpose[1];
    axesTpose[1] = tempAxis;
    /* transpose softmax */
    RealTensor St = S.transpose(axesTpose);
    /* then compute loss */
    int numVecs = St.numel() / (St.shape()[St.dim() - 1]);
    int channels = X.shape()[1];
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (L.max() >= channels)
        T_error("NN error: invalid label value \"%d\", accepted labels are 0~%d.", L.max(), channels - 1);
#endif
    REAL totalLoss = REAL(0.0);
    const REAL minProb = REAL(1e-20);
    for (int i = 0; i < numVecs; i++) {
        const RealTensor::type* ptrbase = St.dptr();
        const RealTensor::type* ptrvec = &(ptrbase[i*channels]);
        REAL S = ptrvec[L[i]];
        if (S < minProb)
            S = minProb;
        totalLoss += -Log(S);
    }
    REAL averageLoss = totalLoss / numVecs;
    Y[0] = averageLoss;
}
void OpCrossEntropyLoss::backward(GraphOp * _op)
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
        dLdY[0] = ElemType(1.0);    /* so setting output gradients to all one */
    }
    else {
        /* passing gradients of the next layer to current layer */
#ifdef NN_ENABLE_RUNTIME_CHECKING
        TensorShape oldShape = dLdY.shape();
#endif
        nextOp->_passGrad(this, &(this->dLdY));
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (oldShape.isEqual(dLdY.shape()) == false) {
            char os[64], ns[64];
            oldShape.toStr(os);
            dLdY.shape().toStr(ns);
            T_error("NN error: shape changed after backward(). "
                "Old shape is: (%s), new shape is: (%s).", os, ns);
        }
#endif
    }
    /* * * * * * * * * * * * */
    /* step 3: compute dL/dX */
    /* * * * * * * * * * * * */
    int channels = X.shape()[1];
    RealTensor onehot = labelAsOnehot(this->L, channels);
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (L.max() >= channels)
        T_error("NN error: invalid label value \"%d\", accepted labels are 0~%d.", L.max(), channels - 1);
    if (onehot.shape() != dLdX.shape()) {
        char a[64], b[64];
        onehot.shape().toStr(a);
        dLdX.shape().toStr(b);
        T_error("NN error: shapes of onehot (%s) and dLdX (%s) is not equal.", a, b);
    }
    if (onehot.shape() != S.shape()) {
        char a[64], b[64];
        onehot.shape().toStr(a);
        S.shape().toStr(b);
        T_error("NN error: shapes of onehot and S is not equal.", a, b);
    }
#endif
    dLdX = dLdY[0] * (S - onehot);
    dLdY.purge();
}
void OpCrossEntropyLoss::linkTo(GraphOp * _op)
{
    T_error("NN error: cannot link a loss op to any other op! "
        "A loss op is the end of a graph.");
}
void OpCrossEntropyLoss::params(Array<RealTensor*>* _param, Array<RealTensor*>* _grad){}
SerializedTensors<REAL> OpCrossEntropyLoss::state(){return SerializedTensors<REAL>();}
bool OpCrossEntropyLoss::loadState(SerializedTensors<REAL>& ser){return false;}
RealTensor OpCrossEntropyLoss::input()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (X.isNull())
        T_error("NN error: cannot retrieve input tensor as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return X;
}
RealTensor OpCrossEntropyLoss::output()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (Y.isNull())
        T_error("NN error: cannot retrieve output tensor as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return Y;
}
RealTensor OpCrossEntropyLoss::inputGrad()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdX.isNull())
        T_error("NN error: cannot retrieve input gradient as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return dLdX;
}
RealTensor OpCrossEntropyLoss::outputGrad()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdY.isNull())
        T_error("NN error: cannot retrieve output gradient as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return dLdY;
}
void OpCrossEntropyLoss::_linkFrom(GraphOp * _op)
{
    this->prevOp = _op;
}
void OpCrossEntropyLoss::_passGrad(GraphOp * _prevOp, RealTensor * _grad)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_prevOp != prevOp)
        T_error("NN error: specified layer is not linked to the input slot of this layer.\n");
#endif
    /* passing gradients from current layer to previous layer */
    this->dLdX.moveStorage(*_grad);
}
void OpCrossEntropyLoss::_passInput(GraphOp * _nextOp, RealTensor * _input)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_nextOp != nextOp)
        T_error("NN error: specified layer is not linked to the output slot of this layer.\n");
#endif
    /* set next layer input to be the output of this layer */
    this->Y.moveStorage(*_input);
}
RealTensor OpCrossEntropyLoss::labelAsOnehot(const IntTensor & L, const int& channels)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (L.dim() < 1)
        T_error("NN error: label dimension should be at least 1.");
#endif
    /* build axes mapping */
    Array<int> axesTpose;
    for (int i = 1; i <= L.dim()+1; i++) axesTpose.append(i);
    int tempAxis = axesTpose[L.dim()];
    axesTpose[L.dim()] = axesTpose[1];
    axesTpose[1] = tempAxis;
    /* build one-hot tensor */
    Array<int> _onehotShape;
    for (int i = 0; i < L.dim(); i++) _onehotShape.append(L.shape()[i]);
    _onehotShape.append(channels);
    RealTensor onehotTensor = RealTensor(TensorShape(_onehotShape));
    /* fill data */
    int numVecs = L.numel();
    Array<RealTensor::type> channelVec;
    for (int i = 0; i < channels; i++) channelVec.append(RealTensor::type(0)); /* reserve slots */
    int elemId = 0;
    for (int i = 0; i < numVecs; i++) {
        for (int ch = 0; ch < channels; ch++) {
            if (ch == L[elemId])
                channelVec[ch] = RealTensor::type(1);
            else
                channelVec[ch] = RealTensor::type(0);
        }
        onehotTensor._dptrWrite(i*channels, channels, channelVec.data());
        elemId++;
    }
    /* transpose channel */
    return onehotTensor.transpose(axesTpose);
}

OpHardDiceLoss::OpHardDiceLoss() { nextOp = prevOp = NULL; }
void OpHardDiceLoss::createOp() 
{ 
    nextOp = prevOp = NULL; 
    Y = RealTensor("0");
    dLdY = RealTensor("0");
}
void OpHardDiceLoss::setTarget(const RealTensor & T){this->T = T;}
RealTensor::type OpHardDiceLoss::item(){return Y[0];}
const char * OpHardDiceLoss::getOpClassName() const
{
    return "OpHardDiceLoss";
}
void OpHardDiceLoss::forward(RealTensor * _input)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (T.isNull())
        T_error("NN error: target is not set. Use setTarget(...) to set target "
            "before calling forward(...). ");
#endif
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
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (this->T.shape() != X.shape()) {
            char s1[64], s2[64];
            this->T.shape().toStr(s1);
            X.shape().toStr(s2);
            T_error("NN error: target shape (%s) and input (%s) not equal.", s1, s2);
        }
#endif
    }
    inShape = X.shape();
    outShape.scalar();
    Y = RealTensor(outShape);
    /* i compute the Dice loss without considering the batch size, */
    /* that is, i treat the whole input tensor as a single object */
    /* and calculate its Dice, dont know if doing this is theoretically */
    /* correct but it seems a small problem and should not affect the */
    /* result too much */
    const REAL smooth = REAL(1e-8);
    Y[0] = (2 * T_hprod(X, T).sum()) / (X.sum() + T.sum() + smooth);
}
void OpHardDiceLoss::backward(GraphOp * _op)
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
        dLdY = RealTensor(outShape); /* gradients will be back propagated from this layer */
        dLdY[0] = ElemType(1.0);     /* so setting output gradients to all one */
    }
    else {
        /* passing gradients of the next layer to current layer */
#ifdef NN_ENABLE_RUNTIME_CHECKING
        TensorShape oldShape = dLdY.shape();
#endif
        nextOp->_passGrad(this, &(this->dLdY));
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (oldShape.isEqual(dLdY.shape()) == false) {
            char os[64], ns[64];
            oldShape.toStr(os);
            dLdY.shape().toStr(ns);
            T_error("NN error: shape changed after backward(). "
                "Old shape is: (%s), new shape is: (%s).", os, ns);
        }
#endif
    }
    /* * * * * * * * * * * * */
    /* step 3: compute dL/dX */
    /* * * * * * * * * * * * */
    const REAL smooth = REAL(1e-8);
    dLdX = -(2 * T - Y[0]) / (X.sum() + T.sum() + smooth);
    dLdY.purge();

    //dLdX.print();
}
void OpHardDiceLoss::linkTo(GraphOp * _op)
{
    T_error("NN error: cannot link a loss op to any other op! "
        "A loss op is the end of a graph.");
}
void OpHardDiceLoss::params(Array<RealTensor*>* _param, Array<RealTensor*>* _grad){}
SerializedTensors<REAL> OpHardDiceLoss::state(){return SerializedTensors<REAL>();}
bool OpHardDiceLoss::loadState(SerializedTensors<REAL>& ser){return false;}
void OpHardDiceLoss::_linkFrom(GraphOp * _op)
{
    this->prevOp = _op;
}
void OpHardDiceLoss::_passGrad(GraphOp * _prevOp, RealTensor * _grad)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_prevOp != prevOp)
        T_error("NN error: specified layer is not linked to the input slot of this layer.\n");
#endif
    /* passing gradients from current layer to previous layer */
    this->dLdX.moveStorage(*_grad);
}
void OpHardDiceLoss::_passInput(GraphOp * _nextOp, RealTensor * _input)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_nextOp != nextOp)
        T_error("NN error: specified layer is not linked to the output slot of this layer.\n");
#endif
    /* set next layer input to be the output of this layer */
    this->Y.moveStorage(*_input);
}


