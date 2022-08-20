#pragma once
#include "nn/nn_global.h"
#include "nn/graph_node.h"

/* defines activation functions */

class OpLeakyReLU : public GraphOp
{
public:
    OpLeakyReLU();
    void createOp(const REAL& nSlope = REAL(0.1));

    virtual const char* getOpClassName() const;
    virtual void forward(RealTensor* _input = NULL);
    virtual void backward(GraphOp* _op = NULL);
    virtual void linkTo(GraphOp* _op);
    virtual void params(Array<RealTensor*>* _param, Array<RealTensor*>* _grad);
    virtual SerializedTensors<REAL> state();
    virtual bool loadState(SerializedTensors<REAL>& ser);

    virtual RealTensor input();
    virtual RealTensor output();
    virtual RealTensor inputGrad();
    virtual RealTensor outputGrad();

public:
    virtual void _linkFrom(GraphOp* _op);
    virtual void _passGrad(GraphOp* _prevOp, RealTensor* _grad);
    virtual void _passInput(GraphOp* _nextOp, RealTensor* _input);

protected:
    RealTensor X, Y;  /* input & output */
    RealTensor dLdY;  /* gradients fetched from next layer */
    RealTensor dLdX;  /* backward passing gradients to input for previous layer */
    REAL nSlope;      /* slope when x<0 */
    TensorShape inShape, outShape;
protected:
    GraphOp* nextOp;
    GraphOp* prevOp;
};
class OpReLU : public OpLeakyReLU {
public:
    OpReLU();
    void createOp();
    virtual const char* getOpClassName() const;
};
class OpSoftmax : public GraphOp
{
public:
    OpSoftmax();
    void createOp();

    virtual const char* getOpClassName() const;
    virtual void forward(RealTensor* _input = NULL);
    virtual void backward(GraphOp* _op = NULL);
    virtual void linkTo(GraphOp* _op);
    virtual void params(Array<RealTensor*>* _param, Array<RealTensor*>* _grad);
    virtual SerializedTensors<REAL> state();
    virtual bool loadState(SerializedTensors<REAL>& ser);

    virtual RealTensor input();
    virtual RealTensor output();
    virtual RealTensor inputGrad();
    virtual RealTensor outputGrad();

public:
    virtual void _linkFrom(GraphOp* _op);
    virtual void _passGrad(GraphOp* _prevOp, RealTensor* _grad);
    virtual void _passInput(GraphOp* _nextOp, RealTensor* _input);

protected:
    RealTensor X, Y;  /* input & output */
    RealTensor dLdY;  /* gradients fetched from next layer */
    RealTensor dLdX;  /* backward passing gradients to input for previous layer */
    TensorShape inShape, outShape;

protected:
    GraphOp* nextOp;
    GraphOp* prevOp;
};

