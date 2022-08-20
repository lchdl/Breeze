#pragma once
#include "nn/nn_global.h"
#include "nn/graph_node.h"

class OpLinear : public GraphOp
{
public:
    OpLinear();
    void createOp(int _inFeatures, int _outFeatures);

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
    RealTensor W, b;  /* layer weights & bias */
    RealTensor dLdW;  /* computed gradient for weights */
    RealTensor dLdb;  /* computed gradient for bias */
    RealTensor dLdX;  /* backward passing gradients to input for previous layer */
    RealTensor dLdY;  /* gradients fetched from next layer */

    TensorShape inShape, outShape;
    int inFeatures, outFeatures;
protected:
    GraphOp* nextOp;
    GraphOp* prevOp;
};

