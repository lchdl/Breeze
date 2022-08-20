#pragma once
#include "nn/nn_global.h"
#include "nn/graph_node.h"

class OpMeanSquareErrorLoss : public GraphOp
{
public:
    OpMeanSquareErrorLoss();
    void createOp();
    void setTarget(const RealTensor& T);
    RealTensor::type item();

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
    RealTensor X, Y, T;  /* input, output, target */
    RealTensor dLdX, dLdY;
    TensorShape inShape, outShape;
protected:
    GraphOp* nextOp;
    GraphOp* prevOp;
};

/* for example input shape: (b,c,x,y,z) */
/*    expected label shape: (b,x,y,z)   */
/* NOTE: input must be raw logits instead of probabilities! */
/* link: https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/ */
class OpCrossEntropyLoss : public GraphOp
{
public:
    OpCrossEntropyLoss();
    void createOp();
    void setTarget(const IntTensor& L); /* label */
    RealTensor::type item();

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
    /* convert label as one-hot representation */
    /* L: "5,2,3", label 0~3 (0: bg, 1~3: fg, 4 channels) => "5,4,2,3" */
    RealTensor labelAsOnehot(const IntTensor& L, const int& channels);

protected:
    RealTensor X, S, Y;  /* input, softmax of input, output loss */
    IntTensor L;         /* label */
    RealTensor dLdX, dLdY;
    TensorShape inShape, outShape;
protected:
    GraphOp* nextOp;
    GraphOp* prevOp;

};

/* calculate Dice loss using the following equation */
/* dice = 2|X*T|/(|X|+|T|) */
/* X: input, T: target {0,1}. X,T must have the same shape. */
class OpHardDiceLoss : public GraphOp
{
public:
    OpHardDiceLoss();
    void createOp();
    void setTarget(const RealTensor& T);
    RealTensor::type item();

    virtual const char* getOpClassName() const;
    virtual void forward(RealTensor* _input = NULL);
    virtual void backward(GraphOp* _op = NULL);
    virtual void linkTo(GraphOp* _op);
    virtual void params(Array<RealTensor*>* _param, Array<RealTensor*>* _grad);
    virtual SerializedTensors<REAL> state();
    virtual bool loadState(SerializedTensors<REAL>& ser);

public:
    virtual void _linkFrom(GraphOp* _op);
    virtual void _passGrad(GraphOp* _prevOp, RealTensor* _grad);
    virtual void _passInput(GraphOp* _nextOp, RealTensor* _input);

protected:
    RealTensor X, Y, T; /* input, dice loss value, target */
    RealTensor dLdX, dLdY;
    TensorShape inShape, outShape;
    GraphOp* nextOp;
    GraphOp* prevOp;
};






