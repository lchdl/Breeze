#pragma once
#include "nn/nn_global.h"
#include "nn/graph_node.h"

class OpConv2d : public GraphOp
{
public:
    OpConv2d();
    void createOp(const int _inChannels, const int _outChannels, 
        const char* _kernel, const char* _stride, const char* _padding);

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
    GraphOp* nextOp;
    GraphOp* prevOp;
    TensorShape inShape, padShape, outShape;
    RealTensor X, Xpad, K, b, Y; /* input, kernel weights, bias, output */
    RealTensor dLdX, dLdK, dLdb, dLdY;

    int inChannels, outChannels;
    Array<int> kernelSize, strideSize, padSize;
};
class OpConvTranspose2d : public GraphOp 
{
public:
    OpConvTranspose2d();
    void createOp(const int _inChannels, const int _outChannels, 
        const char* _kernel, const char* _stride);

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
    GraphOp* nextOp;
    GraphOp* prevOp;

    int inChannels, outChannels;
    TensorShape inShape, outShape;
    Array<int> kernelSize, strideSize;

    RealTensor X, Xc, Y, K, b;
    RealTensor dLdY, dLdK, dLdb, dLdX;
};
class OpConv3d : public GraphOp{
public:
    OpConv3d();
    void createOp(const int _inChannels, const int _outChannels,
        const char* _kernel, const char* _stride, const char* _padding);

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
    GraphOp* nextOp;
    GraphOp* prevOp;
    TensorShape inShape, padShape, outShape;
    RealTensor X, Xpad, K, b, Y; /* input, kernel weights, bias, output */
    RealTensor dLdX, dLdK, dLdb, dLdY;

    int inChannels, outChannels;
    Array<int> kernelSize, strideSize, padSize;
};
class OpConvTranspose3d : public GraphOp{
public:
    OpConvTranspose3d();
    void createOp(const int _inChannels, const int _outChannels,
        const char* _kernel, const char* _stride);

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
    GraphOp* nextOp;
    GraphOp* prevOp;

    int inChannels, outChannels;
    TensorShape inShape, outShape;
    Array<int> kernelSize, strideSize;

    RealTensor X, Xc, Y, K, b;
    RealTensor dLdY, dLdK, dLdb, dLdX;
};

