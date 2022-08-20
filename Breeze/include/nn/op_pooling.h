#pragma once
#include "nn/nn_global.h"
#include "nn/graph_node.h"

RealTensor maxPool2d(const RealTensor& tensor, const TensorShape& kernelSize, IntTensor& indices);
RealTensor maxUnpool2d(const RealTensor& tensor, const TensorShape& kernelSize, const IntTensor& indices, const TensorShape* outShape = NULL);
RealTensor avgPool2d(const RealTensor& tensor, const TensorShape& kernelSize);
RealTensor avgUnpool2d(const RealTensor& tensor, const TensorShape& kernelSize, const TensorShape* outShape = NULL);

RealTensor maxPool3d(const RealTensor& tensor, const TensorShape& kernelSize, IntTensor& indices);
RealTensor maxUnpool3d(const RealTensor & tensor, const TensorShape & kernelSize, const IntTensor & indices, const TensorShape * outShape);
RealTensor avgPool3d(const RealTensor& tensor, const TensorShape& kernelSize);
RealTensor avgUnpool3d(const RealTensor& tensor, const TensorShape& kernelSize, const TensorShape* outShape = NULL);

class OpMaxPool2d : public GraphOp {
public:

    OpMaxPool2d();
    void createOp(const char* _kernel);

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

    TensorShape inShape, outShape;
    TensorShape kernelSize;
    RealTensor X, Y;
    RealTensor dLdX, dLdY;
    IntTensor indices;
};
class OpAvgPool2d : public OpMaxPool2d{
public:
    void createOp(const char* _kernel);

    virtual void forward(RealTensor* _input = NULL);
    virtual const char* getOpClassName() const;
    virtual void backward(GraphOp* _op = NULL);
};

class OpMaxPool3d : public OpMaxPool2d {
public:
    void createOp(const char* _kernel);

    virtual const char* getOpClassName() const;
    virtual void forward(RealTensor* _input = NULL);
    virtual void backward(GraphOp* _op = NULL);
};
class OpAvgPool3d : public OpMaxPool2d {
public:
    void createOp(const char* _kernel);

    virtual const char* getOpClassName() const;
    virtual void forward(RealTensor* _input = NULL);
    virtual void backward(GraphOp* _op = NULL);
};


