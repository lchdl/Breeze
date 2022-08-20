#pragma once
#include "nn/nn_global.h"
#include "nn/graph_node.h"

class OpReshape : public GraphOp
{
public:
    OpReshape();
    void createOp(const char* _inShape, const char* _outShape);

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
    RealTensor X, Y;  /* input, output */
    RealTensor dLdX, dLdY;
    TensorShape inShape, outShape;
protected:
    GraphOp* nextOp;
    GraphOp* prevOp;
};
class OpMultiOutput : public GraphOp
{
public:
    OpMultiOutput();
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
    RealTensor X;  /* input, output */
    RealTensor dLdX;
    Array<RealTensor> dLdYs;

    TensorShape inShape, outShape;
protected:
    Array<GraphOp*> nextOps;
    GraphOp* prevOp;
};

template <typename TensorType>
TensorType T_channelConcat(Array<TensorType*> & tensors) {
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (tensors.size() == 0) {
        T_error("NN error: no tensor to concat.");
    }
    else {
        TensorShape _shape = tensors[0]->shape();
        if (_shape.dim()<2) {
            T_error("NN error: tensor dimension too low (<2) for channel concatenation.");
        }
        for (int i = 1; i < tensors.size(); i++) {
            if (tensors[i]->shape().dim() != _shape.dim()) {
                T_error("NN error: found tensors with different dimension. "
                    "Channel concatenation will fail.");
            }
            for (int j = 0; j < _shape.dim(); j++) {
                if (j == 1) continue; /* ignore channel dimension */
                if (tensors[i]->shape()[j] != _shape[j]) {
                    char a[64], b[64];
                    _shape.toStr(a);
                    tensors[i]->shape().toStr(b);
                    T_error("NN error: tensors can only have different shape "
                        "on channel dimension when concatenating. Got tensor "
                        "with shape (%s) and (%s).", a, b);
                }
            }
        }
    }
#endif
    int C = 0, D = tensors[0]->dim();
    for (int i = 0; i < tensors.size(); i++)
        C += tensors[i]->shape()[1];

    TensorShape resultShape = tensors[0]->shape();
    resultShape[1] = C;

#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (resultShape.isShapeValid() == false)
        T_error("NN error: result shape is invalid.");
#endif

    TensorShape resultShapeT = resultShape;
    T_swap(resultShapeT[0], resultShapeT[1]);

    /* start channel concat */
    TensorType resultTensorT(resultShapeT);
    int iStart = 0;
    for (int i = 0; i < tensors.size(); i++) {
        TensorType t = T_swapBatchChannel(*tensors[i]);
        resultTensorT._dptrWrite(iStart, t.numel(), t, 0);
        iStart += t.numel();
    }

    return T_swapBatchChannel(resultTensorT);
}
template <typename TensorType>
TensorType T_channelConcat(TensorType & tensor1, TensorType & tensor2) {
    Array<TensorType*> arr = T_buildArray<2, TensorType*>({ &tensor1, &tensor2 });
    return T_channelConcat(arr);
}

void T_channelSplit(RealTensor& tensor, const Array<int>& split, Array<RealTensor>& results);

/* concatenate two tensors */
class OpConcat : public GraphOp
{
public:
    OpConcat();
    void createOp();
    virtual const char* getOpClassName() const;
    virtual void forward(RealTensor* _input = NULL); /* deprecated */
    virtual void forward(RealTensor* _input1 = NULL, RealTensor* _input2 = NULL);
    virtual void backward(GraphOp* _op = NULL);
    virtual void linkTo(GraphOp* _op);
    virtual void params(Array<RealTensor*>* _param, Array<RealTensor*>* _grad);
    virtual SerializedTensors<REAL> state();
    virtual bool loadState(SerializedTensors<REAL>& ser);

    virtual RealTensor input(); /* deprecated */
    virtual RealTensor input1();
    virtual RealTensor input2();
    virtual RealTensor output();
    virtual RealTensor inputGrad(); /* deprecated */
    virtual RealTensor inputGrad1();
    virtual RealTensor inputGrad2();
    virtual RealTensor outputGrad();

public:
    virtual void _linkFrom(GraphOp* _op);
    virtual void _passGrad(GraphOp* _prevOp, RealTensor* _grad);
    virtual void _passInput(GraphOp* _nextOp, RealTensor* _input);


protected:
    TensorShape inShape1, inShape2, outShape;
    RealTensor X1, X2, Y;  /* input, output */
    RealTensor dLdX1, dLdX2, dLdY;
    GraphOp* nextOp;
    GraphOp* prevOp1, * prevOp2;
};

