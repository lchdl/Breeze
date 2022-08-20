#pragma once
#include "nn/nn_global.h"
#include "nn/graph_node.h"

/* helper function for 2/3D batch normalization */
void batchNorm(RealTensor & X, RealTensor & running_mu, RealTensor & running_sigma, 
    const REAL & momentum, RealTensor & sample_mu, RealTensor & sample_sigma, 
    RealTensor & gamma, RealTensor & beta, bool train);

class OpBatchNorm : public GraphOp
{
public:
    OpBatchNorm();
    void createOp(const int inChannels, const REAL momentum = REAL(0.1));

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
    RealTensor running_mu, running_sigma; /* store running average of mu and sigma, used in inference */
    RealTensor sample_mu, sample_sigma;   /* store mean and std of the input sample, used in training */
    RealTensor W, B; /* learnable parameters */
    RealTensor dLdW, dLdB; /* gradients for  */

    REAL momentum;
    TensorShape inShape, outShape;
    int inChannels;

protected:
    GraphOp* nextOp;
    GraphOp* prevOp;
};

typedef OpBatchNorm OpBatchNorm2d, OpBatchNorm3d;

