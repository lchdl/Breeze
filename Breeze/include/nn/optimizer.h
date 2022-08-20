#pragma once
#include "nn/nn_global.h"
#include "nn/graph_node.h"

class Optimizer {
public:
    virtual void addOp(GraphOp* _op) = 0; /* adding a layer for optimization */
    virtual void zeroGrad() = 0;
    virtual void step(const REAL& _lr) = 0;

    virtual SerializedTensors<REAL> state() = 0;
    virtual bool loadState(SerializedTensors<REAL>& ser) = 0;
};

/* default stochastic gradient descent optimizer */
class OptimSGD : public Optimizer{
public:
    virtual void addOp(GraphOp* _op); /* adding a layer for optimization */
    virtual void zeroGrad();
    virtual void step(const REAL& _lr);

    virtual SerializedTensors<REAL> state();
    virtual bool loadState(SerializedTensors<REAL>& ser);

protected:
    /* all trainable params and their corresponding gradients */
    Array<RealTensor*> param, grad;
};

