#pragma once
#include "basedefs.h"
#include "nn/nn_global.h"
#include "nn/random.h"
#include "nn/tensor.h"
#include "nn/tensor_types.h"
#include "nn/tensor_conv.h"

class GraphOp 
{
public:
    /* * * * * * * * * * * * * * * * * * * * * * * */
    /* forward computation:                        */
    /* step 1: setting input ("X") properly        */
    /* step 2: determine "inShape" and "outShape", */
    /*         create output tensor ("Y") based on */
    /*         input.                              */
    /* step 3: compute "Y" based on "X"            */
    /* * * * * * * * * * * * * * * * * * * * * * * */
    virtual void forward(RealTensor* _input = NULL) = 0;
    /* * * * * * * * * * * * * * * * * * * * * * * */
    /* accumulate gradient                         */
    /* step 1: fetch output gradients ("dLdY")     */
    /* step 2: compute and accumulate gradients for*/
    /*         trainable parameters ("dLd*")       */
    /* step 3: compute input gradients ("dLdX"),   */
    /*         forming BP chain                    */
    /* step 4: purge "dLdY" to save memory.        */
    /* * * * * * * * * * * * * * * * * * * * * * * */
    virtual void backward(GraphOp* _op = NULL) = 0;
    /* * * * * * * * * * * * * * * * * * * * * * * */
    /* operation needs to be done when linking     */
    /* other layers to the output of the current   */
    /* layer                                       */
    /* * * * * * * * * * * * * * * * * * * * * * * */
    virtual void linkTo(GraphOp* _op) = 0;
    /* collect trainable params */
    virtual void params(Array<RealTensor*>* _param, Array<RealTensor*>* _grad) = 0;

    virtual RealTensor input() = 0;
    virtual RealTensor output() = 0;
    virtual RealTensor inputGrad() = 0;
    virtual RealTensor outputGrad() = 0;


    GraphOp() { _training = true; }
    virtual ~GraphOp() {}


public:
    /* save/load state */
    virtual SerializedTensors<REAL> state() = 0;
    virtual bool loadState(SerializedTensors<REAL>& ser) = 0;

    /* return class name */
    virtual const char* getOpClassName() const = 0;

public:
    virtual void train(const bool& state = true){_training = state;}
    virtual void eval() {_training = false;}
    virtual bool isTraining() const {return _training;}

public:
    /* * * * * * * * * * * * * * * * * * * * * * * */
    /* operation needs to be done when receiving   */
    /* linking request from other layers           */
    /* * * * * * * * * * * * * * * * * * * * * * * */
    virtual void _linkFrom(GraphOp* _op) = 0;
    /* passing gradients to previous layer */
    /* NOTE: use Tensor::moveStorage() to save memory */
    virtual void _passGrad(GraphOp* _prevOp, RealTensor* _grad) = 0;
    /* passing input to next layer */
    /* NOTE: use Tensor::moveStorage() to save memory */
    virtual void _passInput(GraphOp* _nextOp, RealTensor* _input) = 0;

protected:
    bool _training;

};

