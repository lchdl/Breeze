#include "nn/optimizer.h"

void OptimSGD::addOp(GraphOp* _op)
{
    Array<RealTensor*> _param, _grad;
    _op->params(&_param, &_grad);
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_param.size() != _grad.size())
        T_error("NN error: invalid trainable param and grad setting.\n");
#endif
    for (int i = 0; i < _param.size(); i++) {
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (_param[i]->shape() != _grad[i]->shape() ) {
            char p[64], g[64];
            _param[i]->shape().toStr(p);
            _grad[i]->shape().toStr(g);
            T_error("NN error: invalid trainable param and grad setting.\n"
                "param shape is %s, grad shape is %s.\n", p, g);
        }
        for (int t = 0; t < this->param.size(); t++) {
            if (_param[i] == this->param[t])
                T_error("NN error: found duplicated parameters.");
        }
        for (int t = 0; t < this->grad.size(); t++) {
            if (_grad[i] == this->grad[t])
                T_error("NN error: found duplicated gradients.");
        }
#endif
        this->param.append(_param[i]);
        this->grad.append(_grad[i]);
    }
}
void OptimSGD::zeroGrad()
{
    for (int i = 0; i < grad.size(); i++)
        grad[i]->fill_(RealTensor::type(0.0));
}
void OptimSGD::step(const REAL& lr)
{
    for (int t = 0; t < grad.size(); t++) {
        RealTensor* p = param[t];
        RealTensor* g = grad[t];
        int _numel = p->numel();
        /*printf("=====================\n");
        p->print("%.6f");
        g->print("%.6f");*/
        for (int i = 0; i < _numel; i++)
            p->at(i) -= lr * g->at(i);
    }
}
SerializedTensors<REAL> OptimSGD::state(){return SerializedTensors<REAL>();}
bool OptimSGD::loadState(SerializedTensors<REAL>& ser){return false;}


