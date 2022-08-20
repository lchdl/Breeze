#include "nn/nn.h"

class DemoNet {
protected:
    OpConv2d conv1, conv2, conv3, conv4, conv5;
    OpBatchNorm2d bn1, bn2, bn3, bn4, bn5;

    OpMultiOutput split;
    OpConcat merge;
    OpMeanSquareErrorLoss mseloss;
    OptimSGD optim;
public:
    DemoNet() {
        conv1.createOp(1, 2, "3,3", "1,1", "1,1");
        conv2.createOp(2, 1, "3,3", "1,1", "1,1");
        conv3.createOp(2, 1, "3,3", "1,1", "1,1");
        conv4.createOp(2, 1, "3,3", "1,1", "1,1");
        split.createOp();
        merge.createOp();
        mseloss.createOp();

        optim.addOp(&conv1);
        optim.addOp(&conv2);
        optim.addOp(&conv3);
        optim.addOp(&conv4);

        conv1.linkTo(&split);
        split.linkTo(&conv2);
        split.linkTo(&conv3);
        conv2.linkTo(&merge);
        conv3.linkTo(&merge);
        merge.linkTo(&conv4);
        conv4.linkTo(&mseloss);
    }
    REAL train_iter(RealTensor& X, RealTensor& Y) {
        conv1.forward(&X);
        split.forward();
        conv2.forward();
        conv3.forward();
        merge.forward(NULL, NULL);
        conv4.forward();
        mseloss.setTarget(Y);
        mseloss.forward();

        REAL loss = mseloss.item();
        REAL lr = REAL(0.005);

        optim.zeroGrad();

        mseloss.backward(&mseloss);
        conv4.backward();
        merge.backward();
        conv3.backward();
        conv2.backward();
        split.backward();
        conv1.backward();

        optim.step(lr);

        return loss;
    }
};

/* printf with '\n' automatically added */
void print(const char * format, ...)
{
    char fmsg[1024] = { 0 };
    va_list args;
    va_start(args, format);
    vsprintf(fmsg, format, args);
    va_end(args);
    printf("%s\n", fmsg);
}

int main()
{
    print("Demo 1: Tensor.");
    print("create a tensor with shape 8x16x16x16.");
    RealTensor A("8,16,16,16"); 
    print("fill it with zero");
    A.fill_(REAL(0.0));
    A.print();

    DemoNet net;

    RealTensor X = T_buildTensor<REAL>(REAL(0.0), REAL(0.05), "2,1,8,8"), Y("2,1,8,8");
    Y = 1 - X;
    Y.print();

    for (int i = 0; i < 1000; i++) {
        REAL loss = net.train_iter(X, Y);
        print("%.4f", float(loss));
    }

    return 0;
}


