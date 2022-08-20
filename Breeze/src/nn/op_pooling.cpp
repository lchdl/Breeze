#include "nn/op_pooling.h"

RealTensor maxPool2d(const RealTensor& tensor, const TensorShape& kernelSize, IntTensor& indices)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (kernelSize.isShapeValid() == false || kernelSize.dim() != 2)
        T_error("NN error: invalid kernel size in maxpool2d.");
    if (tensor.dim() != 4)
        T_error("NN error: invalid tensor dimension (expects %d, got %d) for maxpool2d.",
            4, tensor.dim());
#endif

    int batchSize = tensor.shape()[0];
    int inChannels = tensor.shape()[1];
    int inputHeight = tensor.shape()[2];
    int inputWidth = tensor.shape()[3];

    TensorShape poolShape(4);
    poolShape[0] = batchSize;
    poolShape[1] = inChannels;
    poolShape[2] = (tensor.shape()[2] + (kernelSize[0] - 1)) / kernelSize[0];
    poolShape[3] = (tensor.shape()[3] + (kernelSize[1] - 1)) / kernelSize[1];

#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (poolShape[2] < 1 || poolShape[3] < 1) {
        char a[64], b[64];
        tensor.shape().toStr(a);
        kernelSize.toStr(b);
        T_error("NN error: output shape of maxpool2d is invalid due to "
            "oversized kernel. Input shape is (%s), while kernel size "
            "is (%s).", a, b);
    }
#endif
    TensorShape inShape = tensor.shape();
    TensorShape outShape = poolShape;
    int outputHeight = poolShape[2];
    int outputWidth = poolShape[3];

    RealTensor resultTensor(poolShape);
    IntTensor(poolShape).moveStorage(indices);

#ifdef NN_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int b = 0; b < batchSize; b++) {
        for (int ich = 0; ich < inChannels; ich++) {
            RealTensor
                inVolume(T_buildShape<2>({ inputHeight, inputWidth })),
                outVolume(T_buildShape<2>({ outputHeight, outputWidth }));
            IntTensor outIndices(T_buildShape<2>({ outputHeight, outputWidth }));
            inVolume._dptrWrite(0, inputHeight*inputWidth, tensor,
                b * inChannels * inputHeight * inputWidth + ich * inputHeight * inputWidth);
            outVolume.fill_(-REAL_MAX);
            outIndices.fill_(-1);
            int I = 0;
            for (int h = 0; h < inputHeight; h++) {
                for (int w = 0; w < inputWidth; w++, I++) {
                    REAL val = inVolume[I];
                    int hOut = h / kernelSize[0],
                        wOut = w / kernelSize[1];
                    REAL cMax = outVolume[hOut * outputWidth + wOut];
                    int ind = (h % kernelSize[0]) * kernelSize[1] + (w % kernelSize[1]);
                    if (val > cMax) {
                        cMax = val;
                        outIndices[hOut * outputWidth + wOut] = ind;
                    }
                    outVolume[hOut * outputWidth + wOut] = cMax;
                }
            }
            resultTensor._dptrWrite(b * inChannels * outputHeight * outputWidth + ich * outputHeight * outputWidth,
                outputHeight * outputWidth, outVolume, 0);
            indices._dptrWrite(b * inChannels * outputHeight * outputWidth + ich * outputHeight * outputWidth,
                outputHeight * outputWidth, outIndices, 0);
        }
    }

#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (indices.min() < 0)
        T_error("NN error: invalid index (<0) found in indices.");
#endif

    return resultTensor;
}
RealTensor maxUnpool2d(const RealTensor & tensor, const TensorShape & kernelSize, const IntTensor& indices, const TensorShape * outShape)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (tensor.dim() != 4)
        T_error("NN error: invalid tensor dimension for maxUnpool2d.");
    if (kernelSize.isShapeValid() == false || kernelSize.dim() != 2)
        T_error("NN error: invalid kernel size for maxUnpool2d.");
    if (outShape && (outShape->isShapeValid() == false || outShape->dim() != 4))
        T_error("NN error: invalid output shape for maxUnpool2d.");
    if (indices.shape().isShapeValid() == false || indices.shape() != tensor.shape())
        T_error("NN error: invalid indices for maxUnpool2d.");
#endif
    TensorShape _outShape;
    if (outShape == NULL) {
        _outShape = T_buildShape<4>({ tensor.shape()[0], tensor.shape()[1], tensor.shape()[2] * kernelSize[0],
            tensor.shape()[3] * kernelSize[1] });
    }
    else
        _outShape = *outShape;
    
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_outShape[0] != tensor.shape()[0] || _outShape[1] != tensor.shape()[1])
        T_error("NN error: output shape is invalid for maxUnpool2d.");
#endif

    RealTensor resultTensor(_outShape);
    resultTensor.fill_(REAL(0.0));

    int batchSize = tensor.shape()[0];
    int inChannels = tensor.shape()[1];
    int outputHeight = _outShape[2];
    int outputWidth = _outShape[3];
    int inputHeight = tensor.shape()[2];
    int inputWidth = tensor.shape()[3];

#ifdef NN_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int b = 0; b < batchSize; b++) {
        for (int ich = 0; ich < inChannels; ich++) {
            RealTensor
                outVolume(T_buildShape<2>({ outputHeight, outputWidth })),
                inVolume(T_buildShape<2>({ inputHeight, inputWidth }));
            IntTensor inIndices(T_buildShape<2>({ inputHeight, inputWidth }));
            inVolume._dptrWrite(0, inputHeight * inputWidth, tensor,
                b * inChannels*inputHeight*inputWidth + ich * inputHeight*inputWidth);
            inIndices._dptrWrite(0, inputHeight * inputWidth, indices,
                b * inChannels*inputHeight*inputWidth + ich * inputHeight*inputWidth);
            outVolume.fill_(REAL(0.0));
            int I = 0;
            for (int h = 0; h < outputHeight; h++) {
                for (int w = 0; w < outputWidth; w++, I++) {
                    int hOut = h / kernelSize[0],
                        wOut = w / kernelSize[1];
                    int ind = (h % kernelSize[0]) * kernelSize[1] + (w % kernelSize[1]);
                    if (ind == inIndices[hOut * inputWidth + wOut])
                        outVolume[I] = inVolume[hOut * inputWidth + wOut];
                }
            }
            resultTensor._dptrWrite(b * inChannels*outputHeight*outputWidth + ich * outputHeight*outputWidth,
                outputHeight*outputWidth, outVolume, 0);
        }
    }

    return resultTensor;
}
RealTensor avgPool2d(const RealTensor & tensor, const TensorShape & kernelSize)
{
    /* NOTE: for average pooling if image size is not fully divisible by kernel size  */
    /* the result may be incorrect on the image boundaries, but here for speed i will */
    /* just ignore this incorrectness */

#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (kernelSize.isShapeValid() == false || kernelSize.dim() != 2)
        T_error("NN error: invalid kernel size in avgpool2d.");
    if (tensor.dim() != 4)
        T_error("NN error: invalid tensor dimension (expects %d, got %d) for avgpool2d.",
            4, tensor.dim());
#endif

    int batchSize = tensor.shape()[0];
    int inChannels = tensor.shape()[1];
    int inputHeight = tensor.shape()[2];
    int inputWidth = tensor.shape()[3];

    TensorShape poolShape(4);
    poolShape[0] = batchSize;
    poolShape[1] = inChannels;
    poolShape[2] = (tensor.shape()[2] + (kernelSize[0] - 1)) / kernelSize[0];
    poolShape[3] = (tensor.shape()[3] + (kernelSize[1] - 1)) / kernelSize[1];

#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (poolShape[2] < 1 || poolShape[3] < 1) {
        char a[64], b[64];
        tensor.shape().toStr(a);
        kernelSize.toStr(b);
        T_error("NN error: output shape of avgpool2d is invalid due to "
            "oversized kernel. Input shape is (%s), while kernel size "
            "is (%s).", a, b);
    }
#endif
    TensorShape inShape = tensor.shape();
    TensorShape outShape = poolShape;
    int outputHeight = poolShape[2];
    int outputWidth = poolShape[3];

    RealTensor resultTensor(poolShape);

#ifdef NN_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int b = 0; b < batchSize; b++) {
        for (int ich = 0; ich < inChannels; ich++) {
            RealTensor
                inVolume(T_buildShape<2>({ inputHeight, inputWidth })),
                outVolume(T_buildShape<2>({ outputHeight, outputWidth }));
            int kernelNumel = kernelSize[0] * kernelSize[1];
            inVolume._dptrWrite(0, inputHeight*inputWidth, tensor,
                b * inChannels * inputHeight * inputWidth + ich * inputHeight * inputWidth);
            outVolume.fill_(REAL(0.0));
            int I = 0;
            for (int h = 0; h < inputHeight; h++) {
                for (int w = 0; w < inputWidth; w++, I++) {
                    REAL val = inVolume[I];
                    int hOut = h / kernelSize[0],
                        wOut = w / kernelSize[1];
                    outVolume[hOut * outputWidth + wOut] += val;
                }
            }
            outVolume /= REAL(kernelNumel);

            resultTensor._dptrWrite(b * inChannels * outputHeight * outputWidth + ich * outputHeight * outputWidth,
                outputHeight * outputWidth, outVolume, 0);
        }
    }

    return resultTensor;
}
RealTensor avgUnpool2d(const RealTensor & tensor, const TensorShape & kernelSize, const TensorShape * outShape)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (tensor.dim() != 4)
        T_error("NN error: invalid tensor dimension for avgUnpool2d.");
    if (kernelSize.isShapeValid() == false || kernelSize.dim() != 2)
        T_error("NN error: invalid kernel size for avgUnpool2d.");
    if (outShape && (outShape->isShapeValid() == false || outShape->dim() != 4))
        T_error("NN error: invalid output shape for avgUnpool2d.");
#endif
    TensorShape _outShape;
    if (outShape == NULL) {
        _outShape = T_buildShape<4>({ tensor.shape()[0], tensor.shape()[1], tensor.shape()[2] * kernelSize[0],
            tensor.shape()[3] * kernelSize[1] });
    }
    else
        _outShape = *outShape;

#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_outShape[0] != tensor.shape()[0] || _outShape[1] != tensor.shape()[1])
        T_error("NN error: output shape is invalid for avgUnpool2d.");
#endif

    RealTensor resultTensor(_outShape);
    resultTensor.fill_(REAL(0.0));

    int batchSize = tensor.shape()[0];
    int inChannels = tensor.shape()[1];
    int outputHeight = _outShape[2];
    int outputWidth = _outShape[3];
    int inputHeight = tensor.shape()[2];
    int inputWidth = tensor.shape()[3];

#ifdef NN_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int b = 0; b < batchSize; b++) {
        for (int ich = 0; ich < inChannels; ich++) {
            RealTensor
                outVolume(T_buildShape<2>({ outputHeight, outputWidth })),
                inVolume(T_buildShape<2>({ inputHeight, inputWidth }));
            inVolume._dptrWrite(0, inputHeight * inputWidth, tensor,
                b * inChannels*inputHeight*inputWidth + ich * inputHeight*inputWidth);
            outVolume.fill_(REAL(0.0));
            int I = 0;
            int kernelNumel = kernelSize[0] * kernelSize[1];
            for (int h = 0; h < outputHeight; h++) {
                for (int w = 0; w < outputWidth; w++, I++) {
                    int hOut = h / kernelSize[0],
                        wOut = w / kernelSize[1];
                    outVolume[I] = inVolume[hOut * inputWidth + wOut];
                }
            }
            outVolume /= REAL(kernelNumel);
            resultTensor._dptrWrite(b * inChannels*outputHeight*outputWidth + ich * outputHeight*outputWidth,
                outputHeight*outputWidth, outVolume, 0);
        }
    }

    return resultTensor;
}

RealTensor maxPool3d(const RealTensor & tensor, const TensorShape & kernelSize, IntTensor & indices)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (kernelSize.isShapeValid() == false || kernelSize.dim() != 3)
        T_error("NN error: invalid kernel size in maxpool3d.");
    if (tensor.dim() != 5)
        T_error("NN error: invalid tensor dimension (expects %d, got %d) for maxpool3d.",
            5, tensor.dim());
#endif

    int batchSize = tensor.shape()[0];
    int inChannels = tensor.shape()[1];
    int inputX = tensor.shape()[2];
    int inputY = tensor.shape()[3];
    int inputZ = tensor.shape()[4];

    TensorShape poolShape(5);
    poolShape[0] = batchSize;
    poolShape[1] = inChannels;
    poolShape[2] = (tensor.shape()[2] + (kernelSize[0] - 1)) / kernelSize[0];
    poolShape[3] = (tensor.shape()[3] + (kernelSize[1] - 1)) / kernelSize[1];
    poolShape[4] = (tensor.shape()[4] + (kernelSize[2] - 1)) / kernelSize[2];

#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (poolShape[2] < 1 || poolShape[3] < 1 || poolShape[4] < 1) {
        char a[64], b[64];
        tensor.shape().toStr(a);
        kernelSize.toStr(b);
        T_error("NN error: output shape of maxpool3d is invalid due to "
            "oversized kernel. Input shape is (%s), while kernel size "
            "is (%s).", a, b);
    }
#endif
    TensorShape inShape = tensor.shape();
    TensorShape outShape = poolShape;
    int outputX = poolShape[2];
    int outputY = poolShape[3];
    int outputZ = poolShape[4];

    RealTensor resultTensor(poolShape);
    IntTensor(poolShape).moveStorage(indices);

#ifdef NN_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int b = 0; b < batchSize; b++) {
        for (int ich = 0; ich < inChannels; ich++) {
            RealTensor
                inVolume(T_buildShape<3>({ inputX, inputY, inputZ })),
                outVolume(T_buildShape<3>({ outputX, outputY, outputZ }));
            IntTensor outIndices(T_buildShape<3>({ outputX, outputY, outputZ }));
            inVolume._dptrWrite(0, inputX*inputY*inputZ, tensor,
                b * inChannels * inputX*inputY*inputZ + ich * inputX*inputY*inputZ);
            outVolume.fill_(-REAL_MAX);
            outIndices.fill_(-1);
            int I = 0;
            for (int x = 0; x < inputX; x++) {
                for (int y = 0; y < inputY; y++) {
                    for (int z = 0; z < inputZ; z++, I++) {
                        REAL val = inVolume[I];
                        int xOut = x / kernelSize[0],
                            yOut = y / kernelSize[1],
                            zOut = z / kernelSize[2];
                        int indOut = xOut * outputY*outputZ + yOut * outputZ + zOut;
                        REAL cMax = outVolume[indOut];
                        int indLocal = (x % kernelSize[0]) * kernelSize[1] * kernelSize[2] +
                            (y % kernelSize[1]) * kernelSize[2] + (z % kernelSize[2]);
                        if (val > cMax) {
                            cMax = val;
                            outIndices[indOut] = indLocal;
                        }
                        outVolume[indOut] = cMax;
                    }
                }
            }
            resultTensor._dptrWrite(b * inChannels*outputX*outputY*outputZ + 
                ich * outputX*outputY*outputZ, outputX*outputY*outputZ, outVolume, 0);
            indices._dptrWrite(b * inChannels*outputX*outputY*outputZ + 
                ich * outputX*outputY*outputZ, outputX*outputY*outputZ, outIndices, 0);
        }
    }

#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (indices.min() < 0)
        T_error("NN error: invalid index (<0) found in indices.");
#endif

    return resultTensor;
}
RealTensor maxUnpool3d(const RealTensor & tensor, const TensorShape & kernelSize, const IntTensor& indices, const TensorShape * outShape)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (tensor.dim() != 5)
        T_error("NN error: invalid tensor dimension for maxUnpool3d.");
    if (kernelSize.isShapeValid() == false || kernelSize.dim() != 3)
        T_error("NN error: invalid kernel size for maxUnpool3d.");
    if (outShape && (outShape->isShapeValid() == false || outShape->dim() != 5))
        T_error("NN error: invalid output shape for maxUnpool3d.");
    if (indices.shape().isShapeValid() == false || indices.shape() != tensor.shape())
        T_error("NN error: invalid indices for maxUnpool3d.");
#endif
    TensorShape _outShape;
    if (outShape == NULL) {
        _outShape = T_buildShape<5>({
            tensor.shape()[0], 
            tensor.shape()[1], 
            tensor.shape()[2] * kernelSize[0],
            tensor.shape()[3] * kernelSize[1], 
            tensor.shape()[4] * kernelSize[2] });
    }
    else
        _outShape = *outShape;

#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_outShape[0] != tensor.shape()[0] || _outShape[1] != tensor.shape()[1])
        T_error("NN error: output shape is invalid for maxUnpool3d.");
#endif

    RealTensor resultTensor(_outShape);
    resultTensor.fill_(REAL(0.0));

    int batchSize = tensor.shape()[0];
    int inChannels = tensor.shape()[1];
    int outputX = _outShape[2];
    int outputY = _outShape[3];
    int outputZ = _outShape[4];
    int inputX = tensor.shape()[2];
    int inputY = tensor.shape()[3];
    int inputZ = tensor.shape()[4];

#ifdef NN_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int b = 0; b < batchSize; b++) {
        for (int ich = 0; ich < inChannels; ich++) {
            RealTensor
                outVolume(T_buildShape<3>({ outputX, outputY, outputZ })),
                inVolume(T_buildShape<3>({ inputX, inputY, inputZ }));
            IntTensor inIndices(T_buildShape<3>({ inputX, inputY, inputZ }));
            inVolume._dptrWrite(0, inputX*inputY*inputZ, tensor,
                b * inChannels*inputX*inputY*inputZ + ich * inputX*inputY*inputZ);
            inIndices._dptrWrite(0, inputX*inputY*inputZ, indices,
                b * inChannels*inputX*inputY*inputZ + ich * inputX*inputY*inputZ);
            outVolume.fill_(REAL(0.0));
            int I = 0;
            for (int x = 0; x < outputX; x++) {
                for (int y = 0; y < outputY; y++) {
                    for (int z = 0; z < outputZ; z++, I++) {
                        int xOut = x / kernelSize[0],
                            yOut = y / kernelSize[1],
                            zOut = z / kernelSize[2];
                        int indLocal = (x % kernelSize[0]) * kernelSize[1] * kernelSize[2] + 
                            (y % kernelSize[1]) * kernelSize[2] + (z % kernelSize[2]);
                        int indOut = xOut * inputY*inputZ + yOut * inputZ + zOut;
                        if (indLocal == inIndices[indOut])
                            outVolume[I] = inVolume[indOut];
                    }
                }
            }
            resultTensor._dptrWrite(b * inChannels*outputX*outputY*outputZ + 
                ich * outputX*outputY*outputZ, outputX*outputY*outputZ, outVolume, 0);
        }
    }

    return resultTensor;
}
RealTensor avgPool3d(const RealTensor & tensor, const TensorShape & kernelSize)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (kernelSize.isShapeValid() == false || kernelSize.dim() != 3)
        T_error("NN error: invalid kernel size in avgpool3d.");
    if (tensor.dim() != 5)
        T_error("NN error: invalid tensor dimension (expects %d, got %d) for avgpool3d.",
            5, tensor.dim());
#endif

    int batchSize = tensor.shape()[0];
    int inChannels = tensor.shape()[1];
    int inputX = tensor.shape()[2];
    int inputY = tensor.shape()[3];
    int inputZ = tensor.shape()[4];

    TensorShape poolShape(5);
    poolShape[0] = batchSize;
    poolShape[1] = inChannels;
    poolShape[2] = (tensor.shape()[2] + (kernelSize[0] - 1)) / kernelSize[0];
    poolShape[3] = (tensor.shape()[3] + (kernelSize[1] - 1)) / kernelSize[1];
    poolShape[4] = (tensor.shape()[4] + (kernelSize[2] - 1)) / kernelSize[2];

#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (poolShape[2] < 1 || poolShape[3] < 1 || poolShape[4] < 1) {
        char a[64], b[64];
        tensor.shape().toStr(a);
        kernelSize.toStr(b);
        T_error("NN error: output shape of avgpool3d is invalid due to "
            "oversized kernel. Input shape is (%s), while kernel size "
            "is (%s).", a, b);
    }
#endif
    TensorShape inShape = tensor.shape();
    TensorShape outShape = poolShape;
    int outputX = poolShape[2];
    int outputY = poolShape[3];
    int outputZ = poolShape[4];
    int kernelNumel = kernelSize[0] * kernelSize[1] * kernelSize[2];

    RealTensor resultTensor(poolShape);

#ifdef NN_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int b = 0; b < batchSize; b++) {
        for (int ich = 0; ich < inChannels; ich++) {
            RealTensor
                inVolume(T_buildShape<3>({ inputX, inputY, inputZ })),
                outVolume(T_buildShape<3>({ outputX, outputY, outputZ }));
            inVolume._dptrWrite(0, inputX*inputY*inputZ, tensor,
                b * inChannels * inputX*inputY*inputZ + ich * inputX*inputY*inputZ);
            outVolume.fill_(REAL(0.0));
            int I = 0;
            for (int x = 0; x < inputX; x++) {
                for (int y = 0; y < inputY; y++) {
                    for (int z = 0; z < inputZ; z++, I++) {
                        REAL val = inVolume[I];
                        int xOut = x / kernelSize[0],
                            yOut = y / kernelSize[1],
                            zOut = z / kernelSize[2];
                        int indOut = xOut * outputY*outputZ + yOut * outputZ + zOut;
                        outVolume[indOut] += val;
                    }
                }
            }
            outVolume /= REAL(kernelNumel);
            resultTensor._dptrWrite(b * inChannels*outputX*outputY*outputZ +
                ich * outputX*outputY*outputZ, outputX*outputY*outputZ, outVolume, 0);
        }
    }

    return resultTensor;
}
RealTensor avgUnpool3d(const RealTensor & tensor, const TensorShape & kernelSize, const TensorShape * outShape)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (tensor.dim() != 5)
        T_error("NN error: invalid tensor dimension for avgUnpool3d.");
    if (kernelSize.isShapeValid() == false || kernelSize.dim() != 3)
        T_error("NN error: invalid kernel size for avgUnpool3d.");
    if (outShape && (outShape->isShapeValid() == false || outShape->dim() != 5))
        T_error("NN error: invalid output shape for avgUnpool3d.");
#endif
    TensorShape _outShape;
    if (outShape == NULL) {
        _outShape = T_buildShape<5>({
            tensor.shape()[0],
            tensor.shape()[1],
            tensor.shape()[2] * kernelSize[0],
            tensor.shape()[3] * kernelSize[1],
            tensor.shape()[4] * kernelSize[2] });
    }
    else
        _outShape = *outShape;

#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_outShape[0] != tensor.shape()[0] || _outShape[1] != tensor.shape()[1])
        T_error("NN error: output shape is invalid for avgUnpool3d.");
#endif

    RealTensor resultTensor(_outShape);
    resultTensor.fill_(REAL(0.0));

    int batchSize = tensor.shape()[0];
    int inChannels = tensor.shape()[1];
    int outputX = _outShape[2];
    int outputY = _outShape[3];
    int outputZ = _outShape[4];
    int inputX = tensor.shape()[2];
    int inputY = tensor.shape()[3];
    int inputZ = tensor.shape()[4];
    int kernelNumel = kernelSize[0] * kernelSize[1] * kernelSize[2];

#ifdef NN_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int b = 0; b < batchSize; b++) {
        for (int ich = 0; ich < inChannels; ich++) {
            RealTensor
                outVolume(T_buildShape<3>({ outputX, outputY, outputZ })),
                inVolume(T_buildShape<3>({ inputX, inputY, inputZ }));
            inVolume._dptrWrite(0, inputX*inputY*inputZ, tensor,
                b * inChannels*inputX*inputY*inputZ + ich * inputX*inputY*inputZ);
            outVolume.fill_(REAL(0.0));
            int I = 0;
            for (int x = 0; x < outputX; x++) {
                for (int y = 0; y < outputY; y++) {
                    for (int z = 0; z < outputZ; z++, I++) {
                        int xOut = x / kernelSize[0],
                            yOut = y / kernelSize[1],
                            zOut = z / kernelSize[2];
                        int indOut = xOut * inputY*inputZ + yOut * inputZ + zOut;
                        outVolume[I] = inVolume[indOut];
                    }
                }
            }
            outVolume /= REAL(kernelNumel);
            resultTensor._dptrWrite(b * inChannels*outputX*outputY*outputZ +
                ich * outputX*outputY*outputZ, outputX*outputY*outputZ, outVolume, 0);
        }
    }

    return resultTensor;
}

OpMaxPool2d::OpMaxPool2d() { nextOp = prevOp = NULL; }
void OpMaxPool2d::createOp(const char * _kernel)
{
    this->kernelSize = T_buildArray<int>(_kernel);
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (this->kernelSize.isShapeValid() == false)
        T_error("NN error: kernel size for maxpool2d is invalid.");
    if (this->kernelSize.dim() != 2)
        T_error("NN error: expected 2d kernel for maxpool2d but "
            "got kernel with %d dimension(s).", this->kernelSize.dim());
#endif
}
const char * OpMaxPool2d::getOpClassName() const { return "OpMaxPool2d"; }
void OpMaxPool2d::forward(RealTensor * _input)
{
    if (_input != NULL) {
        /* input comes from external source */
        X = *_input;
    }
    else { /* input comes from internal data flow */
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (prevOp == NULL) {
            T_error("NN error: cannot feed forward because no data has given.\n");
            return;
        }
#endif
        /* request input data from previous layer */
        prevOp->_passInput(this, &(this->X));
    }
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (X.dim() != 4) {
        char s[64];
        X.shape().toStr(s);
        T_error("NN error: %s expects a 4D tensor input with shape (B, C, H, W), "
            "got tensor with shape (%s).", getOpClassName(), s);
    }
#endif
    maxPool2d(X, this->kernelSize, this->indices).moveStorage(Y);
    X.purge();
}
void OpMaxPool2d::backward(GraphOp * _op)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_op == NULL && nextOp == NULL) {
        T_error("NN error: backward op set to NULL, which means the gradient flows from "
            "next op, but next op is NULL.");
        return;
    }
#endif
    /* * * * * * * * * * */
    /* step 2: get dL/dY */
    /* * * * * * * * * * */
    if (_op == this) {
        dLdY = RealTensor(outShape);
        /* gradients will be back propagated from this  */
        /* layer so setting output gradients to all one */
        dLdY.fill_(RealTensor::type(1.0));
    }
    else {
        /* passing gradients of the next layer to current layer */
        nextOp->_passGrad(this, &(this->dLdY));
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (outShape.isEqual(dLdY.shape()) == false) {
            char es[64], as[64];
            outShape.toStr(es);
            dLdY.shape().toStr(as);
            T_error("NN error: out shape changed after backward(). "
                "Expected shape: (%s), actual shape: (%s).", es, as);
        }
#endif
    }


    maxUnpool2d(dLdY, this->kernelSize, this->indices, &(this->inShape)).moveStorage(dLdX);
    dLdY.purge();

}
void OpMaxPool2d::linkTo(GraphOp * _op)
{
    this->nextOp = _op;
    if (_op != NULL)
        _op->_linkFrom(this);
}
void OpMaxPool2d::params(Array<RealTensor*>* _param, Array<RealTensor*>* _grad){}
SerializedTensors<REAL> OpMaxPool2d::state() { return SerializedTensors<REAL>(); }
bool OpMaxPool2d::loadState(SerializedTensors<REAL>& ser) { return true; }
RealTensor OpMaxPool2d::input()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (X.isNull())
        T_error("NN error: cannot retrieve input tensor as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return X;
}
RealTensor OpMaxPool2d::output()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (Y.isNull())
        T_error("NN error: cannot retrieve output tensor as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return Y;
}
RealTensor OpMaxPool2d::inputGrad()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdX.isNull())
        T_error("NN error: cannot retrieve input gradient as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return dLdX;
}
RealTensor OpMaxPool2d::outputGrad()
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdY.isNull())
        T_error("NN error: cannot retrieve output gradient as the "
            "tensor storage is moved elsewhere or undefined.");
#endif
    return dLdY;
}
void OpMaxPool2d::_linkFrom(GraphOp * _op) { this->prevOp = _op; }
void OpMaxPool2d::_passGrad(GraphOp * _prevOp, RealTensor * _grad)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_prevOp != prevOp)
        T_error("NN error: specified layer is not linked to the input slot of this layer.\n");
#endif
    /* passing gradients from current layer to previous layer */
    this->dLdX.moveStorage(*_grad);
}
void OpMaxPool2d::_passInput(GraphOp * _nextOp, RealTensor * _input)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_nextOp != nextOp)
        T_error("NN error: specified layer is not linked to the output slot of this layer.\n");
#endif
    /* set next layer input to be the output of this layer */
    this->Y.moveStorage(*_input);
}

void OpAvgPool2d::createOp(const char * _kernel)
{
    this->kernelSize = T_buildArray<int>(_kernel);
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (this->kernelSize.isShapeValid() == false)
        T_error("NN error: kernel size for avgpool2d is invalid.");
    if (this->kernelSize.dim() != 2)
        T_error("NN error: expected 2d kernel for avgpool2d but "
            "got kernel with %d dimension(s).", this->kernelSize.dim());
#endif
}
void OpAvgPool2d::forward(RealTensor * _input)
{
    if (_input != NULL) {
        /* input comes from external source */
        X = *_input;
    }
    else { /* input comes from internal data flow */
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (prevOp == NULL) {
            T_error("NN error: cannot feed forward because no data has given.\n");
            return;
        }
#endif
        /* request input data from previous layer */
        prevOp->_passInput(this, &(this->X));
    }
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (X.dim() != 4) {
        char s[64];
        X.shape().toStr(s);
        T_error("NN error: %s expects a 4D tensor input with shape (B, C, H, W), "
            "got tensor with shape (%s).", getOpClassName(), s);
    }
#endif
    avgPool2d(X, this->kernelSize).moveStorage(Y);
    X.purge();
}
const char * OpAvgPool2d::getOpClassName() const { return "OpAvgPool2d"; }
void OpAvgPool2d::backward(GraphOp * _op)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_op == NULL && nextOp == NULL) {
        T_error("NN error: backward op set to NULL, which means the gradient flows from "
            "next op, but next op is NULL.");
        return;
    }
#endif
    /* * * * * * * * * * */
    /* step 2: get dL/dY */
    /* * * * * * * * * * */
    if (_op == this) {
        dLdY = RealTensor(outShape);
        /* gradients will be back propagated from this  */
        /* layer so setting output gradients to all one */
        dLdY.fill_(RealTensor::type(1.0));
    }
    else {
        /* passing gradients of the next layer to current layer */
        nextOp->_passGrad(this, &(this->dLdY));
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (outShape.isEqual(dLdY.shape()) == false) {
            char es[64], as[64];
            outShape.toStr(es);
            dLdY.shape().toStr(as);
            T_error("NN error: out shape changed after backward(). "
                "Expected shape: (%s), actual shape: (%s).", es, as);
        }
#endif
    }

    avgUnpool2d(dLdY, this->kernelSize, &(this->inShape)).moveStorage(dLdX);
    dLdY.purge();

}

void OpMaxPool3d::createOp(const char * _kernel)
{
    this->kernelSize = T_buildArray<int>(_kernel);
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (this->kernelSize.isShapeValid() == false)
        T_error("NN error: kernel size for maxpool3d is invalid.");
    if (this->kernelSize.dim() != 3)
        T_error("NN error: expected 3d kernel for maxpool3d but "
            "got kernel with %d dimension(s).", this->kernelSize.dim());
#endif
}
const char * OpMaxPool3d::getOpClassName() const { return "OpMaxPool3d"; }
void OpMaxPool3d::forward(RealTensor * _input)
{
    if (_input != NULL) {
        /* input comes from external source */
        X = *_input;
    }
    else { /* input comes from internal data flow */
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (prevOp == NULL) {
            T_error("NN error: cannot feed forward because no data has given.\n");
            return;
        }
#endif
        /* request input data from previous layer */
        prevOp->_passInput(this, &(this->X));
    }
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (X.dim() != 5) {
        char s[64];
        X.shape().toStr(s);
        T_error("NN error: %s expects a 5D tensor input with shape (B, C, X, Y, Z), "
            "got tensor with shape (%s).", getOpClassName(), s);
    }
#endif
    maxPool3d(X, this->kernelSize, this->indices).moveStorage(Y);
    X.purge();
}
void OpMaxPool3d::backward(GraphOp * _op)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_op == NULL && nextOp == NULL) {
        T_error("NN error: backward op set to NULL, which means the gradient flows from "
            "next op, but next op is NULL.");
        return;
    }
#endif
    /* * * * * * * * * * */
    /* step 2: get dL/dY */
    /* * * * * * * * * * */
    if (_op == this) {
        dLdY = RealTensor(outShape);
        /* gradients will be back propagated from this  */
        /* layer so setting output gradients to all one */
        dLdY.fill_(RealTensor::type(1.0));
    }
    else {
        /* passing gradients of the next layer to current layer */
        nextOp->_passGrad(this, &(this->dLdY));
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (outShape.isEqual(dLdY.shape()) == false) {
            char es[64], as[64];
            outShape.toStr(es);
            dLdY.shape().toStr(as);
            T_error("NN error: out shape changed after backward(). "
                "Expected shape: (%s), actual shape: (%s).", es, as);
        }
#endif
    }

    maxUnpool3d(dLdY, this->kernelSize, this->indices, &(this->inShape)).moveStorage(dLdX);
    dLdY.purge();
}

void OpAvgPool3d::createOp(const char * _kernel)
{
    this->kernelSize = T_buildArray<int>(_kernel);
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (this->kernelSize.isShapeValid() == false)
        T_error("NN error: kernel size for avgpool3d is invalid.");
    if (this->kernelSize.dim() != 3)
        T_error("NN error: expected 3d kernel for avgpool3d but "
            "got kernel with %d dimension(s).", this->kernelSize.dim());
#endif
}
const char * OpAvgPool3d::getOpClassName() const { return "OpAvgPool3d"; }
void OpAvgPool3d::forward(RealTensor * _input)
{
    if (_input != NULL) {
        /* input comes from external source */
        X = *_input;
    }
    else { /* input comes from internal data flow */
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (prevOp == NULL) {
            T_error("NN error: cannot feed forward because no data has given.\n");
            return;
        }
#endif
        /* request input data from previous layer */
        prevOp->_passInput(this, &(this->X));
    }
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (X.dim() != 5) {
        char s[64];
        X.shape().toStr(s);
        T_error("NN error: %s expects a 5D tensor input with shape (B, C, X, Y, Z), "
            "got tensor with shape (%s).", getOpClassName(), s);
    }
#endif
    avgPool3d(X, this->kernelSize).moveStorage(Y);
    X.purge();
}
void OpAvgPool3d::backward(GraphOp * _op)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_op == NULL && nextOp == NULL) {
        T_error("NN error: backward op set to NULL, which means the gradient flows from "
            "next op, but next op is NULL.");
        return;
    }
#endif
    /* * * * * * * * * * */
    /* step 2: get dL/dY */
    /* * * * * * * * * * */
    if (_op == this) {
        dLdY = RealTensor(outShape);
        /* gradients will be back propagated from this  */
        /* layer so setting output gradients to all one */
        dLdY.fill_(RealTensor::type(1.0));
    }
    else {
        /* passing gradients of the next layer to current layer */
        nextOp->_passGrad(this, &(this->dLdY));
#ifdef NN_ENABLE_RUNTIME_CHECKING
        if (outShape.isEqual(dLdY.shape()) == false) {
            char es[64], as[64];
            outShape.toStr(es);
            dLdY.shape().toStr(as);
            T_error("NN error: out shape changed after backward(). "
                "Expected shape: (%s), actual shape: (%s).", es, as);
        }
#endif
    }
    avgUnpool3d(dLdY, this->kernelSize, &(this->inShape)).moveStorage(dLdX);
    dLdY.purge();
}

