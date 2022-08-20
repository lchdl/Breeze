/* define (transposed) convolutions and their backward functions */
#pragma once
#include "nn/nn_global.h"
#include "nn/tensor_types.h"

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                           2D convolution                              */
/*    (C_in, H, W) * (C_out, C_in, K_h, K_w) = (C_out, H_out, W_out)     */
/*  ** ARBITRARY STRIDE                                                  */
/*  ** SUITABLE FOR SMALL KERNELS                                        */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
template <typename TensorType>
TensorType T_conv2d_s(
    const TensorType& tensor, 
    const TensorType& kernel,
    const Array<int>& stride = Array<int>(), 
    const Array<int>& padding = Array<int>()) 
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (tensor.dim() != 3 || kernel.dim() != 4 ||
        tensor.shape()[0] != kernel.shape()[1]) {
        char a[64], b[64];
        tensor.shape().toStr(a);
        kernel.shape().toStr(b);
        T_error("Invalid tensor dimension or shape setting. Input tensor "
            "should be 3D with shape (C_in, H, W) and kernel should be 4D "
            "with shape (C_out, C_in, K_h, K_w), got input tensor shape "
            "(%s) and kernel shape (%s).", a, b);
    }
    if (stride.size() != 0 && stride.size() != 2) {
        T_error("Invalid stride setting in conv2d. Stride can be an empty "
            "array to indicate default stride (1,1) or can be an array with "
            "length of two to indicate the stride amount of two dimensions "
            "(height & width).");
    }
    if (padding.size() != 0 && padding.size() != 2) {
        T_error("Invalid padding setting in conv2d. Padding can be an empty "
            "array to indicate default padding (0,0) or can be an array with "
            "length of two to indicate the padding amount of two dimensions "
            "(height & width).");
    }
#endif
    
    /* parse stride and padding */
    Array<int> _stride = T_buildArray<2, int>({ 1, 1 }),
        _padding = T_buildArray<2, int>({ 0, 0 });
    if (stride.size() > 0) { _stride[0] = stride[0]; _stride[1] = stride[1]; }
    if (padding.size() > 0) { _padding[0] = padding[0]; _padding[1] = padding[1]; }
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_stride[0] < 1 || _stride[1] < 1)
        T_error("error: invalid stride setting in conv2d.");
    if (_padding[0] < 0 || _padding[1] < 0)
        T_error("error: invalid padding setting in conv2d.");
#endif

    /* compute sizes needed for convolution */
    int inputHeight = tensor.shape()[1], inputWidth = tensor.shape()[2];
    int kernelHeight = kernel.shape()[2], kernelWidth = kernel.shape()[3];
    int kernelNumel = kernelHeight * kernelWidth;
    int inChannels = kernel.shape()[1], outChannels = kernel.shape()[0];
    int outputHeight = (inputHeight + 2 * _padding[0] - kernelHeight) / _stride[0] + 1;
    int outputWidth = (inputWidth + 2 * _padding[1] - kernelWidth) / _stride[1] + 1;

#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (outputHeight < 1 || outputWidth < 1) {
        char a[64], b[64];
        tensor.shape().toStr(a);
        kernel.shape().toStr(b);
        T_error("Cannot perform 2D convolution due to invalid result image "
            "size (<1). Maybe the kernel size is too large. Tensor size "
            "is (%s), while kernel size is (%s).", a, b);
    }
#endif

    /* pad tensor if needed */
    const TensorType* padded = NULL;
    bool needPadding = false;
    if (_padding[0] > 0 || _padding[1] > 0)
        needPadding = true;
    if (needPadding) {
        TensorType* _pTensor = new TensorType();
        (*_pTensor) = tensor.pad(T_buildArray<6, int>(
            { 0, 0, _padding[0], _padding[0], _padding[1], _padding[1] }));
        padded = _pTensor;
    }
    else padded = &tensor;

    int paddedHeight = padded->shape()[1], paddedWidth = padded->shape()[2];
    TensorShape shpA = T_buildShape<2>({ outputWidth * outputHeight, kernelNumel * inChannels });
    TensorType tensorA(shpA);
    /* fill data */
    int addr = 0;
    for (int ih = 0; ih < outputHeight; ih++) {
        for (int iw = 0; iw < outputWidth; iw++) {
            for (int ic = 0; ic < inChannels; ic++) {
                /* fill each convolution unit */
                for (int ikh = 0; ikh < kernelHeight; ikh++) {
                    for (int ikw = 0; ikw < kernelWidth; ikw++) {
                        int h = ih * _stride[0] + ikh, w = iw * _stride[1] + ikw;
                        /* write [ic, h, w] */
                        tensorA[addr++] = (*padded)[ic * paddedHeight * paddedWidth + h * paddedWidth + w];
                    }
                }
            }
        }
    }
    TensorType tensorB = kernel.reshape(T_buildShape<2>({ outChannels, kernelNumel*inChannels }));
    TensorType tensorC = T_mm_t(tensorA, tensorB).transpose().reshape(T_buildShape<3>({ outChannels, outputHeight, outputWidth }));
    
    if (needPadding)
        delete padded;

    return tensorC;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                 2D batched convolution (fast version),                */
/* (B, C_in, H, W) * (C_out, C_in, K_h, K_w) = (B, C_out, H_out, W_out)  */
/*  ** ARBITRARY STRIDE                                                  */
/*  ** SUITABLE FOR SMALL KERNELS                                        */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
template <typename TensorType>
TensorType T_conv2d(
    const TensorType& tensor, 
    const TensorType& kernel,
    const Array<int>& stride = Array<int>(),
    const Array<int>& padding = Array<int>()) {
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (tensor.dim() != 4 || kernel.dim() != 4 ||
        tensor.shape()[1] != kernel.shape()[1]) {
        char a[64], b[64];
        tensor.shape().toStr(a);
        kernel.shape().toStr(b);
        T_error("Invalid tensor dimension or shape setting. Input tensor "
            "should be 4D with shape (B, C_in, H, W) and kernel should be "
            "4D with shape (C_out, C_in, K_h, K_w), got input tensor shape "
            "(%s) and kernel shape (%s).", a, b);
    }
    if (stride.size() != 0 && stride.size() != 2) {
        T_error("Invalid stride setting in conv2d. Stride can be an empty "
            "array to indicate default stride (1,1) or can be an array with "
            "length of two to indicate the stride amount of two dimensions "
            "(height & width).");
    }
    if (padding.size() != 0 && padding.size() != 2) {
        T_error("Invalid padding setting in conv2d. Padding can be an empty "
            "array to indicate default padding (0,0) or can be an array with "
            "length of two to indicate the padding amount of two dimensions "
            "(height & width).");
    }
#endif

    /* parse stride and padding */
    Array<int> _stride = T_buildArray<2, int>({ 1, 1 }),
        _padding = T_buildArray<2, int>({ 0, 0 });
    if (stride.size() > 0) { _stride[0] = stride[0]; _stride[1] = stride[1]; }
    if (padding.size() > 0) { _padding[0] = padding[0]; _padding[1] = padding[1]; }
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_stride[0] < 1 || _stride[1] < 1)
        T_error("error: invalid stride setting in conv2d.");
    if (_padding[0] < 0 || _padding[1] < 0)
        T_error("error: invalid padding setting in conv2d.");
#endif

    /* compute sizes needed for convolution */
    int inputHeight = tensor.shape()[2], inputWidth = tensor.shape()[3];
    int kernelHeight = kernel.shape()[2], kernelWidth = kernel.shape()[3];
    int kernelNumel = kernelHeight * kernelWidth;
    int inChannels = kernel.shape()[1], outChannels = kernel.shape()[0];
    int outputHeight = (inputHeight + 2 * _padding[0] - kernelHeight) / _stride[0] + 1;
    int outputWidth = (inputWidth + 2 * _padding[1] - kernelWidth) / _stride[1] + 1;
    int batchSize = tensor.shape()[0];

    TensorType resultTensor(T_buildShape<4>({ batchSize, outChannels, outputHeight, outputWidth }));
    TensorShape mcImageShape = T_buildShape<3>({ inChannels, inputHeight, inputWidth });
    int mcImageNumel = mcImageShape.numel();

#ifdef NN_ENABLE_OPENMP
#pragma omp parallel for schedule (static)
#endif
    for (int i = 0; i < batchSize; i++) {
        TensorType mcImageTensor(mcImageShape);
        mcImageTensor._dptrWrite(0, mcImageNumel, tensor, i*mcImageNumel);
        TensorType mcConvResult;
        T_conv2d_s(mcImageTensor, kernel, stride, padding).moveStorage(mcConvResult);
        int mcResultNumel = mcConvResult.numel();
        resultTensor._dptrWrite(i*mcResultNumel, mcResultNumel, mcConvResult, 0);
    }
    return resultTensor;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                       2D transposed convolution                       */
/*    (C_in, H, W) * (C_out, C_in, K_h, K_w) = (C_out, H_out, W_out)     */
/*  ** ARBITRARY STRIDE                                                  */
/*  ** SUITABLE FOR SMALL KERNELS                                        */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
template <typename TensorType>
TensorType T_convTranspose2d_s(
    const TensorType& tensor, 
    const TensorType& kernel,
    const Array<int>& stride = Array<int>())
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (tensor.dim() != 3 || kernel.dim() != 4 ||
        tensor.shape()[0] != kernel.shape()[1]) {
        char a[64], b[64];
        tensor.shape().toStr(a);
        kernel.shape().toStr(b);
        T_error("Invalid tensor dimension or shape setting. Input tensor "
            "should be 3D with shape (C_in, H, W) and kernel should be 4D "
            "with shape (C_out, C_in, K_h, K_w), got input tensor shape "
            "(%s) and kernel shape (%s).", a, b);
    }
    if (stride.size() != 0 && stride.size() != 2) {
        T_error("Invalid stride setting in convTranspose2d. Stride can be an empty "
            "array to indicate default stride (1,1) or can be an array with "
            "length of two to indicate the stride amount of two dimensions "
            "(height & width).");
    }
#endif
    Array<int> _stride = T_buildArray<2, int>({ 1,1 });
    if (stride.size() > 0) {
        _stride[0] = stride[0];
        _stride[1] = stride[1];
    }
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_stride[0] < 1 || _stride[1] < 1)
        T_error("error: invalid stride setting in convTranspose2d.");
#endif

    int inChannels = kernel.shape()[1];
    int outChannels = kernel.shape()[0];
    int kernelHeight = kernel.shape()[2];
    int kernelWidth = kernel.shape()[3];
    int kernelNumel = kernelHeight * kernelWidth;
    /* constructing X */
    bool unitStride = (_stride[0] == 1 && _stride[1] == 1) ? true : false;
    TensorType X;
    if (unitStride) X = tensor;
    else X = tensor.bonpad(_stride, true);
    X = X.pad(T_buildArray<4, int>({ kernelHeight - 1, 
        kernelHeight - 1, kernelWidth - 1, kernelWidth - 1 }));
    /* constructing K (reverse kernel in every channel) */
    TensorType K = kernel;
    typename TensorType::type* Kptr = &(K[0]);
    for (int i = 0; i < outChannels*inChannels; i++, Kptr += kernelNumel)
        T_reverse(Kptr, kernelNumel);
    /* transposed convolution can be converted to ordinary convolution */
    return T_conv2d_s(X, K);
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                    2D batched transposed convolution                  */
/* (B, C_in, H, W) * (C_out, C_in, K_h, K_w) = (B, C_out, H_out, W_out)  */
/*  ** ARBITRARY STRIDE                                                  */
/*  ** SUITABLE FOR SMALL KERNELS                                        */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
template <typename TensorType>
TensorType T_convTranspose2d(
    const TensorType& tensor, 
    const TensorType& kernel,
    const Array<int>& stride = Array<int>())
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (tensor.dim() != 4 || kernel.dim() != 4 ||
        tensor.shape()[1] != kernel.shape()[1]) {
        char a[64], b[64];
        tensor.shape().toStr(a);
        kernel.shape().toStr(b);
        T_error("Invalid tensor dimension or shape setting. Input tensor "
            "should be 4D with shape (B, C_in, H, W) and kernel should be "
            "4D with shape (C_out, C_in, K_h, K_w), got input tensor shape "
            "(%s) and kernel shape (%s).", a, b);
    }
    if (stride.size() != 0 && stride.size() != 2) {
        T_error("Invalid stride setting in convTranspose2d. Stride can be an empty "
            "array to indicate default stride (1,1) or can be an array with "
            "length of two to indicate the stride amount of two dimensions "
            "(height & width).");
    }
#endif
    /* parse stride */
    Array<int> _stride = T_buildArray<2, int>({ 1, 1 });
    if (stride.size() > 0) { _stride[0] = stride[0]; _stride[1] = stride[1]; }
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_stride[0] < 1 || _stride[1] < 1)
        T_error("error: invalid stride setting in convTranspose2d.");
#endif
    /* compute sizes needed for convolution */
    int inputHeight = tensor.shape()[2], inputWidth = tensor.shape()[3];
    int kernelHeight = kernel.shape()[2], kernelWidth = kernel.shape()[3];
    int kernelNumel = kernelHeight * kernelWidth;
    int inChannels = kernel.shape()[1], outChannels = kernel.shape()[0];
    int outputHeight = ((inputHeight - 1) * _stride[0] + 1) + (kernelHeight - 1) * 2 - (kernelHeight - 1); /* bonpad, pad, conv */
    int outputWidth = ((inputWidth - 1) * _stride[1] + 1) + (kernelWidth - 1) * 2 - (kernelWidth - 1);
    int batchSize = tensor.shape()[0];

    TensorShape mcImageShape = T_buildShape<3>({ inChannels, inputHeight, inputWidth });
    TensorType resultTensor(T_buildShape<4>({ batchSize, outChannels, outputHeight, outputWidth }));
    int mcImageNumel = mcImageShape.numel();

#ifdef NN_ENABLE_OPENMP
#pragma omp parallel for schedule (static)
#endif
    for (int i = 0; i < batchSize; i++) {
        TensorType mcImageTensor(mcImageShape);
        mcImageTensor._dptrWrite(0, mcImageNumel, tensor, i*mcImageNumel);
        TensorType mcConvResult;
        T_convTranspose2d_s(mcImageTensor, kernel, stride).moveStorage(mcConvResult);
        int mcResultNumel = mcConvResult.numel();
        resultTensor._dptrWrite(i*mcResultNumel, mcResultNumel, mcConvResult, 0);
    }
    return resultTensor;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                           3D convolution                              */
/*      (C_in, X_in, Y_in, Z_in) * (C_out, C_in, K_x, K_y, K_z) =        */
/*                    (C_out, X_out, Y_out, Z_out)                       */
/*  ** ARBITRARY STRIDE                                                  */
/*  ** SUITABLE FOR SMALL KERNELS                                        */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
template <typename TensorType>
TensorType T_conv3d_s(
    const TensorType& tensor,
    const TensorType& kernel,
    const Array<int>& stride = Array<int>(),
    const Array<int>& padding = Array<int>())
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (tensor.dim() != 4 || kernel.dim() != 5 ||
        tensor.shape()[0] != kernel.shape()[1]) {
        char a[64], b[64];
        tensor.shape().toStr(a);
        kernel.shape().toStr(b);
        T_error("Invalid tensor dimension or shape setting. Input tensor "
            "should be 4D with shape (C_in, X_in, Y_in, Z_in) and kernel "
            "should be 5D with shape (C_out, C_in, K_x, K_y, K_z), got "
            "input tensor shape (%s) and kernel shape (%s).", a, b);
    }
    if (stride.size() != 0 && stride.size() != 3) {
        T_error("Invalid stride setting in conv3d. Stride can be an empty "
            "array to indicate default stride (1,1,1) or can be an array with "
            "length of three to indicate the stride amount of three dimensions "
            "(X & Y & Z).");
    }
    if (padding.size() != 0 && padding.size() != 3) {
        T_error("Invalid padding setting in conv3d. Padding can be an empty "
            "array to indicate default padding (0,0,0) or can be an array with "
            "length of three to indicate the padding amount of three dimensions "
            "(X & Y & Z).");
    }
#endif

    /* parse stride and padding */
    Array<int> _stride = T_buildArray<3, int>({ 1, 1, 1 }),
        _padding = T_buildArray<3, int>({ 0, 0, 0 });
    if (stride.size() > 0) { _stride[0] = stride[0]; _stride[1] = stride[1]; _stride[2] = stride[2];}
    if (padding.size() > 0) { _padding[0] = padding[0]; _padding[1] = padding[1]; _padding[2] = padding[2];}
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_stride[0] < 1 || _stride[1] < 1 || _stride[2] < 1)
        T_error("error: invalid stride setting in conv3d.");
    if (_padding[0] < 0 || _padding[1] < 0 || _padding[2] < 0)
        T_error("error: invalid padding setting in conv3d.");
#endif

    /* compute sizes needed for convolution */
    int inputX = tensor.shape()[1], inputY = tensor.shape()[2], inputZ = tensor.shape()[3];
    int kernelX = kernel.shape()[2], kernelY = kernel.shape()[3], kernelZ = kernel.shape()[4];
    int kernelNumel = kernelX * kernelY * kernelZ;
    int inChannels = kernel.shape()[1], outChannels = kernel.shape()[0];
    int outputX = (inputX + 2 * _padding[0] - kernelX) / _stride[0] + 1;
    int outputY = (inputY + 2 * _padding[1] - kernelY) / _stride[1] + 1;
    int outputZ = (inputZ + 2 * _padding[2] - kernelZ) / _stride[2] + 1;

#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (outputX < 1 || outputY < 1 || outputZ < 1) {
        char a[64], b[64];
        tensor.shape().toStr(a);
        kernel.shape().toStr(b);
        T_error("Cannot perform 3D convolution due to invalid result image "
            "size (<1). Maybe the kernel size is too large. Tensor size "
            "is (%s), while kernel size is (%s).", a, b);
    }
#endif

    /* pad tensor if needed */
    const TensorType* padded = NULL;
    bool needPadding = false;
    if (_padding[0] > 0 || _padding[1] > 0 || _padding[2] > 0)
        needPadding = true;
    if (needPadding) {
        TensorType* _pTensor = new TensorType();
        (*_pTensor) = tensor.pad(T_buildArray<8, int>({ 0, 0, _padding[0], _padding[0], _padding[1], _padding[1], _padding[2], _padding[2] }));
        padded = _pTensor;
    }
    else padded = &tensor;

    /* NOTE: For 2D convolution, I applied parallelism on the batch dimension.     */
    /* But for 3D convolution, since the number of voxels of a 3D image is usually */
    /* much more than a 2D image, the batch size can be very small (about 1~2) due */
    /* to CPU/GPU memory constraint. So for a 3D convolution I am not going to apply */
    /* parallelism on the batch dimension. Instead, the parallelism is applied to  */
    /* each output channel. */

    TensorType resultTensor(T_buildShape<4>({ outChannels, outputX, outputY, outputZ }));

#ifdef NN_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int och = 0; och < outChannels; och++) {
//        printf("%d ", och);
        /* for each output channel, organize data and prepare for convolution */
        int paddedX = padded->shape()[1], paddedY = padded->shape()[2], paddedZ = padded->shape()[3];
        TensorType tensorA(T_buildShape<2>({ outputX * outputY * outputZ, kernelNumel * inChannels }));
        int addr = 0;
        for (int ix = 0; ix < outputX; ix++) {
            for (int iy = 0; iy < outputY; iy++) {
                for (int iz = 0; iz < outputZ; iz++) {
                    for (int ich = 0; ich < inChannels; ich++) {
                        /* fill convolution unit */
                        for (int ikx = 0; ikx < kernelX; ikx++) {
                            for (int iky = 0; iky < kernelY; iky++) {
                                for (int ikz = 0; ikz < kernelZ; ikz++) {
                                    int x = ix * _stride[0] + ikx;
                                    int y = iy * _stride[1] + iky;
                                    int z = iz * _stride[2] + ikz;
                                    /* write [ich, x, y, z] */
                                    tensorA[addr++] = (*padded)[ich * paddedX * paddedY * paddedZ + x * paddedY * paddedZ + y * paddedZ + z];
                                }
                            }
                        }
                    }
                }
            }
        }
        /* copy kernel only for that output channel */
        TensorType tensorB(T_buildShape<2>({ 1, kernelNumel * inChannels }));
        tensorB._dptrWrite(0, kernelNumel * inChannels, kernel, och * kernelNumel * inChannels);
        //printf("* ");
        TensorType tensorC = T_mm_t(tensorA, tensorB);
        /* write this output channel to result tensor */
        resultTensor._dptrWrite(och * outputX * outputY * outputZ, outputX * outputY * outputZ, tensorC);
    }
    return resultTensor;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                       3D batched convolution                          */
/*    (B, C_in, X_in, Y_in, Z_in) * (C_out, C_in, K_x, K_y, K_z) =       */
/*                    (C_out, X_out, Y_out, Z_out)                       */
/*  ** ARBITRARY STRIDE                                                  */
/*  ** SUITABLE FOR SMALL KERNELS                                        */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
template <typename TensorType>
TensorType T_conv3d(
    const TensorType& tensor,
    const TensorType& kernel,
    const Array<int>& stride = Array<int>(),
    const Array<int>& padding = Array<int>())
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (tensor.dim() != 5 || kernel.dim() != 5 ||
        tensor.shape()[1] != kernel.shape()[1]) {
        char a[64], b[64];
        tensor.shape().toStr(a);
        kernel.shape().toStr(b);
        T_error("Invalid tensor dimension or shape setting. Input tensor "
            "should be 5D with shape (B, C_in, X, Y, Z) and kernel should be "
            "5D with shape (C_out, C_in, K_x, K_y, K_z), got input tensor "
            "shape (%s) and kernel shape (%s).", a, b);
    }
    if (stride.size() != 0 && stride.size() != 3) {
        T_error("Invalid stride setting in conv3d. Stride can be an empty "
            "array to indicate default stride (1,1,1) or can be an array with "
            "length of three to indicate the stride amount of three dimensions "
            "(X & Y & Z).");
    }
    if (padding.size() != 0 && padding.size() != 3) {
        T_error("Invalid padding setting in conv3d. Padding can be an empty "
            "array to indicate default padding (0,0,0) or can be an array with "
            "length of three to indicate the padding amount of three dimensions "
            "(X & Y & Z).");
    }
#endif

    /* parse stride and padding */
    Array<int> _stride = T_buildArray<3, int>({ 1, 1, 1 }),
        _padding = T_buildArray<3, int>({ 0, 0, 0 });
    if (stride.size() > 0) { _stride[0] = stride[0]; _stride[1] = stride[1]; _stride[2] = stride[2]; }
    if (padding.size() > 0) { _padding[0] = padding[0]; _padding[1] = padding[1]; _padding[2] = padding[2]; }
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_stride[0] < 1 || _stride[1] < 1 || _stride[2] < 1)
        T_error("error: invalid stride setting in conv3d.");
    if (_padding[0] < 0 || _padding[1] < 0 || _padding[2] < 0)
        T_error("error: invalid padding setting in conv3d.");
#endif

    /* compute sizes needed for convolution */
    int inputX = tensor.shape()[2], inputY = tensor.shape()[3], inputZ = tensor.shape()[4];
    int kernelX = kernel.shape()[2], kernelY = kernel.shape()[3], kernelZ = kernel.shape()[4];
    int kernelNumel = kernelX * kernelY * kernelZ;
    int inChannels = kernel.shape()[1], outChannels = kernel.shape()[0];
    int outputX = (inputX + 2 * _padding[0] - kernelX) / _stride[0] + 1;
    int outputY = (inputY + 2 * _padding[1] - kernelY) / _stride[1] + 1;
    int outputZ = (inputZ + 2 * _padding[2] - kernelZ) / _stride[2] + 1;
    int batchSize = tensor.shape()[0];

#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (outputX < 1 || outputY < 1 || outputZ < 1) {
        char a[64], b[64];
        tensor.shape().toStr(a);
        kernel.shape().toStr(b);
        T_error("Cannot perform 3D convolution due to invalid result image "
            "size (<1). Maybe the kernel size is too large. Tensor size "
            "is (%s), while kernel size is (%s).", a, b);
    }
#endif

    /* NOTE: since we already applied parallelism to the output channels, */
    /* here we just simply loop all the samples in a batch */

    TensorType resultTensor(T_buildShape<5>({ batchSize, outChannels, outputX, outputY, outputZ }));
    TensorShape mcImageShape = T_buildShape<4>({ inChannels, inputX, inputY, inputZ });
    int mcImageNumel = mcImageShape.numel();
    for (int i = 0; i < batchSize; i++) {
        TensorType mcImageTensor(mcImageShape);
        mcImageTensor._dptrWrite(0, mcImageNumel, tensor, i*mcImageNumel);
        TensorType mcConvResult;
        T_conv3d_s(mcImageTensor, kernel, stride, padding).moveStorage(mcConvResult);
        int mcResultNumel = mcConvResult.numel();
        resultTensor._dptrWrite(i*mcResultNumel, mcResultNumel, mcConvResult, 0);
    }
    return resultTensor;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                       3D transposed convolution                       */
/*  (C_in, X, Y, Z) * (C_out, C_in, K_x, K_y, K_z) = (C_out, X', Y', Z') */
/*  ** ARBITRARY STRIDE                                                  */
/*  ** SUITABLE FOR SMALL KERNELS                                        */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
template <typename TensorType>
TensorType T_convTranspose3d_s(
    const TensorType& tensor,
    const TensorType& kernel,
    const Array<int>& stride = Array<int>())
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (tensor.dim() != 4 || kernel.dim() != 5 ||
        tensor.shape()[0] != kernel.shape()[1]) {
        char a[64], b[64];
        tensor.shape().toStr(a);
        kernel.shape().toStr(b);
        T_error("Invalid tensor dimension or shape setting. Input tensor "
            "should be 4D with shape (C_in, X, Y, Z) and kernel should be 5D "
            "with shape (C_out, C_in, K_x, K_y, K_z), got input tensor shape "
            "(%s) and kernel shape (%s).", a, b);
    }
    if (stride.size() != 0 && stride.size() != 3) {
        T_error("Invalid stride setting in convTranspose3d. Stride can be an empty "
            "array to indicate default stride (1,1,1) or can be an array with "
            "length of three to indicate the stride amount of three dimensions "
            "(X & Y & Z).");
    }
#endif
    Array<int> _stride = T_buildArray<3, int>({ 1,1,1 });
    if (stride.size() > 0) {
        _stride[0] = stride[0];
        _stride[1] = stride[1];
        _stride[2] = stride[2];
    }
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_stride[0] < 1 || _stride[1] < 1 || _stride[2] < 1)
        T_error("error: invalid stride setting in convTranspose3d.");
#endif

    int inChannels = kernel.shape()[1];
    int outChannels = kernel.shape()[0];
    int kernelX = kernel.shape()[2], kernelY = kernel.shape()[3], kernelZ = kernel.shape()[4];
    int kernelNumel = kernelX * kernelY * kernelZ;
    /* constructing X */
    bool unitStride = (_stride[0] == 1 && _stride[1] == 1 && _stride[2] == 1) ? true : false;
    TensorType X;
    if (unitStride) X = tensor;
    else X = tensor.bonpad(_stride, true);
    X = X.pad(T_buildArray<6, int>({ kernelX - 1, kernelX - 1, kernelY - 1, kernelY - 1, kernelZ - 1, kernelZ - 1 }));
    /* constructing K (reverse kernel in every channel) */
    TensorType K = kernel;
    typename TensorType::type* Kptr = &(K[0]);
    for (int i = 0; i < outChannels*inChannels; i++, Kptr += kernelNumel)
        T_reverse(Kptr, kernelNumel);
    /* transposed convolution can be converted to ordinary convolution */
    return T_conv3d_s(X, K);
}


/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                  3D batched transposed convolution                    */
/*    (B, C_in, X_in, Y_in, Z_in) * (C_out, C_in, K_x, K_y, K_z) =       */
/*                    (C_out, X_out, Y_out, Z_out)                       */
/*  ** ARBITRARY STRIDE                                                  */
/*  ** SUITABLE FOR SMALL KERNELS                                        */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
template <typename TensorType>
TensorType T_convTranspose3d(
    const TensorType& tensor,
    const TensorType& kernel,
    const Array<int>& stride = Array<int>())
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (tensor.dim() != 5 || kernel.dim() != 5 ||
        tensor.shape()[1] != kernel.shape()[1]) {
        char a[64], b[64];
        tensor.shape().toStr(a);
        kernel.shape().toStr(b);
        T_error("Invalid tensor dimension or shape setting. Input tensor "
            "should be 5D with shape (B, C_in, X, Y, Z) and kernel should be "
            "5D with shape (C_out, C_in, K_x, K_y, K_z), got input tensor shape "
            "(%s) and kernel shape (%s).", a, b);
    }
    if (stride.size() != 0 && stride.size() != 3) {
        T_error("Invalid stride setting in convTranspose3d. Stride can be an empty "
            "array to indicate default stride (1,1,1) or can be an array with "
            "length of three to indicate the stride amount of three dimensions "
            "(X & Y & Z).");
    }
#endif

    /* parse stride */
    Array<int> _stride = T_buildArray<3, int>({ 1, 1, 1 });
    if (stride.size() > 0) { _stride[0] = stride[0]; _stride[1] = stride[1]; _stride[2] = stride[2]; }
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_stride[0] < 1 || _stride[1] < 1 || _stride[2] < 1)
        T_error("error: invalid stride setting in convTranspose3d.");
#endif
    /* compute sizes needed for convolution */
    int inputX = tensor.shape()[2], inputY = tensor.shape()[3], inputZ = tensor.shape()[4];
    int kernelX = kernel.shape()[2], kernelY = kernel.shape()[3], kernelZ = kernel.shape()[4];
    int kernelNumel = kernelX * kernelY * kernelZ;
    int inChannels = kernel.shape()[1], outChannels = kernel.shape()[0];
    int outputX = ((inputX - 1) * _stride[0] + 1) + (kernelX - 1) * 2 - (kernelX - 1); /* bonpad, pad, conv */
    int outputY = ((inputY - 1) * _stride[1] + 1) + (kernelY - 1) * 2 - (kernelY - 1);
    int outputZ = ((inputZ - 1) * _stride[2] + 1) + (kernelZ - 1) * 2 - (kernelZ - 1);
    int batchSize = tensor.shape()[0];

    TensorShape mcImageShape = T_buildShape<4>({ inChannels, inputX, inputY, inputZ });
    TensorType resultTensor(T_buildShape<5>({ batchSize, outChannels, outputX, outputY, outputZ }));
    int mcImageNumel = mcImageShape.numel();

#ifdef NN_ENABLE_OPENMP
#pragma omp parallel for schedule (static)
#endif
    for (int i = 0; i < batchSize; i++) {
        TensorType mcImageTensor(mcImageShape);
        mcImageTensor._dptrWrite(0, mcImageNumel, tensor, i*mcImageNumel);
        TensorType mcConvResult;
        T_convTranspose3d_s(mcImageTensor, kernel, stride).moveStorage(mcConvResult);
        int mcResultNumel = mcConvResult.numel();
        resultTensor._dptrWrite(i*mcResultNumel, mcResultNumel, mcConvResult, 0);
    }
    return resultTensor;

}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * * * * * * * BACKWARD FUNCTIONS  * * * * * * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

template <typename TensorType>
void _T_conv2d_backward(
    /* inputs */
    const TensorType& dLdY, 
    const TensorType& X, 
    const TensorType& K, 
    const Array<int>& stride,
    /* computed gradients */
    TensorType& dLdK, 
    TensorType& dLdb, 
    TensorType& dLdX)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdY.dim() != 4)
        T_error("Invalid dimension for dLdY (expected 4, but %d was given.)", dLdY.dim());
    if (X.dim() != 4)
        T_error("Invalid dimension for X (expected 4, but %d was given.)", X.dim());
    if (K.dim() != 4)
        T_error("Invalid dimension for K (expected 4, but %d was given.)", K.dim());
    if (stride.size() != 2)
        T_error("Invalid stride setting (expected size 2, but %d was given.)", stride.size());
    if (dLdY.shape()[0] != X.shape()[0])
        T_error("Batch size not equal.");
    if (dLdY.shape()[1] != K.shape()[0])
        T_error("Output channels not equal.");
    if (K.shape()[2] < 1 || K.shape()[3] < 1)
        T_error("Invalid kernel size.");
#endif

    int batchSize = dLdY.shape()[0];
    int outChannels = K.shape()[0], inChannels = K.shape()[1];
    int kernelHeight = K.shape()[2], kernelWidth = K.shape()[3];
    int yHeight = dLdY.shape()[2], yWidth = dLdY.shape()[3];
    int xHeight = X.shape()[2], xWidth = X.shape()[3];
    int hStride = stride[0], wStride = stride[1];
    int I;

    /* compute dLdK */
    {
        /* dLdY: (batch, outchannels,  outheight, outwidth) */
        /*    X: (batch,  inchannels,   inheight, inwidth)  */
        /*    K: (outchannels, inchannels, kheight, kwidth) */

        TensorType tensorA(T_buildShape<3>({ batchSize, outChannels, yHeight*yWidth }));
        TensorType tensorB(T_buildShape<3>({ batchSize, inChannels*kernelHeight*kernelWidth, yHeight*yWidth }));
        I = 0;
        for (int b = 0; b < batchSize; b++) {
            for (int och = 0; och < outChannels; och++) {
                for (int yh = 0; yh < yHeight; yh++) {
                    for (int yw = 0; yw < yWidth; yw++) {
                        tensorA[I++] = dLdY[b * outChannels * yHeight * yWidth + och * yHeight*yWidth + yh * yWidth + yw];
                        //                printf("writing [%d,%d,%d]\n", och, yx, yy);
                    }
                }
            }
        }
        I = 0;
        for (int b = 0; b < batchSize; b++) {
            for (int ich = 0; ich < inChannels; ich++) {
                for (int kh = 0; kh < kernelHeight; kh++) {
                    for (int kw = 0; kw < kernelWidth; kw++) {
                        for (int yh = 0; yh < yHeight; yh++) {
                            for (int yw = 0; yw < yWidth; yw++) {
                                int x = yw * wStride + kw, y = yh * hStride + kh;
                                tensorB[I++] = X[b * inChannels * xHeight * xWidth + ich * xHeight * xWidth + y * xWidth + x];
                                //                        printf("writing [%d,%d,%d]\n", ich, y, x);
                            }
                        }
                    }
                }
            }
        }
        TensorType(T_buildShape<4>({ outChannels, inChannels, kernelHeight, kernelWidth })).moveStorage(dLdK);
        dLdK.fill_(TensorType::type(0.0));
#ifdef NN_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int b = 0; b < batchSize; b++) {
            TensorType _tensorA(T_buildShape<2>({ outChannels, yHeight*yWidth }));
            TensorType _tensorB(T_buildShape<2>({ inChannels*kernelHeight*kernelWidth, yHeight*yWidth }));
            int Anumel = outChannels * yHeight * yWidth;
            int Bnumel = inChannels * kernelHeight * kernelWidth * yHeight * yWidth;
            _tensorA._dptrWrite(0, Anumel, tensorA, b*Anumel);
            _tensorB._dptrWrite(0, Bnumel, tensorB, b*Bnumel);
            TensorType _tensorC = T_mm_t(_tensorA, _tensorB).reshape(T_buildShape<4>({ outChannels, inChannels, kernelHeight, kernelWidth }));
#ifdef NN_ENABLE_OPENMP
#pragma omp critical(bconv2d_dldk)
#endif
            dLdK += _tensorC;
        }
    }

    /* compute dLdX */
    {
        /* dLdY: (batchsize, outchannels,  outheight, outwidth) */
        /*    K: (outchannels, in channels, kheight, kwidth)    */
        /*   Kp: (inchannels, outchannels, kheight, kwidth)     */
        /* Kp is the transposed & reversed version of K         */
        
        /* bed-of-nail padding for dLdY */
        TensorType dLdY_bon = dLdY.bonpad(stride, true).pad(T_buildArray<4, int>({ kernelHeight - 1, kernelHeight - 1, kernelWidth - 1,kernelWidth - 1 }));
        
        /* transpose and inverse kernel */
        TensorType Kp = K.transpose(T_buildArray<4, int>({ 2,1,3,4 }));
        int Knumel = kernelWidth * kernelHeight;
        for (int c = 0; c < inChannels * outChannels; c++) {
            int iStart = c * Knumel;
            REAL* ptr = &(Kp[iStart]);
            T_reverse(ptr, Knumel);
        }

        dLdX = T_conv2d(dLdY_bon, Kp);
    }

    /* compute dLdb */
    {
        TensorType(T_buildShape<1>({ outChannels })).moveStorage(dLdb);
        dLdb.fill_(TensorType::type(0.0));

        for (int b = 0; b < batchSize; b++) {
            for (int c = 0; c < outChannels; c++) {
                int offset = b * outChannels * yHeight * yWidth + c * yHeight * yWidth;
                int n = yHeight * yWidth;
                typename TensorType::type sum = TensorType::type(0.0);
                for (int i = offset; i < offset + n; i++) {
                    sum += dLdY[i];
                }
                dLdb[c] += sum;
            }
        }
    }
}

template <typename TensorType>
void _T_convTranspose2d_backward(
    /* inputs */
    const TensorShape& inShape, /* shape of dLdX */
    const TensorType& dLdY, 
    const TensorType& X, 
    const TensorType& K, 
    const Array<int>& stride,
    /* computed gradients */
    TensorType& dLdK, 
    TensorType& dLdb, 
    TensorType& dLdX)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdY.dim() != 4)
        T_error("Invalid dimension for dLdY (expected 4, but %d was given.)", dLdY.dim());
    if (X.dim() != 4)
        T_error("Invalid dimension for X (expected 4, but %d was given.)", X.dim());
    if (K.dim() != 4)
        T_error("Invalid dimension for K (expected 4, but %d was given.)", K.dim());
    if (stride.size() != 2)
        T_error("Invalid stride setting (expected size 2, but %d was given.)", stride.size());
    if (dLdY.shape()[0] != X.shape()[0])
        T_error("Batch size not equal.");
    if (dLdY.shape()[1] != K.shape()[0])
        T_error("Output channels not equal.");
    if (K.shape()[2] < 1 || K.shape()[3] < 1)
        T_error("Invalid kernel size.");
#endif
    Array<int> _stride = T_buildArray<2, int>({ 1,1 });
    if (stride.size() > 0) {
        _stride[0] = stride[0];
        _stride[1] = stride[1];
    }
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_stride[0] < 1 || _stride[1] < 1)
        T_error("error: invalid stride setting in convTranspose2d.");
#endif

    int inChannels = K.shape()[1];
    int outChannels = K.shape()[0];
    int kernelHeight = K.shape()[2];
    int kernelWidth = K.shape()[3];
    int kernelNumel = kernelHeight * kernelWidth;
    int batchSize = X.shape()[0];
    int outputHeight = dLdY.shape()[2];
    int outputWidth = dLdY.shape()[3];
    
    /* construct Xc */
    bool unitStride = (_stride[0] == 1 && _stride[1] == 1) ? true : false;
    TensorType Xc;
    if (unitStride) Xc = X;
    else Xc = X.bonpad(_stride, true);
    Xc = Xc.pad(T_buildArray<4, int>({ kernelHeight - 1, kernelHeight - 1, kernelWidth - 1, kernelWidth - 1 }));

    /* construct Kc */
    TensorType Kc = K;
    typename TensorType::type* Kptr = NULL;
    Kptr = &(Kc[0]);
    for (int i = 0; i < outChannels*inChannels; i++, Kptr += kernelNumel)
        T_reverse(Kptr, kernelNumel);

    /* calculate gradients using ordinary convolution backward */
    TensorType dLdKc, dLdXc;
    /* NOTE: here we must use unit stride for ordinary backward convolution */
    _T_conv2d_backward(dLdY, Xc, Kc, T_buildArray<2, int>({1,1}), dLdKc, dLdb, dLdXc); 
    /* reconstruct dLdK from dLdKc */
    dLdK = dLdKc;
    Kptr = &(dLdK[0]);
    for (int i = 0; i < outChannels*inChannels; i++, Kptr += kernelNumel)
        T_reverse(Kptr, kernelNumel);

    /* reconstruct dLdX from dLdXc */
    dLdXc = dLdXc.shrink(T_buildArray<4, int>({ kernelHeight - 1, kernelHeight - 1, kernelWidth - 1, kernelWidth - 1 }));
    //TensorType(T_buildShape<4>({
    //    batchSize, inChannels, (outputHeight + (_stride[0] - 1)) / _stride[0],
    //    (outputWidth + (_stride[1] - 1)) / _stride[1] })).moveStorage(dLdX);
    TensorType(inShape).moveStorage(dLdX);

    /* downsample dLdXc to construct dLdX */
    Array<int> steps;
    int D = dLdXc.dim(), S = 2;
    for (int i = 0; i < D; i++) {
        if (i < D - S) steps.append(1);
        else steps.append(_stride[i - D + S]);
    }
#ifdef NN_ENABLE_RUNTIME_CHECKING
    {
        int cx = dLdXc.shape()[2], cy = dLdXc.shape()[3];
        int ix = inShape[2], iy = inShape[3];
        int xStep = steps[2], yStep = steps[3];
        /* ensure downsample is correct */
        bool valid = true;
        valid &= (cx >= (ix - 1) * xStep + 1 && cx <= ix * xStep) ? true : false;
        valid &= (cy >= (iy - 1) * yStep + 1 && cy <= iy * yStep) ? true : false;
        if (!valid)
            T_error("Incorrect inShape setting.");
    }
#endif
    TensorCoord coord(D);
    TensorShape shp = dLdXc.shape();
    int I = 0;
    do {
        dLdX[I++] = dLdXc[coord];
    } while (coord.next(shp, steps));

}

template <typename TensorType>
void _T_conv3d_backward(
    /* inputs */
    const TensorType& dLdY,
    const TensorType& X,
    const TensorType& K,
    const Array<int>& stride,
    /* computed gradients */
    TensorType& dLdK,
    TensorType& dLdb,
    TensorType& dLdX)
{


#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdY.dim() != 5)
        T_error("Invalid dimension for dLdY (expected 5, but %d was given.)", dLdY.dim());
    if (X.dim() != 5)
        T_error("Invalid dimension for X (expected 5, but %d was given.)", X.dim());
    if (K.dim() != 5)
        T_error("Invalid dimension for K (expected 5, but %d was given.)", K.dim());
    if (stride.size() != 3)
        T_error("Invalid stride setting (expected size 3, but %d was given.)", stride.size());
    if (dLdY.shape()[0] != X.shape()[0])
        T_error("Batch size not equal.");
    if (dLdY.shape()[1] != K.shape()[0])
        T_error("Output channels not equal.");
    if (K.shape()[2] < 1 || K.shape()[3] < 1 || K.shape()[4] < 1)
        T_error("Invalid kernel size.");
#endif

    int batchSize = dLdY.shape()[0];
    int outChannels = K.shape()[0], inChannels = K.shape()[1];
    int kX = K.shape()[2], kY = K.shape()[3], kZ = K.shape()[4];
    int yX = dLdY.shape()[2], yY = dLdY.shape()[3], yZ = dLdY.shape()[4];
    int xX = X.shape()[2], xY = X.shape()[3], xZ = X.shape()[4];
    int xStride = stride[0], yStride = stride[1], zStride = stride[2];
    
    /* compute dLdK */
    {
        /* dLdY: (batch, outchannels, yX, yY, yZ) */
        /*    X: (batch,  inchannels, xX, xY, xZ) */
        /*    K: (outchannels, inchannels, kX, kY, kZ) */

        /* here i provided two implementations */

        /* v2 - slightly faster than v1 */
        {
            
            TensorType(T_buildShape<5>({ outChannels, inChannels, kX, kY, kZ })).moveStorage(dLdK);
            dLdK.fill_(TensorType::type(0.0));

            for (int b = 0; b < batchSize; b++) {
                TensorType dLdK_sample(T_buildShape<5>({ outChannels, inChannels, kX, kY, kZ }));
                TensorType tensorB(T_buildShape<2>({ inChannels*kX*kY*kZ, yX*yY*yZ }));
                int I = 0;
                for (int ich = 0; ich < inChannels; ich++) {
                    for (int kx = 0; kx < kX; kx++) {
                        for (int ky = 0; ky < kY; ky++) {
                            for (int kz = 0; kz < kZ; kz++) {
                                for (int yx = 0; yx < yX; yx++) {
                                    for (int yy = 0; yy < yY; yy++) {
                                        for (int yz = 0; yz < yZ; yz++) {
                                            int x = yx * xStride + kx, y = yy * yStride + ky, z = yz * zStride + kz;
                                            tensorB[I++] = X[b * inChannels*xX*xY*xZ + ich * xX*xY*xZ + x * xY*xZ + y * xZ + z];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
//#ifdef NN_ENABLE_OPENMP
//#pragma omp parallel for schedule(static)
//#endif
                for (int och = 0; och < outChannels; och++) {
                    TensorType tensorA(T_buildShape<2>({ 1, yX*yY*yZ }));
                    int J = 0;
                    for (int yx = 0; yx < yX; yx++) {
                        for (int yy = 0; yy < yY; yy++) {
                            for (int yz = 0; yz < yZ; yz++) {
                                tensorA[J++] = dLdY[b * outChannels*yX*yY*yZ + och * yX*yY*yZ + yx * yY*yZ + yy * yZ + yz];
                            }
                        }
                    }
                    TensorType tensorC = T_mm_t(tensorA, tensorB);
                    int channelStride = tensorC.numel();
                    dLdK_sample._dptrWrite(channelStride * och, channelStride, tensorC, 0);
                }
//#ifdef NN_ENABLE_OPENMP
//#pragma omp critical(bconv2d_dldk)
//#endif
                dLdK += dLdK_sample;
            }

        }

        /* v1 - old implementation */
//        { 
//            int I;
//            TensorType tensorA(T_buildShape<3>({ batchSize, outChannels, yX*yY*yZ }));
//            TensorType tensorB(T_buildShape<3>({ batchSize, inChannels*kX*kY*kZ, yX*yY*yZ }));
//            I = 0;
//            for (int b = 0; b < batchSize; b++) {
//                for (int och = 0; och < outChannels; och++) {
//                    for (int yx = 0; yx < yX; yx++) {
//                        for (int yy = 0; yy < yY; yy++) {
//                            for (int yz = 0; yz < yZ; yz++) {
//                                tensorA[I++] = dLdY[b * outChannels*yX*yY*yZ + och * yX*yY*yZ + yx * yY*yZ + yy * yZ + yz];
//                            }
//                        }
//                    }
//                }
//            }
//            I = 0;
//            for (int b = 0; b < batchSize; b++) {
//                for (int ich = 0; ich < inChannels; ich++) {
//                    for (int kx = 0; kx < kX; kx++) {
//                        for (int ky = 0; ky < kY; ky++) {
//                            for (int kz = 0; kz < kZ; kz++) {
//                                for (int yx = 0; yx < yX; yx++) {
//                                    for (int yy = 0; yy < yY; yy++) {
//                                        for (int yz = 0; yz < yZ; yz++) {
//                                            int x = yx * xStride + kx, y = yy * yStride + ky, z = yz * zStride + kz;
//                                            tensorB[I++] = X[b * inChannels*xX*xY*xZ + ich * xX*xY*xZ + x * xY*xZ + y * xZ + z];
//                                        }
//                                    }
//                                }
//                            }
//                        }
//                    }
//                }
//            }
//            TensorType(T_buildShape<5>({ outChannels, inChannels, kX, kY, kZ })).moveStorage(dLdK);
//            dLdK.fill_(TensorType::type(0.0));
////#ifdef NN_ENABLE_OPENMP
////#pragma omp parallel for schedule(static)
////#endif
//            for (int b = 0; b < batchSize; b++) {
//                TensorType _tensorA(T_buildShape<2>({ outChannels, yX*yY*yZ }));
//                TensorType _tensorB(T_buildShape<2>({ inChannels*kX*kY*kZ, yX*yY*yZ }));
//                int Anumel = outChannels * yX * yY * yZ;
//                int Bnumel = inChannels * kX * kY * kZ * yX * yY * yZ;
//                _tensorA._dptrWrite(0, Anumel, tensorA, b*Anumel);
//                _tensorB._dptrWrite(0, Bnumel, tensorB, b*Bnumel);
//                TensorType _tensorC = T_mm_t(_tensorA, _tensorB).
//                    reshape(T_buildShape<5>({ outChannels, inChannels, kX, kY, kZ }));
////#ifdef NN_ENABLE_OPENMP
////#pragma omp critical(bconv2d_dldk)
////#endif
//                dLdK += _tensorC;
//            }
//
//        }

    }

    /* compute dLdX */
    {
        /* dLdY: (batchsize, outchannels,  outheight, outwidth) */
        /*    K: (outchannels, in channels, kheight, kwidth)    */
        /*   Kp: (inchannels, outchannels, kheight, kwidth)     */
        /* Kp is the transposed & reversed version of K         */

        /* bed-of-nail padding for dLdY */
        TensorType dLdY_bon = dLdY.bonpad(stride, true).
            pad(T_buildArray<6, int>({ kX - 1, kX - 1, kY - 1,kY - 1, kZ - 1,kZ - 1 }));

        /* transpose and inverse kernel */
        TensorType Kp = K.transpose(T_buildArray<5, int>({ 2,1,3,4,5 }));
        int Knumel = kX * kY * kZ;
        for (int c = 0; c < inChannels * outChannels; c++) {
            int iStart = c * Knumel;
            REAL* ptr = &(Kp[iStart]);
            T_reverse(ptr, Knumel);
        }

        dLdX = T_conv3d(dLdY_bon, Kp);

    }

    /* compute dLdb */
    {
        
        TensorType(T_buildShape<1>({ outChannels })).moveStorage(dLdb);
        dLdb.fill_(TensorType::type(0.0));

        for (int b = 0; b < batchSize; b++) {
            for (int c = 0; c < outChannels; c++) {
                int offset = b * outChannels * yX*yY*yZ + c * yX*yY*yZ;
                int n = yX*yY*yZ;
                typename TensorType::type sum = (const typename TensorType::type)(0.0);
                for (int i = offset; i < offset + n; i++) {
                    sum += dLdY[i];
                }
                dLdb[c] += sum;
            }
        }

    }
}


template <typename TensorType>
void _T_convTranspose3d_backward(
    /* inputs */
    const TensorShape& inShape,
    const TensorType& dLdY,
    const TensorType& X,
    const TensorType& K,
    const Array<int>& stride,
    /* computed gradients */
    TensorType& dLdK,
    TensorType& dLdb,
    TensorType& dLdX)
{
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (dLdY.dim() != 5)
        T_error("Invalid dimension for dLdY (expected 5, but %d was given.)", dLdY.dim());
    if (X.dim() != 5)
        T_error("Invalid dimension for X (expected 5, but %d was given.)", X.dim());
    if (K.dim() != 5)
        T_error("Invalid dimension for K (expected 5, but %d was given.)", K.dim());
    if (stride.size() != 3)
        T_error("Invalid stride setting (expected size 3, but %d was given.)", stride.size());
    if (dLdY.shape()[0] != X.shape()[0])
        T_error("Batch size not equal.");
    if (dLdY.shape()[1] != K.shape()[0])
        T_error("Output channels not equal.");
    if (K.shape()[2] < 1 || K.shape()[3] < 1 || K.shape()[4] < 1)
        T_error("Invalid kernel size.");
#endif
    Array<int> _stride = T_buildArray<3, int>({ 1,1,1 });
    if (stride.size() > 0) { _stride[0] = stride[0]; _stride[1] = stride[1]; _stride[2] = stride[2]; }
#ifdef NN_ENABLE_RUNTIME_CHECKING
    if (_stride[0] < 1 || _stride[1] < 1 || _stride[2] < 1)
        T_error("error: invalid stride setting in convTranspose3d.");
#endif

    int inChannels = K.shape()[1];
    int outChannels = K.shape()[0];
    int kernelX = K.shape()[2];
    int kernelY = K.shape()[3];
    int kernelZ = K.shape()[4];
    int kernelNumel = kernelX * kernelY * kernelZ;
    int batchSize = X.shape()[0];
    int outputX = dLdY.shape()[2];
    int outputY = dLdY.shape()[3];
    int outputZ = dLdY.shape()[4];

    /* construct Xc */
    bool unitStride = (_stride[0] == 1 && _stride[1] == 1 && _stride[2] == 1) ? true : false;
    TensorType Xc;
    if (unitStride) Xc = X;
    else Xc = X.bonpad(_stride, true);
    Xc = Xc.pad(T_buildArray<6, int>({ kernelX - 1, kernelX - 1, kernelY - 1, kernelY - 1,kernelZ - 1, kernelZ - 1 }));

    /* construct Kc */
    TensorType Kc = K;
    typename TensorType::type* Kptr = NULL;
    Kptr = &(Kc[0]);
    for (int i = 0; i < outChannels*inChannels; i++, Kptr += kernelNumel)
        T_reverse(Kptr, kernelNumel);

    /* calculate gradients using ordinary convolution backward */
    TensorType dLdKc, dLdXc;
    /* NOTE: here we must use unit stride for ordinary backward convolution */
    _T_conv3d_backward(dLdY, Xc, Kc, T_buildArray<3, int>({ 1,1,1 }), dLdKc, dLdb, dLdXc);
    /* reconstruct dLdK from dLdKc */
    dLdK = dLdKc;
    Kptr = &(dLdK[0]);
    for (int i = 0; i < outChannels*inChannels; i++, Kptr += kernelNumel)
        T_reverse(Kptr, kernelNumel);

    /* reconstruct dLdX from dLdXc */
    dLdXc = dLdXc.shrink(T_buildArray<6, int>({ kernelX - 1, kernelX - 1, kernelY - 1, kernelY - 1, kernelZ - 1, kernelZ - 1 }));
    //TensorType(T_buildShape<5>({
    //    batchSize, inChannels, 
    //    (outputX + (_stride[0] - 1)) / _stride[0],
    //    (outputY + (_stride[1] - 1)) / _stride[1],
    //    (outputZ + (_stride[2] - 1)) / _stride[2]})).moveStorage(dLdX);
    TensorType(inShape).moveStorage(dLdX);

    Array<int> steps;
    int D = dLdXc.dim(), S = 3;
    for (int i = 0; i < D; i++) {
        if (i < D - S) steps.append(1);
        else steps.append(_stride[i - D + S]);
    }
#ifdef NN_ENABLE_RUNTIME_CHECKING
    {
        int cx = dLdXc.shape()[2], cy = dLdXc.shape()[3], cz = dLdXc.shape()[4];
        int ix = inShape[2], iy = inShape[3], iz = inShape[4];
        int xStep = steps[2], yStep = steps[3], zStep = steps[4];
        /* ensure downsample is correct */
        bool valid = true;
        valid &= (cx >= (ix - 1) * xStep + 1 && cx <= ix * xStep) ? true : false;
        valid &= (cy >= (iy - 1) * yStep + 1 && cy <= iy * yStep) ? true : false;
        valid &= (cz >= (iz - 1) * zStep + 1 && cz <= iz * zStep) ? true : false;
        if (!valid)
            T_error("Incorrect inShape setting.");
    }
#endif
    TensorCoord coord(D);
    TensorShape shp = dLdXc.shape();
    int I = 0;
    do {
        dLdX[I++] = dLdXc[coord];
    } while (coord.next(shp, steps));

}
