# Breeze
A super light-weight CPU deep learning framework with no external dependency.
---
**Implemented features**
1) 2/3D batched convolution & transposed convolution and their backprop calculations
2) batch normalization and its backprop calculation
3) 2/3D max ppoling & average pooling (including backprop)
4) MLP (linear transform) layer
5) simple activation layers (ReLU, LeakyReLU, softmax)
6) cross entropy loss, Dice loss, mean square error (MSE) loss
7) optimized convolution operations (AVX+OpenMP) and high performance matrix multiplication
8) a simple builtin tensor library written from scratch
