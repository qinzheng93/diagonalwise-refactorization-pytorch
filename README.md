# Diagonalwise Refactorization: An Efficient Training Method for Depthwise Convolutions

This is the PyTorch implementation of Diagonalwise Refactorization.

Diagonalwise Refactorization is an efficient implementation for depthwise convolutions.
The key idea of Diagonalwise Refactorization is to rearrange the weight vectors of a depthwise convolution into a large diagonal weight matrixi, so as to convert the depthwise convolution into one single standard convolution, which is well supported by the cuDNN library that is highly-optimized for GPU computations.

In PyTorch, Diagonalwise Refactorization is implemented in Python and can be further accelerated using C++.


