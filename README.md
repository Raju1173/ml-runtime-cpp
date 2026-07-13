# CNN Training Runtime

A small CNN runtime built from scratch in C++ as an educational project

# Benchmarks (GEMM 512×512)
- ~25 GFLOPS on 512×512 matrix multiplication (single core)

| Version            | Time (ms) |
|--------------------|-----------|
| Naive (ijk)        | ~66 ms    |
| Reordered (ikj)    | ~43 ms    |
| + pointer reuse    | ~13 ms    |
| + blocking (64 X 64, optimal on benchmark CPU) | ~10.4 ms  |

CPU: Intel i5-13420H (13th Gen)

# Features
### Tensor Core
- Manual memory management using raw pointers
- Explicit shape tracking
- Contiguous row major layout
### Layers
- Convolution (im2col + GEMM)
- ReLU
- Max Pooling
- Linear
### Training
- Full backward propagation for all layers
- Softmax + Cross Entropy loss
- Stochastic Gradient Descent (SGD)
### Data Flow
- im2col / col2im transformations
- Manual gradient propagation

# Demo (Training/DoodlePredictor.py)

- Trained on a subset of Google QuickDraw (10 classes)
- Achieved ~70% accuracy on test data
- Supports real time inference on hand-drawn input
