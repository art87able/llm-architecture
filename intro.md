# Optimization in PyTorch

This book covers optimization techniques in PyTorch, including Gradient Descent, SGD, Numerical Stability, and L1 Regularization.

## Learning Goals

By completing this material, you will:

1. Understand preprocessing design choices (tokenization, fixed vocabulary)
2. Implement and train Logistic Regression manually in PyTorch using SGD
3. Explain why numerical stability matters in softmax and log-loss
4. Understand how optimization parameters (learning rate, batch size) affect training
5. Understand the effect of L1 regularization and why it encourages sparsity
6. Understand how optimization algorithms behave on different loss landscapes

## Contents

### Part 1 - SGD for Logistic Regression

Using the SST-2 (Stanford Sentiment Treebank) dataset for binary sentiment classification:

- Data loading and preprocessing (text cleaning, tokenization, Bag-of-Words)
- Implementing Logistic Regression in PyTorch
- Training with mini-batch SGD
- Experimenting with learning rates and batch sizes
- L1 Regularization and sparsity

### Part 2 - Comparing Optimization Algorithms

Implementing and comparing optimizers on convex and non-convex functions:

- Gradient Descent (GD)
- Momentum
- AdaGrad
- Adam

### Bonus

An exploration of the L1 regularization phenomenon and proximal descent.

## Setup

```bash
pip install -r requirements.txt
```

To build the book locally:

```bash
jupyter-book build .
```
