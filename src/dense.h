#ifndef DENSE_H
#define DENSE_H

#include "tensor.h"

typedef struct {
    Tensor* weights;
    Tensor* biases;
    Tensor* grad_weights;
    Tensor* grad_biases;
    Tensor* input_cache;
    size_t input_size;
    size_t output_size;
} DenseLayer;

DenseLayer* dense_create(size_t input_size, size_t output_size);
void dense_free(DenseLayer* layer);

// Forward pass
Tensor* dense_forward(DenseLayer* layer, const Tensor* input);

// Backward pass
Tensor* dense_backward(DenseLayer* layer, const Tensor* grad_output);

#endif // DENSE_H
