#include "dense.h"
#include <stdlib.h>

DenseLayer* dense_create(size_t input_size, size_t output_size) {
    // TODO: Implement dense layer creation with weight initialization
    return NULL;
}

void dense_free(DenseLayer* layer) {
    // TODO: Implement dense layer cleanup
}

Tensor* dense_forward(DenseLayer* layer, const Tensor* input) {
    // TODO: Implement forward pass: output = input @ weights + biases
    return NULL;
}

Tensor* dense_backward(DenseLayer* layer, const Tensor* grad_output, float learning_rate) {
    // TODO: Implement backward pass with gradient computation and weight updates
    return NULL;
}
