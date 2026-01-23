#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "tensor.h"

typedef struct {
    enum {
        ACTIVATION_RELU,
        ACTIVATION_SOFTMAX,
        ACTIVATION_NONE
    } type;
    Tensor* input_cache;
} Activation;

Activation* activation_relu(void);
Activation* activation_softmax(void);
void activation_free(Activation* act);

// Forward pass
Tensor* activation_forward(Activation* act, const Tensor* input);

// Backward pass
Tensor* activation_backward(Activation* act, const Tensor* grad_output);

#endif // ACTIVATIONS_H
