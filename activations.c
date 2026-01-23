#include "activations.h"
#include <stdlib.h>
#include <math.h>

Activation* activation_relu(void) {
    // TODO: Implement ReLU activation creation
    return NULL;
}

Activation* activation_softmax(void) {
    // TODO: Implement Softmax activation creation
    return NULL;
}

void activation_free(Activation* act) {
    // TODO: Implement activation cleanup
}

Tensor* activation_forward(Activation* act, const Tensor* input) {
    // TODO: Implement forward pass for ReLU and Softmax
    return NULL;
}

Tensor* activation_backward(Activation* act, const Tensor* grad_output) {
    // TODO: Implement backward pass for ReLU and Softmax
    return NULL;
}
