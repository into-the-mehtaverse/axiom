#include "loss.h"
#include <math.h>

float loss_cross_entropy(const Tensor* predictions, const Tensor* targets) {
    // TODO: Implement cross-entropy loss calculation
    return 0.0f;
}

Tensor* loss_cross_entropy_grad(const Tensor* predictions, const Tensor* targets) {
    // TODO: Implement cross-entropy gradient
    return NULL;
}

float loss_mse(const Tensor* predictions, const Tensor* targets) {
    // TODO: Implement MSE loss calculation
    return 0.0f;
}

Tensor* loss_mse_grad(const Tensor* predictions, const Tensor* targets) {
    // TODO: Implement MSE gradient
    return NULL;
}
