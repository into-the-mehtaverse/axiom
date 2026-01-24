#include "loss.h"
#include <math.h>

float loss_cross_entropy(const Tensor* predictions, const Tensor* targets) {
    if (predictions == NULL || targets == NULL) return 0.0f;
    if (predictions->ndim != targets->ndim) return 0.0f;
        for (size_t i = 0; i < predictions->ndim; i++) {
            if (predictions->shape[i] != targets->shape[i]) return 0.0f;
        }

    float loss = 0.0;
    float epsilon = 1e-7f; // for clipping , numerical stability

    for (size_t i = 0; i < targets->size; i++) {
        float current = targets->data[i];
        float clipped = (predictions->data[i] < epsilon) ? epsilon :
                (predictions->data[i] > 1.0f - epsilon) ? 1.0f - epsilon :
                predictions->data[i];
        loss += -log(clipped) * current;
    }
    return loss / (float)predictions->shape[0];
}

Tensor* loss_cross_entropy_grad(const Tensor* predictions, const Tensor* targets) {
    if (predictions == NULL || targets == NULL) return NULL;
    if (predictions->ndim != targets->ndim) return NULL;
    for (size_t i = 0; i < predictions->ndim; i++) {
        if (predictions->shape[i] != targets->shape[i]) return NULL;
    }

    Tensor* grad = tensor_subtract(predictions, targets);
    if (grad == NULL) return NULL;

    return grad;
}

float loss_mse(const Tensor* predictions, const Tensor* targets) {
    if (predictions == NULL || targets == NULL) return 0.0f;
    if (predictions->ndim != targets->ndim) return 0.0f;
        for (size_t i = 0; i < predictions->ndim; i++) {
            if (predictions->shape[i] != targets->shape[i]) return 0.0f;
        }

    float sum = 0.0;
    for (size_t i = 0; i < targets->size; i++) {
        float diff = predictions->data[i] - targets->data[i];
        sum += diff * diff;
    }
    return sum / (float)targets->size;
}

Tensor* loss_mse_grad(const Tensor* predictions, const Tensor* targets) {
    if (predictions == NULL || targets == NULL) return NULL;
    if (predictions->ndim != targets->ndim) return NULL;
    for (size_t i = 0; i < predictions->ndim; i++) {
        if (predictions->shape[i] != targets->shape[i]) return NULL;
    }

    Tensor* grad = tensor_subtract(predictions, targets);
    if (grad == NULL) return NULL;

    // Divide each element by batch size
    for (size_t i = 0; i < grad->size; i++) {
        grad->data[i] /= (float)predictions->shape[0];
    }

    return grad;
}
