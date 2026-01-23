#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"

// Cross-entropy loss for classification
float loss_cross_entropy(const Tensor* predictions, const Tensor* targets);
Tensor* loss_cross_entropy_grad(const Tensor* predictions, const Tensor* targets);

// Mean Squared Error for regression
float loss_mse(const Tensor* predictions, const Tensor* targets);
Tensor* loss_mse_grad(const Tensor* predictions, const Tensor* targets);

#endif // LOSS_H
