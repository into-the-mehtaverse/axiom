#include "optimizer.h"
#include <stdlib.h>

Optimizer* optimizer_sgd_create(float learning_rate) {
    // TODO: Implement SGD optimizer creation
    return NULL;
}

void optimizer_free(Optimizer* opt) {
    // TODO: Implement optimizer cleanup
}

void optimizer_step(Optimizer* opt, void* layer, const void* gradients) {
    // TODO: Implement weight update step: weights -= lr * gradients
}
