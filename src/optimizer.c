#include "optimizer.h"
#include <stdlib.h>
#include "dense.h"
#include "axiom.h"

Optimizer* optimizer_sgd_create(float learning_rate) {
    Optimizer* opt = malloc(sizeof(Optimizer));
    if (opt == NULL) return NULL;

    opt->learning_rate = learning_rate;
    opt->type = OPTIMIZER_SGD;

    return opt;
}

void optimizer_free(Optimizer* opt) {
    if (opt == NULL) return;
    free(opt);
}

void optimizer_step(Optimizer* opt, Layer* layer) {
    if (opt == NULL || layer == NULL) return;

    switch (layer->type) {
        case LAYER_DENSE: {
            DenseLayer* dense = layer->layer.dense;
            if (dense == NULL || dense->grad_weights == NULL || dense->grad_biases == NULL) return;

            // update weights: W -= lr * dW
            for (size_t i = 0; i < dense->weights->size; i++) {
                dense->weights->data[i] -= opt->learning_rate * dense->grad_weights->data[i];
            }

            // update biases: b -= lr * db
            for (size_t i = 0; i < dense->biases->size; i++) {
                dense->biases->data[i] -= opt->learning_rate * dense->grad_biases->data[i];
            }
            break;
        }
        case LAYER_ACTIVATION:
            // activations have no trainable parameters
            break;
        default:
            // unknown layer type - do nothing
            break;
    }
}
