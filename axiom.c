#include "axiom.h"
#include <stdlib.h>

AxiomNet* axiom_create(void) {
    AxiomNet* net = malloc(sizeof(AxiomNet));
    if (net == NULL) return NULL;

    net->layers = NULL;
    net->optimizer = NULL;
    net->num_layers = 0;

    return net;
}

void axiom_free(AxiomNet* net) {
    if (net == NULL) return;

    Layer* current = net->layers; // layers is a linkedlist inside of the axiomnet struct
    while (current != NULL) {
        Layer* next = current->next;

        if (current->type == LAYER_DENSE) {
            dense_free(current->layer.dense);
        } else if (current->type == LAYER_ACTIVATION) {
            activation_free(current->layer.activation);
        }

        free(current);

        current = next;
    }

    if (net->optimizer != NULL) {
        optimizer_free(net->optimizer);
    }

    free(net);
}

void axiom_add(AxiomNet* net, void* layer, int layer_type) {
    // TODO: Implement adding layers to network
}

void axiom_train(AxiomNet* net, Tensor* x_train, Tensor* y_train,
                 size_t epochs, float learning_rate) {
    // TODO: Implement training loop with forward/backward passes
}

Tensor* axiom_forward(AxiomNet* net, const Tensor* input) {
    // TODO: Implement forward pass through entire network
    return NULL;
}

void axiom_save(AxiomNet* net, const char* filename) {
    // TODO: Implement model serialization
}

AxiomNet* axiom_load(const char* filename) {
    // TODO: Implement model loading
    return NULL;
}

DenseLayer* axiom_layer_dense(size_t input_size, size_t output_size) {
    return dense_create(input_size, output_size);
}

Activation* axiom_activation_relu(void) {
    return activation_relu();
}

Activation* axiom_activation_softmax(void) {
    return activation_softmax();
}
