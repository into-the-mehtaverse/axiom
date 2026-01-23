#ifndef AXIOM_H
#define AXIOM_H

#include "tensor.h"
#include "dense.h"
#include "activations.h"
#include "optimizer.h"

typedef struct Layer {
    enum {
        LAYER_DENSE,
        LAYER_ACTIVATION
    } type;
    union {
        DenseLayer* dense;
        Activation* activation;
    } layer;
    struct Layer* next;
} Layer;

typedef struct {
    Layer* layers;
    Optimizer* optimizer;
    size_t num_layers;
} AxiomNet;

// Network creation and management
AxiomNet* axiom_create(void);
void axiom_free(AxiomNet* net);

// Add layers to network
void axiom_add(AxiomNet* net, void* layer, int layer_type);

// Training
void axiom_train(AxiomNet* net, Tensor* x_train, Tensor* y_train,
                 size_t epochs, float learning_rate);

// Inference
Tensor* axiom_forward(AxiomNet* net, const Tensor* input);

// Model serialization
void axiom_save(AxiomNet* net, const char* filename);
AxiomNet* axiom_load(const char* filename);

// Convenience functions for creating layers
DenseLayer* axiom_layer_dense(size_t input_size, size_t output_size);
Activation* axiom_activation_relu(void);
Activation* axiom_activation_softmax(void);

#endif // AXIOM_H
