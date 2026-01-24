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
    if (net == NULL || layer == NULL) return;

    // create new layer node to wrap the layer input
    Layer* new_layer = malloc(sizeof(Layer));
    if (new_layer == NULL) return;

    new_layer->type = layer_type;
    new_layer->next = NULL;

    // hard coded layer switch
    if (layer_type == LAYER_DENSE) {
        new_layer->layer.dense = (DenseLayer*)layer;
    } else if (layer_type == LAYER_ACTIVATION) {
        new_layer->layer.activation = (Activation*)layer;
    }

    // if list empty, add as first layer
    if (net->layers == NULL) {
        net->layers = new_layer;
    } else {
        //find tail and append;
        Layer* current = net->layers;
        while (current->next != NULL) {
            current = current->next;
        }
        current->next = new_layer;
    }

    net->num_layers++;
}

Tensor* axiom_forward(AxiomNet* net, const Tensor* input) {
    if (net == NULL || input == NULL) return NULL;

    Tensor* current_x = tensor_copy(input);  // copying to avoid input modification
    if (current_x == NULL) return NULL;

    Layer* current_layer = net->layers;
    while (current_layer != NULL) {
        Tensor* next_x = NULL;

        if (current_layer->type == LAYER_DENSE) {
            next_x = dense_forward(current_layer->layer.dense, current_x);
        } else if (current_layer->type == LAYER_ACTIVATION) {
            next_x = activation_forward(current_layer->layer.activation, current_x);
        }

        if (next_x == NULL) {
            tensor_free(current_x);
            return NULL;
        }

        tensor_free(current_x);  // free previous output
        current_x = next_x;
        current_layer = current_layer->next;
    }

    return current_x;
}

Tensor* axiom_backward(AxiomNet* net, const Tensor* grad_output, float learning_rate) {
    if (net == NULL || grad_output == NULL) return NULL;

    Tensor* current_grad = tensor_copy(grad_output);  // copying to avoid input modification
    if (current_grad == NULL) return NULL;

    Layer** layers = malloc(net->num_layers * sizeof(Layer*));
    if (layers == NULL) {
        tensor_free(current_grad);
        return NULL;
    }
    Layer* current = net->layers;
    for (size_t i = 0; i < net->num_layers; i++) {
        layers[net->num_layers - 1 - i] = current;  // reverse the list
        current = current->next;
    }


    for (size_t i = 0; i < net->num_layers; i++) {
        Tensor* next_grad = NULL;
        if (layers[i]->type == LAYER_DENSE) {
            next_grad = dense_backward(layers[i]->layer.dense, current_grad, learning_rate);
        } else if (layers[i]->type == LAYER_ACTIVATION) {
            next_grad = activation_backward(layers[i]->layer.activation, current_grad);
        }

        if (next_grad == NULL) {
            tensor_free(current_grad);
            free(layers);
            return NULL;
        }

        Tensor* old_grad = current_grad;
        current_grad = next_grad;
        tensor_free(old_grad);
    }

    free(layers);

    return current_grad;


}

void axiom_train(AxiomNet* net, Tensor* x_train, Tensor* y_train,
    size_t epochs, float learning_rate) {

    if (net == NULL || x_train == NULL || y_train == NULL) return;


    for(size_t epoch = 0; epoch < epochs; epoch++) {
        Tensor* predictions = axiom_forward(net, x_train);
        if (predictions == NULL) continue;

        float loss = loss_cross_entropy(predictions, y_train);
        printf("Epoch %zu: Loss = %f\n", epoch, loss);


        Tensor* grad = loss_cross_entropy_grad(predictions, y_train);
        if (grad == NULL) {
            tensor_free(predictions);
            continue;
        }

        Tensor* grad_outputs = axiom_backward(net, grad, learning_rate);


        tensor_free(predictions);
        tensor_free(grad);
        tensor_free(grad_outputs);
    }
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
