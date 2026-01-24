#include "axiom.h"
#include "loss.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define AXIOM_MAGIC "AXIO"

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
    if (net == NULL || filename == NULL) return;

    FILE* f = fopen(filename, "wb");
    if (f == NULL) return;

    fwrite(AXIOM_MAGIC, 1, 4, f);
    uint32_t n = (uint32_t)net->num_layers;
    fwrite(&n, sizeof(uint32_t), 1, f);

    Layer* cur = net->layers; // loop over each layer in order and define the main attributes in binary to the checkpoint file;
    while (cur != NULL) {
        uint8_t layer_type = (cur->type == LAYER_DENSE) ? 0 : 1;
        fwrite(&layer_type, sizeof(uint8_t), 1, f);

        if (cur->type == LAYER_DENSE) {
            DenseLayer* d = cur->layer.dense;
            uint32_t in_sz = (uint32_t)d->input_size;
            uint32_t out_sz = (uint32_t)d->output_size;
            fwrite(&in_sz, sizeof(uint32_t), 1, f);
            fwrite(&out_sz, sizeof(uint32_t), 1, f);
            fwrite(d->weights->data, sizeof(float), (size_t)(in_sz * out_sz), f);
            fwrite(d->biases->data, sizeof(float), (size_t)out_sz, f);
        } else {
            uint8_t act_type = (cur->layer.activation->type == ACTIVATION_RELU) ? 0 : 1;
            fwrite(&act_type, sizeof(uint8_t), 1, f);
        }

        cur = cur->next;
    }

    fclose(f);
}

AxiomNet* axiom_load(const char* filename) {
    if (filename == NULL) return NULL;

    FILE* f = fopen(filename, "rb");
    if (f == NULL) return NULL;

    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, AXIOM_MAGIC, 4) != 0) {
        fclose(f);
        return NULL;
    }

    uint32_t n32;
    if (fread(&n32, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        return NULL;
    }
    size_t num_layers = (size_t)n32;

    AxiomNet* net = axiom_create(); // initialize a net
    if (net == NULL) {
        fclose(f);
        return NULL;
    }

    // loop thru each each saved layer: add it to the net; for dense layers, overwrite weights and biases from the checkpoint file.
    for (size_t i = 0; i < num_layers; i++) {
        uint8_t layer_type;
        if (fread(&layer_type, sizeof(uint8_t), 1, f) != 1) {
            axiom_free(net);
            fclose(f);
            return NULL;
        }

        if (layer_type == 0) {
            uint32_t in_sz, out_sz;
            if (fread(&in_sz, sizeof(uint32_t), 1, f) != 1 ||
                fread(&out_sz, sizeof(uint32_t), 1, f) != 1) {
                axiom_free(net);
                fclose(f);
                return NULL;
            }
            DenseLayer* d = dense_create((size_t)in_sz, (size_t)out_sz);
            if (d == NULL) {
                axiom_free(net);
                fclose(f);
                return NULL;
            }
            size_t nw = (size_t)in_sz * (size_t)out_sz;
            size_t nb = (size_t)out_sz;
            if (fread(d->weights->data, sizeof(float), nw, f) != nw ||
                fread(d->biases->data, sizeof(float), nb, f) != nb) {
                dense_free(d);
                axiom_free(net);
                fclose(f);
                return NULL;
            }
            axiom_add(net, d, LAYER_DENSE);
        } else {
            uint8_t act_type;
            if (fread(&act_type, sizeof(uint8_t), 1, f) != 1) {
                axiom_free(net);
                fclose(f);
                return NULL;
            }
            Activation* act = (act_type == 0) ? activation_relu() : activation_softmax();
            if (act == NULL) {
                axiom_free(net);
                fclose(f);
                return NULL;
            }
            axiom_add(net, act, LAYER_ACTIVATION);
        }
    }

    fclose(f);
    return net;
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
