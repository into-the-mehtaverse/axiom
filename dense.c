#include "dense.h"
#include <stdlib.h>

DenseLayer* dense_create(size_t input_size, size_t output_size) {
    DenseLayer* dense = malloc(sizeof(DenseLayer));
    if (dense == NULL) return NULL;

    // allocate tensors

    size_t weights_shape[] = {input_size, output_size};
    dense->weights = tensor_create(weights_shape, 2);
    if (dense->weights == NULL) {
        free(dense);
        return NULL;
    }

    // only output size for bias
    size_t biases_shape[] = {output_size};
    dense->biases = tensor_create(biases_shape, 1);
    if (dense->biases == NULL) {
        tensor_free(dense->weights);
        free(dense);
        return NULL;
    }

    tensor_rand(dense->weights, -0.1f, 0.1f, 42);
    tensor_fill(dense->biases, 0.0f);

    dense->input_cache = NULL;  // will be filled during forward pass
    dense->input_size = input_size;
    dense->output_size = output_size;

    return dense;
}

void dense_free(DenseLayer* layer) {
    if (layer == NULL) return;

    if (layer->weights != NULL) {
        tensor_free(layer->weights);
    }

    if (layer->biases != NULL) {
        tensor_free(layer->biases);
    }

    if (layer->input_cache != NULL) {
        tensor_free(layer->input_cache);
    }

    free(layer);
}

Tensor* dense_forward(DenseLayer* layer, const Tensor* input) {
    // TODO: Implement forward pass: output = input @ weights + biases
    return NULL;
}

Tensor* dense_backward(DenseLayer* layer, const Tensor* grad_output, float learning_rate) {
    // TODO: Implement backward pass with gradient computation and weight updates
    return NULL;
}
