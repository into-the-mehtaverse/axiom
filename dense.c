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
    if (layer == NULL || input == NULL) return NULL;

    if (input->ndim != 2) return NULL;
    if (input->shape[1] != layer->input_size) return NULL;

    Tensor* output = tensor_matmul(input, layer->weights);
    if (output == NULL) return NULL;

    Tensor* b = tensor_broadcast(layer->biases, output->shape, output->ndim);

    Tensor* result = tensor_add(output, b);
    if (result == NULL) {
        tensor_free(output);
        tensor_free(b);
        return NULL;
    }
    tensor_free(output); // if i assign the output of above to this same variable, i'll get a memory leak and the old one will be lost;
    tensor_free(b);

    if (layer->input_cache != NULL) {
        tensor_free(layer->input_cache);
    }
    layer->input_cache = tensor_copy(input);

    return result;
}

Tensor* dense_backward(DenseLayer* layer, const Tensor* grad_output, float learning_rate) {
    if (layer == NULL || grad_output == NULL) return NULL;
    if (grad_output->ndim != 2) return NULL;
    if (grad_output->shape[1] != layer->output_size) return NULL;

    if (layer->input_cache == NULL) return NULL;

    Tensor* input_transposed = tensor_transpose(layer->input_cache);

    // compute gradients for weights
    Tensor* grad_weights = tensor_matmul(input_transposed, grad_output);
    tensor_free(input_transposed);

    // sum bias over batch (axis 0). bias is shared across batch.
    // ex. if grad_output is [[0.1, 0.2], [0.3, 0.4]] then grad_biases would be [0.4, 0.6];
    size_t biases_shape[] = {layer->output_size};
    Tensor* grad_biases = tensor_create(biases_shape, 1);
        if (grad_biases == NULL) {
        tensor_free(grad_weights);
        return NULL;
        }


    for (size_t j = 0; j < layer->output_size; j++) {
        float sum = 0.0f;
        for (size_t i = 0; i < grad_output->shape[0]; i++) {
            size_t idx = i * grad_output->strides[0] + j * grad_output->strides[1];
            sum += grad_output->data[idx];
        }
        grad_biases->data[j] = sum;
    }

    // update weights and biases with learning rate via gradient descent
    for(size_t i = 0; i < layer->weights->size; i++) {
        layer->weights->data[i] -= learning_rate * grad_weights->data[i];
    }
    for(size_t i = 0; i < layer->biases->size; i++) {
        layer->biases->data[i] -= learning_rate * grad_biases->data[i];
    }

    // free the temp tensors
    tensor_free(grad_weights);
    tensor_free(grad_biases);

    // compute the gradient for input into next layer in the backprop order (the previous layer)
    Tensor* weights_transposed = tensor_transpose(layer->weights);
    if(weights_transposed == NULL) {
        return NULL;
    }

    Tensor* grad_input = tensor_matmul(grad_output, weights_transposed);
    tensor_free(weights_transposed);
    if (grad_input == NULL) return NULL;


    return grad_input;
}
