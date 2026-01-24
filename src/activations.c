#include "activations.h"
#include <stdlib.h>
#include <math.h>

Activation* activation_relu(void) {
    Activation* act = malloc(sizeof(Activation));
    act->type = ACTIVATION_RELU;
    act->input_cache = NULL;  // Will be filled during forward pass
    act->output_cache = NULL;
    return act;
}

Activation* activation_softmax(void) {
    Activation* act = malloc(sizeof(Activation));
    act->type = ACTIVATION_SOFTMAX;
    act->input_cache = NULL;  // Will be filled during forward pass
    act->output_cache = NULL;
    return act;
}

void activation_free(Activation* act) {
    if (act == NULL) return;

    if (act->input_cache != NULL) {
        tensor_free(act->input_cache);
    }

    if (act->output_cache != NULL) {
        tensor_free(act->output_cache);
    }

    free(act);
}

Tensor* activation_forward(Activation* act, const Tensor* input) {
    if (act == NULL || input == NULL) return NULL;

    if (act->type == ACTIVATION_RELU) {

        //output tensor
        Tensor* output = tensor_create(input->shape, input->ndim);

        for (size_t i = 0; i < input->size; i++) {
            output->data[i] = (input->data[i] > 0) ? input->data[i] : 0.0f;
        }

        if (act->input_cache != NULL) {
            tensor_free(act->input_cache);
        }

        if (act->output_cache != NULL) {
            tensor_free(act->output_cache);
        }

        act->input_cache = tensor_copy(input);
        act->output_cache = tensor_copy(output);

        return output;
    }

    if (act->type == ACTIVATION_SOFTMAX) {

        //output tensor
        Tensor* output = tensor_create(input->shape, input->ndim);

        // 2d tensor only for now ; can extend to be more flexible later by calc num dims then doing this for each group
        size_t batch_size = input->shape[0];
        size_t num_classes = input->shape[1];

        // go thru batch
        for (size_t i = 0; i < batch_size; i++) {
            // find max in current example
            float max_val = input->data[i * input->strides[0] + 0 * input->strides[1]];
            for (size_t j = 1; j < num_classes; j++) {
                float val = input->data[i * input->strides[0] + j * input->strides[1]];
                if (val > max_val) max_val = val;
            }

            // compute exp(x - max) and sum
            float sum = 0.0f;
            for (size_t j = 0; j < num_classes; j++) {
                size_t idx = i * input->strides[0] + j * input->strides[1];
                float exp_val = expf(input->data[idx] - max_val);
                output->data[idx] = exp_val;
                sum += exp_val;
            }

            // normalize by sum
            for (size_t j = 0; j < num_classes; j++) {
                size_t idx = i * input->strides[0] + j * input->strides[1];
                output->data[idx] /= sum;
            }
        }

        act->input_cache = tensor_copy(input);
        act->output_cache = tensor_copy(output);

        return output;
    }

    // if activation type not known will return null
    return NULL;
}

Tensor* activation_backward(Activation* act, const Tensor* grad_output) {

    // validate
    if (act == NULL || grad_output == NULL) return NULL;

    if (act->input_cache == NULL || act->output_cache == NULL) return NULL;

    if (grad_output->ndim != act->output_cache->ndim) return NULL;
    for (size_t i = 0; i < grad_output->ndim; i++) {
        if (grad_output->shape[i] != act->output_cache->shape[i]) return NULL;
    }

    // switch
    if (act->type == ACTIVATION_RELU) {
        Tensor* grad_input = tensor_create(grad_output->shape, grad_output->ndim);
        if (grad_input == NULL) return NULL;

        // gradient passes through where input > 0, everything else 0
        for (size_t i = 0; i < grad_output->size; i++) {
            grad_input->data[i] = (act->input_cache->data[i] > 0.0f) ? grad_output->data[i] : 0.0f;
        }

        return grad_input;
    }

    if (act->type == ACTIVATION_SOFTMAX) {
        Tensor* grad_input = tensor_create(grad_output->shape, grad_output->ndim);
        if (grad_input == NULL) return NULL;

        size_t batch_size = grad_output->shape[0];
        size_t num_classes = grad_output->shape[1];

        // go thru each example
        for (size_t i = 0; i < batch_size; i++) {

            // get dot of output cache and last calculated layer's grad output
            float dot = 0.0f;
            for (size_t j = 0; j < num_classes; j++) {
                size_t idx = i * grad_output->strides[0] + j * grad_output->strides[1];
                dot += grad_output->data[idx] * act->output_cache->data[idx]; //softmax outputs are coupled - this is captures that relationship. softmax outputs sum to one, so changing one input affects all of them.
            }

            for (size_t j = 0; j < num_classes; j++) {
                size_t idx = i * grad_output->strides[0] + j * grad_output->strides[1];
                grad_input->data[idx] = act->output_cache->data[idx] * (grad_output->data[idx] - dot);  //this scales the gradient by the probability (how much this input contributed to the output)
            }
        }
        return grad_input;
    }
    return NULL;
}
