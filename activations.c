#include "activations.h"
#include <stdlib.h>
#include <math.h>

Activation* activation_relu(void) {
    Activation* act = malloc(sizeof(Activation));
    act->type = ACTIVATION_RELU;
    act->input_cache = NULL;  // Will be filled during forward pass
    return act;
}

Activation* activation_softmax(void) {
    Activation* act = malloc(sizeof(Activation));
    act->type = ACTIVATION_SOFTMAX;
    act->input_cache = NULL;  // Will be filled during forward pass
    return act;
}

void activation_free(Activation* act) {
    if (act == NULL) return;

    if (act->input_cache != NULL) {
        tensor_free(act->input_cache);
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

        act->input_cache = tensor_copy(input);

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

        return output;
    }

    // if activation type not known will return null
    return NULL;
}

Tensor* activation_backward(Activation* act, const Tensor* grad_output) {
    // TODO: Implement backward pass for ReLU and Softmax
    return NULL;
}
