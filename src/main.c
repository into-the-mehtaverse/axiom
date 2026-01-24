#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "axiom.h"

static void run_test(void) {
    printf("=== Axiom smoke test ===\n");

    /* Tiny network: 4 -> 4 (ReLU) -> 2 (Softmax) */
    AxiomNet* net = axiom_create();
    if (!net) {
        printf("FAIL: axiom_create\n");
        return;
    }

    axiom_add(net, axiom_layer_dense(4, 4), LAYER_DENSE);
    axiom_add(net, axiom_activation_relu(), LAYER_ACTIVATION);
    axiom_add(net, axiom_layer_dense(4, 2), LAYER_DENSE);
    axiom_add(net, axiom_activation_softmax(), LAYER_ACTIVATION);

    /* Dummy data: 2 examples, 4 features -> 2 classes */
    size_t x_shape[] = {2, 4};
    size_t y_shape[] = {2, 2};
    Tensor* x_train = tensor_create(x_shape, 2);
    Tensor* y_train = tensor_create(y_shape, 2);
    if (!x_train || !y_train) {
        printf("FAIL: tensor_create\n");
        if (x_train) tensor_free(x_train);
        if (y_train) tensor_free(y_train);
        axiom_free(net);
        return;
    }

    tensor_fill(x_train, 0.5f);
    tensor_fill(y_train, 0.0f);
    y_train->data[0] = 1.0f; /* one-hot [1,0] */
    y_train->data[3] = 1.0f; /* one-hot [0,1] */

    printf("Training 5 epochs...\n");
    axiom_train(net, x_train, y_train, 5, 0.01f);

    Tensor* out = axiom_forward(net, x_train);
    if (out) {
        printf("Predictions shape [%zu, %zu]\n", out->shape[0], out->shape[1]);
        printf("  ex0: [%.4f, %.4f]\n", out->data[0], out->data[1]);
        printf("  ex1: [%.4f, %.4f]\n", out->data[2], out->data[3]);
        tensor_free(out);
    }

    tensor_free(x_train);
    tensor_free(y_train);
    axiom_free(net);
    printf("=== Done ===\n");
}

int main(int argc, char* argv[]) {
    if (argc >= 2 && strcmp(argv[1], "test") == 0) {
        run_test();
        return 0;
    }

    if (argc < 2) {
        printf("Usage: %s <command> [options]\n", argv[0]);
        printf("Commands:\n");
        printf("  test                           Run smoke test\n");
        printf("  train --epochs <n> --lr <rate> Train the model\n");
        printf("  predict <model_file> <input>   Run inference\n");
        return 1;
    }

    if (strcmp(argv[1], "train") == 0) {
        printf("Training not yet implemented (use 'test' for now)\n");
    } else if (strcmp(argv[1], "predict") == 0) {
        printf("Inference not yet implemented\n");
    } else {
        printf("Unknown command: %s\n", argv[1]);
        return 1;
    }

    return 0;
}
