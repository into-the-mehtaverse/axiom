#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "axiom.h"
#include "mnist.h"

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

    /* Less trivial data: 4 examples, 4 features -> 2 classes */
    size_t x_shape[] = {4, 4};
    size_t y_shape[] = {4, 2};
    Tensor* x_train = tensor_create(x_shape, 2);
    Tensor* y_train = tensor_create(y_shape, 2);
    if (!x_train || !y_train) {
        printf("FAIL: tensor_create\n");
        if (x_train) tensor_free(x_train);
        if (y_train) tensor_free(y_train);
        axiom_free(net);
        return;
    }

    /* Random-ish inputs (seed 42), one-hot targets: ex0,1 -> class 0; ex2,3 -> class 1 */
    tensor_rand(x_train, -0.5f, 0.5f, 42);
    tensor_fill(y_train, 0.0f);
    y_train->data[0] = 1.0f;
    y_train->data[1] = 0.0f;
    y_train->data[2] = 1.0f;
    y_train->data[3] = 0.0f;
    y_train->data[4] = 0.0f;
    y_train->data[5] = 1.0f;
    y_train->data[6] = 0.0f;
    y_train->data[7] = 1.0f;

    printf("Training 25 epochs, lr=0.05 ...\n");
    axiom_train(net, x_train, y_train, 25, 0.05f);

    Tensor* out_orig = axiom_forward(net, x_train);
    if (!out_orig) {
        printf("FAIL: axiom_forward (original)\n");
        tensor_free(x_train);
        tensor_free(y_train);
        axiom_free(net);
        return;
    }
    printf("Predictions shape [%zu, %zu]\n", out_orig->shape[0], out_orig->shape[1]);
    for (size_t i = 0; i < 4; i++) {
        printf("  ex%zu: [%.4f, %.4f]\n", i,
               out_orig->data[i * 2], out_orig->data[i * 2 + 1]);
    }

    /* Verify save/load: save net, free, load, forward again; predictions must match. */
    const char* ckpt = "build/smoke_checkpoint.bin";
    printf("Verifying save/load ...\n");
    axiom_save(net, ckpt);
    axiom_free(net);
    net = axiom_load(ckpt);
    if (!net) {
        printf("FAIL: axiom_load\n");
        tensor_free(out_orig);
        tensor_free(x_train);
        tensor_free(y_train);
        return;
    }
    Tensor* out_loaded = axiom_forward(net, x_train);
    if (!out_loaded) {
        printf("FAIL: axiom_forward (after load)\n");
        tensor_free(out_orig);
        tensor_free(x_train);
        tensor_free(y_train);
        axiom_free(net);
        return;
    }
    if (out_orig->size != out_loaded->size) {
        printf("FAIL: save/load (output size mismatch %zu vs %zu)\n", out_orig->size, out_loaded->size);
        tensor_free(out_orig);
        tensor_free(out_loaded);
        tensor_free(x_train);
        tensor_free(y_train);
        axiom_free(net);
        return;
    }
    for (size_t i = 0; i < out_orig->size; i++) {
        if (out_orig->data[i] != out_loaded->data[i]) {
            printf("FAIL: save/load (predictions differ at index %zu: %.6f vs %.6f)\n",
                   i, out_orig->data[i], out_loaded->data[i]);
            tensor_free(out_orig);
            tensor_free(out_loaded);
            tensor_free(x_train);
            tensor_free(y_train);
            axiom_free(net);
            return;
        }
    }
    printf("PASS: save/load (predictions match)\n");
    tensor_free(out_orig);
    tensor_free(out_loaded);
    tensor_free(x_train);
    tensor_free(y_train);
    axiom_free(net);
    printf("=== Done ===\n");
}

static void run_mnist_load(void) {
    printf("=== MNIST loader smoke test ===\n");
    const char* base = "data/MNIST";
    Tensor *x_train = NULL, *y_train = NULL, *x_test = NULL, *y_test = NULL;
    if (mnist_load(base, &x_train, &y_train, &x_test, &y_test) != 0) {
        printf("FAIL: mnist_load(\"%s\")\n", base);
        return;
    }
    printf("x_train [%zu, %zu]\n", x_train->shape[0], x_train->shape[1]);
    printf("y_train [%zu, %zu]\n", y_train->shape[0], y_train->shape[1]);
    printf("x_test  [%zu, %zu]\n", x_test->shape[0], x_test->shape[1]);
    printf("y_test  [%zu, %zu]\n", y_test->shape[0], y_test->shape[1]);
    /* First 5 train labels (argmax of one-hot) */
    printf("First 5 train labels: ");
    for (size_t i = 0; i < 5; i++) {
        size_t argmax = 0;
        for (size_t k = 1; k < 10; k++)
            if (y_train->data[i * 10 + k] > y_train->data[i * 10 + argmax]) argmax = k;
        printf("%zu ", argmax);
    }
    printf("\n=== Done ===\n");
    tensor_free(x_train);
    tensor_free(y_train);
    tensor_free(x_test);
    tensor_free(y_test);
}

int main(int argc, char* argv[]) {
    if (argc >= 2 && strcmp(argv[1], "test") == 0) {
        run_test();
        return 0;
    }
    if (argc >= 2 && strcmp(argv[1], "mnist") == 0) {
        run_mnist_load();
        return 0;
    }

    if (argc < 2) {
        printf("Usage: %s <command> [options]\n", argv[0]);
        printf("Commands:\n");
        printf("  test                           Run smoke test\n");
        printf("  mnist                          Smoke-test MNIST loader\n");
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
