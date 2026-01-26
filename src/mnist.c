#include "mnist.h"
#include "tensor.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MNIST_IMAGE_MAGIC 0x00000803u
#define MNIST_LABEL_MAGIC 0x00000801u
#define IMG_ROWS 28
#define IMG_COLS 28
#define IMG_SIZE (IMG_ROWS * IMG_COLS)
#define NUM_CLASSES 10
#define PATH_MAX 256

static uint32_t read_be32(FILE* f) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) return 0;
    return (uint32_t)b[0] << 24 | (uint32_t)b[1] << 16 |
           (uint32_t)b[2] << 8  | (uint32_t)b[3];
}

static int build_path(char* buf, size_t cap, const char* base, const char* name) {
    int n = snprintf(buf, cap, "%s/%s", base, name);
    return (n > 0 && (size_t)n < cap) ? 0 : -1;
}

/**
 * Load IDX3 images -> [N, 784] float, normalized to [0,1].
 * Caller must tensor_free result.
 */
static Tensor* load_images(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    uint32_t magic = read_be32(f);
    if (magic != MNIST_IMAGE_MAGIC) {
        fclose(f);
        return NULL;
    }
    uint32_t num = read_be32(f);
    uint32_t rows = read_be32(f);
    uint32_t cols = read_be32(f);
    if (rows != IMG_ROWS || cols != IMG_COLS) {
        fclose(f);
        return NULL;
    }

    size_t n = (size_t)num;
    size_t shape[] = { n, IMG_SIZE };
    Tensor* x = tensor_create(shape, 2);
    if (!x) {
        fclose(f);
        return NULL;
    }

    size_t total = n * IMG_SIZE;
    unsigned char* raw = malloc(total);
    if (!raw) {
        tensor_free(x);
        fclose(f);
        return NULL;
    }
    if (fread(raw, 1, total, f) != total) {
        free(raw);
        tensor_free(x);
        fclose(f);
        return NULL;
    }
    fclose(f);

    for (size_t i = 0; i < total; i++)
        x->data[i] = (float)raw[i] / 255.0f;
    free(raw);
    return x;
}

/**
 * Load IDX1 labels -> [N, 10] one-hot float.
 * Caller must tensor_free result.
 */
static Tensor* load_labels_onehot(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    uint32_t magic = read_be32(f);
    if (magic != MNIST_LABEL_MAGIC) {
        fclose(f);
        return NULL;
    }
    uint32_t num = read_be32(f);
    size_t n = (size_t)num;

    unsigned char* raw = malloc(n);
    if (!raw) {
        fclose(f);
        return NULL;
    }
    if (fread(raw, 1, n, f) != n) {
        free(raw);
        fclose(f);
        return NULL;
    }
    fclose(f);

    size_t shape[] = { n, NUM_CLASSES };
    Tensor* y = tensor_create(shape, 2);
    if (!y) {
        free(raw);
        return NULL;
    }
    tensor_fill(y, 0.0f);
    for (size_t i = 0; i < n; i++) {
        unsigned char lab = raw[i];
        if (lab < NUM_CLASSES)
            y->data[i * NUM_CLASSES + (size_t)lab] = 1.0f;
    }
    free(raw);
    return y;
}

int mnist_load(const char* base_path,
               Tensor** out_x_train, Tensor** out_y_train,
               Tensor** out_x_test, Tensor** out_y_test) {
    if (!base_path || !out_x_train || !out_y_train || !out_x_test || !out_y_test) {
        return -1;
    }
    *out_x_train = NULL;
    *out_y_train = NULL;
    *out_x_test = NULL;
    *out_y_test = NULL;

    char path[PATH_MAX];
    Tensor* x_train = NULL;
    Tensor* y_train = NULL;
    Tensor* x_test = NULL;
    Tensor* y_test = NULL;

    if (build_path(path, sizeof path, base_path, "train-images.idx3-ubyte") != 0) return -1;
    x_train = load_images(path);
    if (!x_train) return -1;

    if (build_path(path, sizeof path, base_path, "train-labels.idx1-ubyte") != 0) {
        tensor_free(x_train);
        return -1;
    }
    y_train = load_labels_onehot(path);
    if (!y_train) {
        tensor_free(x_train);
        return -1;
    }

    if (build_path(path, sizeof path, base_path, "t10k-images-idx3-ubyte") != 0) {
        tensor_free(x_train);
        tensor_free(y_train);
        return -1;
    }
    x_test = load_images(path);
    if (!x_test) {
        tensor_free(x_train);
        tensor_free(y_train);
        return -1;
    }

    if (build_path(path, sizeof path, base_path, "t10k-labels-idx1-ubyte") != 0) {
        tensor_free(x_train);
        tensor_free(y_train);
        tensor_free(x_test);
        return -1;
    }
    y_test = load_labels_onehot(path);
    if (!y_test) {
        tensor_free(x_train);
        tensor_free(y_train);
        tensor_free(x_test);
        return -1;
    }

    *out_x_train = x_train;
    *out_y_train = y_train;
    *out_x_test = x_test;
    *out_y_test = y_test;
    return 0;
}
