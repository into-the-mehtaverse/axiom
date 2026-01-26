#ifndef MNIST_H
#define MNIST_H

#include "tensor.h"

/**
 * Load MNIST train and test data from IDX files under base_path.
 *
 * base_path should be the directory containing:
 *   train-images.idx3-ubyte, train-labels.idx1-ubyte
 *   t10k-images-idx3-ubyte,  t10k-labels-idx1-ubyte
 *
 * On success, allocates four tensors and sets *out_x_train, *out_y_train,
 * *out_x_test, *out_y_test. Caller must tensor_free each.
 *
 * - x_train: [60000, 784] float, pixels normalized to [0, 1]
 * - y_train: [60000, 10] float, one-hot
 * - x_test:  [10000, 784] float
 * - y_test:  [10000, 10] float, one-hot
 *
 * Returns 0 on success, -1 on error (file open/read or invalid format).
 */
int mnist_load(const char* base_path,
               Tensor** out_x_train, Tensor** out_y_train,
               Tensor** out_x_test, Tensor** out_y_test);

#endif /* MNIST_H */
