#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

typedef struct {
    float* data;
    size_t* shape;
    size_t* strides;
    size_t ndim;
    size_t size;
} Tensor;

// Tensor creation and memory management
Tensor* tensor_create(size_t* shape, size_t ndim);
void tensor_free(Tensor* t);
Tensor* tensor_copy(const Tensor* t);

// Matrix operations
Tensor* tensor_matmul(const Tensor* a, const Tensor* b);
Tensor* tensor_add(const Tensor* a, const Tensor* b);
Tensor* tensor_transpose(const Tensor* t);
Tensor* tensor_broadcast(const Tensor* t, size_t* new_shape, size_t new_ndim);

// Element-wise operations
Tensor* tensor_apply(const Tensor* t, float (*func)(float));
void tensor_fill(Tensor* t, float value);
void tensor_rand(Tensor* t, float min, float max, unsigned int seed);

#endif // TENSOR_H
