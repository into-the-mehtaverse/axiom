#include "tensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

Tensor* tensor_create(size_t* shape, size_t ndim) {
    // i include error handling here, freeing previous allocations if error occurs at any step
    // allocate tensor struct
    Tensor* tensor = malloc(sizeof(Tensor));
    if (tensor == NULL) return NULL;

    // calculate total size
    size_t total_size = 1;
    for (int i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }
    // allocate data array
    tensor->data = malloc(total_size * sizeof(float));
    if (tensor->data == NULL) { free(tensor); return NULL; }

    // allocate and copy shape
    tensor->shape = malloc(ndim * sizeof(size_t));

    if (tensor->shape == NULL) {
        free(tensor->data);
        free(tensor);
        return NULL;
    }

    for (int i = 0; i < ndim; i++) {
        tensor->shape[i] = shape[i];
    }

    // allocate strides array
    tensor->strides = malloc(ndim * sizeof(size_t));
    if (tensor->strides == NULL) {
        free(tensor->shape);
        free(tensor->data);
        free(tensor);
        return NULL;
    }

    // calculate strides
    tensor->strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        tensor->strides[i] = tensor->strides[i + 1] * shape[i + 1];
    }

    // initialize fields
    tensor->ndim = ndim;
    tensor->size = total_size;

    return tensor;
}

void tensor_free(Tensor* t) {
    if (t == NULL) return;

    // only freeing these here bc they were created w malloc, ndim and size aren't pointers;
    free(t->strides);
    free(t->shape);
    free(t->data);
    free(t);
}

Tensor* tensor_copy(const Tensor* t) {
    if (t == NULL) return NULL;

    // create blank copy with same shape
    Tensor* copy = tensor_create(t->shape, t->ndim);
    if (copy == NULL) return NULL;

    // copy data
    for (int i = 0; i < t->size; i++) {
        copy->data[i] = t->data[i];
    }

    return copy;
}

Tensor* tensor_matmul(const Tensor* a, const Tensor* b) {
    if (a == NULL || b == NULL) return NULL;

    // define parameters
    int m = a->shape[0];
    int n = a->shape[1];
    int p = b->shape[1];

    if (n != b->shape[0]) return NULL;

    // allocate result
    size_t shape[] = {m, p};
    Tensor* result = tensor_create(shape, 2);
    if (result == NULL) return NULL;

    // matmul
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            result->data[i * result->strides[0] + j * result->strides[1]] = 0.0f; // adding f so its a float literal and not implicity coverted from double; hygiene
        } // malloc doesnt zero intialize values so i must reset to results to 0 otherwise we'd be running matmul on garbage values

        for(int k = 0; k < n; k++) {
            float a_ik = a->data[i * a->strides[0] + k * a->strides[1]];
            for(int j = 0; j < p; j++) {
                result->data[i * result->strides[0] + j * result->strides[1]] += a_ik * b->data[k * b->strides[0] + j * b->strides[1]];
            }
        }
    }

    return result;
}

Tensor* tensor_add(const Tensor* a, const Tensor* b) {
    if (a == NULL || b == NULL) return NULL;
    if (a->ndim != b->ndim) return NULL;

    // ensure shapes match
    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) return NULL;
    }

    Tensor* result = tensor_create(a->shape, a->ndim);
    if (result == NULL) return NULL;

    for (int i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }

    return result;
}

Tensor* tensor_transpose(const Tensor* t) {
    if (t == NULL) return NULL;
    if (t->ndim != 2) return NULL;


    // create tp transpose result;
    size_t tp_shape[] = {t->shape[1], t->shape[0]};
    Tensor* tp = tensor_create(tp_shape, 2);
    if (tp == NULL) return NULL;

    for (int i = 0; i < t->shape[0]; i++) {
        for (int j = 0; j < t->shape[1]; j++) {
            tp->data[j * tp->strides[0] + i * tp->strides[1]] = t->data[i * t->strides[0] + j * t->strides[1]];
        }
    }

    return tp;
}

Tensor* tensor_broadcast(const Tensor* t, size_t* new_shape, size_t new_ndim) {
    if (t == NULL || new_shape == NULL) return NULL;
    if (new_ndim < t->ndim) return NULL;  // can't be broadcasting to smaller dimensions


    for (int i = 0; i < t->ndim; i++) {
        size_t orig_dim = t->shape[t->ndim - 1 - i];
        size_t new_dim = new_shape[new_ndim - 1 - i];
        if (orig_dim != 1 && orig_dim != new_dim) {
            return NULL;  // incompatible; either they have to be same, or one of htem has to be 1
        }
    }

    Tensor* result = tensor_create(new_shape, new_ndim);
    if (result == NULL) return NULL;

    // allocate array to hold multi-dimensional indices
    size_t* result_indices = malloc(result->ndim * sizeof(size_t));
    if (result_indices == NULL) {
        tensor_free(result);
        return NULL;
    }

    size_t* orig_indices = malloc(t->ndim * sizeof(size_t));
    if (orig_indices == NULL) {
        free(result_indices);
        tensor_free(result);
        return NULL;
    }

    // iterate through each position in result
    for (size_t flat_idx = 0; flat_idx < result->size; flat_idx++) {
        // convert flat index to multi-dimensional indices in result
        size_t remaining = flat_idx;
        for (int dim = result->ndim - 1; dim >= 0; dim--) {
            result_indices[dim] = remaining / result->strides[dim];
            remaining = remaining % result->strides[dim];
        }

        for (int dim = 0; dim < t->ndim; dim++) {
            int result_dim = result->ndim - t->ndim + dim;
            if (t->shape[dim] == 1) {
                orig_indices[dim] = 0;  // broadcast dimension of size1
            } else {
                orig_indices[dim] = result_indices[result_dim];
            }
        }

        // calculate flat index in original tensor
        size_t orig_flat_idx = 0;
        for (int dim = 0; dim < t->ndim; dim++) {
            orig_flat_idx += orig_indices[dim] * t->strides[dim];
        }

        // copy value from original to result
        result->data[flat_idx] = t->data[orig_flat_idx];
    }

    free(result_indices);
    free(orig_indices);

    return result;
}

Tensor* tensor_apply(const Tensor* t, float (*func)(float)) {
    // TODO: Implement element-wise function application
    return NULL;
}

void tensor_fill(Tensor* t, float value) {
    // TODO: Implement tensor filling
}

void tensor_rand(Tensor* t, float min, float max) {
    // TODO: Implement random initialization
}
