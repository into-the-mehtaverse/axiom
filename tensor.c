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
    for (size_t i = 0; i < ndim; i++) {
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
    for (size_t i = ndim - 2; i >= 0; i--) {
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
    for (size_t i = 0; i < t->size; i++) {
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
    for (size_t i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) return NULL;
    }

    Tensor* result = tensor_create(a->shape, a->ndim);
    if (result == NULL) return NULL;

    for (size_t i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }

    return result;
}

Tensor* tensor_subtract(const Tensor* a, const Tensor* b) {
    // this is gonna do a - b
    if (a == NULL || b == NULL) return NULL;
    if (a->ndim != b->ndim) return NULL;

    // ensure shapes match
    for (size_t i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) return NULL;
    }

    Tensor* result = tensor_create(a->shape, a->ndim);
    if (result == NULL) return NULL;

    for (size_t i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] - b->data[i];
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
    // for now doing this version with a copy into result tensor, i know its less efficient but for sake of moving forward faster just implmeneting this here now

    if (t == NULL || new_shape == NULL) return NULL;
    if (new_ndim < t->ndim) return NULL;

    // check broadcasting compatibility
    size_t offset = new_ndim - t->ndim;
    for (size_t d = 0; d < t->ndim; d++) {
        size_t orig_dim = t->shape[d];
        size_t new_dim = new_shape[offset + d];
        if (orig_dim != 1 && orig_dim != new_dim) {
            return NULL;  // incompatible; either dims are same or one has to be 1
        }
    }

    // create result tensor
    Tensor* result = tensor_create(new_shape, new_ndim);
    if (result == NULL) return NULL;

    // for each position in result, find corresponding position in original
    for (size_t flat_idx = 0; flat_idx < result->size; flat_idx++) {
        size_t remaining = flat_idx;
        size_t orig_flat = 0;

        // convert flat index to coordinates and map to original
        for (size_t dim = 0; dim < new_ndim; dim++) {
            size_t idx = remaining / result->strides[dim];
            remaining = remaining % result->strides[dim];

            // map to original tensor for dimensions that exist in original
            if (dim >= offset) {
                size_t orig_dim_idx = dim - offset;
                if (t->shape[orig_dim_idx] != 1) {
                    orig_flat += idx * t->strides[orig_dim_idx];
                }
                // if original dimension is size 1, use index 0 (broadcasting)
            }
        }

        result->data[flat_idx] = t->data[orig_flat];
    }

    return result;
}

Tensor* tensor_apply(const Tensor* t, float (*func)(float)) {
    if (t == NULL || func == NULL) return NULL;

    Tensor* result = tensor_create(t->shape, t->ndim);
    if (result == NULL) return NULL;

    for (size_t i = 0; i < t->size; i++) {
        result->data[i] = func(t->data[i]);
    }
    return result;
}

void tensor_fill(Tensor* t, float value) {
    if (t == NULL) return;

    for (size_t i = 0; i < t->size; i++) {
        t->data[i] = value;
    }
}

void tensor_rand(Tensor* t, float min, float max, unsigned int seed) {
    if (t == NULL) return;

    srand(seed); // this updates the global random state, so all subsequent calls will be impacted. can add a local rng state later


    float range = max - min;
    for (size_t i = 0; i < t->size; i++) {
        float random = (float)rand() / (float)RAND_MAX; // get it between 0.0 and 1.0
        t->data[i] = min + random * range;
    }
}
