# I Wrote a Neural Network Library in Pure C (And You Should Too)

Axiom is a neural network library I wrote in pure C that achieves **96.5% accuracy on MNIST**—with zero external dependencies. It's just ~1,000 lines of C and the standard library.

I used an LLM to scaffold boilerplate (function definitions, etc, directed by me), but every line of logic is mine. The matrix multiplications, the backpropagation, the memory management are all handwritten.

Why? Because modern ML frameworks are black boxes, and it was time for me to go deeper. I began learning ML 100 days ago (I've been documenting my progress daily on my X page here: link). In the first ~92 days, I completed Andrew Ng's DeepLearning specialization & all related projects, built a segmentation studio using the SAM model, finished three kaggle comps, and wrote the LSTM forward pass in python and pure c without imports. Over the last 8 days, I built Axiom to cement my knowledge and solidify my mastery over the fundamentals. I wasn't satisfied with using methods like Pytorch's "loss.backward()" and training with tensors without knowing what's behind the hood; abstraction removes boilerplate at the expense of learning. The only way to truly know is to build it yourself.

Quick API usage example:
```c
// Define network: 784 → 128 (ReLU) → 10 (Softmax)
AxiomNet* net = axiom_create();
axiom_add(net, axiom_layer_dense(784, 128), LAYER_DENSE);
axiom_add(net, axiom_activation_relu(), LAYER_ACTIVATION);
axiom_add(net, axiom_layer_dense(128, 10), LAYER_DENSE);
axiom_add(net, axiom_activation_softmax(), LAYER_ACTIVATION);

// Train and save
axiom_train(net, x_train, y_train, epochs, learning_rate, batch_size);
axiom_save(net, "model.bin");
```
If you want to run it with MNIST, add a data folder to the root, and within an MNIST subfolder, add the four MNIST files.

---

## Architecture

Axiom is built in layers of abstraction, each depending only on what's below it:

```
┌─────────────────────────────────────┐
│            AxiomNet API             │  ← Network orchestration, training loop
├─────────────────────────────────────┤
│   Dense │ Activations │ Optimizer   │  ← Layers with forward/backward passes
├─────────────────────────────────────┤
│         Loss Functions              │  ← Cross-entropy, MSE + gradients
├─────────────────────────────────────┤
│           Tensor Engine             │  ← Data, shapes, strides, matmul
└─────────────────────────────────────┘
```

The network itself is a linked list of layers. Each layer stores its own weights, gradients, and cached inputs (for backprop). Forward pass walks the list head-to-tail; backward pass reverses it.

I purposely built this library modularly so that I can extend it with more layers / optimizers / methods down the line. Adding a new layer type means implementing `forward()` and `backward()`. The rest of the machinery stays untouched.

---

## Technical Decisions

### Stride-Based Tensor Indexing

I wanted to understand how a tensor actually works. My understanding: a tensor is a blob of floats *plus metadata* that tells you how to interpret them. My `Tensor` struct stores:

```c
typedef struct {
    float* data;      // raw values
    size_t* shape;    // dimensions [batch, features]
    size_t* strides;  // elements to jump per dimension
    size_t ndim;
    size_t size;
} Tensor;
```

Strides are the key insight. For a 2D tensor with shape `[3, 4]`, the strides are `[4, 1]`—meaning to move one row, you skip 4 elements; to move one column, you skip 1. Element `[i, j]` lives at `data[i * strides[0] + j * strides[1]]`.

Why does this matter? Strides let you change how data is *viewed* without copying it. A transpose is just swapping the strides and shape—the underlying data stays put. (My implementation does copy for simplicity, but the architecture supports the optimization.)

Every matrix operation—matmul, add, broadcast—uses stride-aware indexing. It's more verbose than flat indexing, but it's what makes the tensor engine general-purpose. When you're working with large amounts of data with expensive operations, this matters immensely.

### Cache-Optimal Matrix Multiplication

My first matmul implementation was the textbook triple loop:

```c
// Naive ijk ordering
for (int i = 0; i < m; i++) {
    for (int j = 0; j < p; j++) {
        for (int k = 0; k < n; k++) {
            C[i][j] += A[i][k] * B[k][j];
        }
    }
}
```

I realized this was running way slower than what I was used to seeing in Pytorch. I asked my good friend Opus 4.5 for some hints as to why this was the case, and it told me to look into how caching works and how data is stored on the CPU.

What I learned: the problem is memory access patterns. In row-major storage (how C lays out 2D arrays), values in the same row are stored next to each other sequentially. There's a limit to how much can be stored in layer of memory, and there's a difference in how fast each memory layer is. There are registers, which are the fastest / available for immediate access, then the L1 / L2 type caches, then the RAM, each of which is progressively slower to access. In fact, L1 cache access might be 2-3x CPU cycles (essentially how time is measured with CPU processes), whereas accessing RAM might be 200x CPU cycles. So, since `B[k][j]` with varying `k` in the inner loop jumps across rows—each access is a cache miss. Meaning you need to access RAM instead of L1 cache (for example) everytime you jump between rows, so it could be ~200x slower for these operations.

The fix is reordering the loops:

```c
// Cache-friendly ikj ordering
for (int i = 0; i < m; i++) {
    for (int k = 0; k < n; k++) {
        float a_ik = A[i * stride_a0 + k * stride_a1];
        for (int j = 0; j < p; j++) {
            C[i * stride_c0 + j] += a_ik * B[k * stride_b0 + j];
        }
    }
}
```

Now the inner loop walks `B` sequentially across columns and `C` sequentially across columns. Both are cache-friendly. The improvement is dramatic—not from changing the math, but from respecting how hardware actually works.

### Manual Memory Management & Backpropagation

In Python, you allocate objects and the garbage collector cleans up. In C, every `malloc` needs a corresponding `free`, and partial failures need careful unwinding:

```c
Tensor* tensor_create(size_t* shape, size_t ndim) {
    Tensor* t = malloc(sizeof(Tensor));
    if (!t) return NULL;

    t->data = malloc(total_size * sizeof(float));
    if (!t->data) { free(t); return NULL; }

    t->shape = malloc(ndim * sizeof(size_t));
    if (!t->shape) { free(t->data); free(t); return NULL; }

    t->strides = malloc(ndim * sizeof(size_t));
    if (!t->strides) { free(t->shape); free(t->data); free(t); return NULL; }
    // ...
}
```

Every allocation can fail. Every failure must clean up everything allocated before it. This cascading pattern repeats throughout the codebase.

Backpropagation adds another dimension: intermediate values from the forward pass must be cached for the backward pass. My `DenseLayer` stores `input_cache`—the input tensor it saw during forward—because computing weight gradients requires it:

```c
// In backward pass:
// grad_weights = input^T @ grad_output (chain rule)
Tensor* input_transposed = tensor_transpose(layer->input_cache);
Tensor* grad_weights = tensor_matmul(input_transposed, grad_output);
```

The gradient with respect to weights is the outer product of the cached input and the incoming gradient. This is the chain rule made concrete. And because I'm managing memory manually, I have to remember to free `input_transposed` immediately after use, free old gradients before storing new ones, and free the cache when the layer is destroyed.

I validated all of this with Leaks from macos. The final result: **zero memory leaks** across training and inference.

---

## What I'd Do Differently

**GPU support.** CPU-only limits you to toy datasets. Real neural network libraries use CUDA or Metal for parallelism. This would be a significant undertaking—essentially a rewrite of the tensor engine. Still, learning how the GPU code works is an extension of the base concepts (memory management, cache access, parallelism, etc), so I'm highly confident going into this with stronger intution.

**More layers and optimizers.** Axiom only has dense layers, ReLU, and softmax. No convolutions, no dropout, no batch norm. The optimizer is vanilla SGD with a fixed learning rate—no momentum, no Adam. The architecture is modular enough that adding these is just a matter of implementing `forward()` and `backward()` for each new component.

**Strided views for broadcasting.** My broadcast implementation copies data into a new tensor. A more efficient approach would use virtual strides—setting stride to 0 along broadcast dimensions so the same element is reused. I opted for the copy to keep moving as broadcasting rules are still new to me and I wanted to keep the project moving along.

**Code cleanup.** There are hardcoded switch statements keyed on layer type scattered through the codebase. As layers multiply, this becomes a maintenance nightmare. The right fix is a vtable-style dispatch: each layer type provides function pointers for `forward`, `backward`, and `free`. The network code becomes layer-agnostic.

---

## Closing Thoughts

Building Axiom gave me what no tutorial could: intuition.

I now understand C not as syntax but as a way of thinking—where data lives, how it moves, what "ownership" means without a garbage collector. I understand backpropagation to the root level instead of API calling—caching the right values, applying the chain rule, flowing gradients backward through a graph.

And I understand PyTorch. When I call `loss.backward()`, I know there's a graph of tensors, each caching its inputs, each computing gradients with respect to its parameters. The magic is just matrix multiplication and careful memory management, repeated a thousand times.

The black box is open.

---

*[View the source on GitHub →](#)*
