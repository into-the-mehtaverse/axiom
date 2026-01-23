# c-neural-net
Welcome to axiom - my lightweight, dependency-free neural network library written in pure C from scratch.

> **Status:** Active Development
> **Goal:** 95%+ Accuracy on MNIST using purely handwritten backpropagation and matrix operations.

## ⚡️ Why?
Modern ML frameworks (PyTorch, TensorFlow) abstract away the actual engineering. I built this to understand the "black box" at the lowest level.
This library implements tensors, computational graphs, and automatic differentiation entirely from scratch—managing every byte of memory manually.

Only C in this repo, no BLAS, LAPACK, or Python.

Additionally, **No LLMs were harmed or used in the writing of this code.**

## 🚀 Features
- **Pure C Implementation:** Zero external dependencies. Standard library only (`<stdlib.h>`, `<math.h>`).
- **Custom Tensor Engine:** Handwritten matrix operations (matmul, transpose, broadcast).
- **Automatic Differentiation:** Implements full backpropagation for dense layers.
- **Memory Safety:** rigorously tested with Valgrind to ensure **0 memory leaks**.
- **Optimization:** Stochastic Gradient Descent (SGD) with configurable learning rates.
- **Serialization:** Save and load trained models for inference.

## 🛠 Tech Stack & Architecture
- **Language:** C
- **Build System:** Makefile
- **Profiling:** Valgrind (Memcheck)

### Core Components
1. **`tensor.c`**: The engine. Handles raw data pointers, shape strides, and matrix math.
2. **`layers.c`**: Implements the forward and backward passes for `Dense` (Fully Connected) layers.
3. **`activations.c`**: ReLU (hidden layers) and Softmax (output probability distribution).
4. **`optimizer.c`**: Handles weight updates via SGD.

## 📊 Benchmarks (MNIST)
Training a 3-layer network (784 -> 128 -> 10) on the MNIST dataset:

| Metric | Result |
|--------|--------|
| **Accuracy** | **~96.5%** |
| **Training Time** | ~15s (CPU) |
| **Memory Usage** | < 50MB |
| **Leaks** | **0 bytes** |

## 💻 Usage

### Build & Run
\`\`\`bash
make
./main train --epochs 10 --lr 0.01
\`\`\`

### C API Example
\`\`\`c
// Define the network architecture
AxiomNet* net = axiom_create();
axiom_add(net, axiom_layer_dense(784, 128));
axiom_add(net, axiom_activation_relu());
axiom_add(net, axiom_layer_dense(128, 10));
axiom_add(net, axiom_activation_softmax());

// Train on data
axiom_train(net, x_train, y_train, epochs, learning_rate);

// Save the model
axiom_save(net, "mnist_model.bin");
\`\`\`

## 📜 License
MIT
```
