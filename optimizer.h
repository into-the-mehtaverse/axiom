#ifndef OPTIMIZER_H
#define OPTIMIZER_H

typedef struct {
    float learning_rate;
    enum {
        OPTIMIZER_SGD
    } type;
} Optimizer;

Optimizer* optimizer_sgd_create(float learning_rate);
void optimizer_free(Optimizer* opt);

// Update weights and biases using gradients
void optimizer_step(Optimizer* opt, void* layer, const void* gradients);

#endif // OPTIMIZER_H
