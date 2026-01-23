#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "axiom.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <command> [options]\n", argv[0]);
        printf("Commands:\n");
        printf("  train --epochs <n> --lr <rate>  Train the model\n");
        printf("  predict <model_file> <input>     Run inference\n");
        return 1;
    }

    if (strcmp(argv[1], "train") == 0) {
        // TODO: Parse --epochs and --lr flags
        // TODO: Load MNIST data
        // TODO: Create network and train
        printf("Training not yet implemented\n");
    } else if (strcmp(argv[1], "predict") == 0) {
        // TODO: Load model and run inference
        printf("Inference not yet implemented\n");
    } else {
        printf("Unknown command: %s\n", argv[1]);
        return 1;
    }

    return 0;
}
