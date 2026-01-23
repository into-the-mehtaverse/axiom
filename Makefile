CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -O2 -g
LDFLAGS = -lm

# Source files
SRCS = tensor.c dense.c activations.c optimizer.c loss.c axiom.c main.c
OBJS = $(SRCS:.c=.o)
TARGET = main

# Default target
all: $(TARGET)

# Build executable
$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

# Compile object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJS) $(TARGET)

# Run with Valgrind for memory leak detection
valgrind: $(TARGET)
	valgrind --leak-check=full --show-leak-kinds=all ./$(TARGET)

.PHONY: all clean valgrind
