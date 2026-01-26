CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -O2 -g -Isrc
LDFLAGS = -lm

SRCS = src/tensor.c src/dense.c src/activations.c src/optimizer.c src/loss.c src/axiom.c src/mnist.c src/main.c
OBJS = $(patsubst src/%.c,build/%.o,$(SRCS))
TARGET = build/main

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

build/%.o: src/%.c
	@mkdir -p build
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf build

valgrind: $(TARGET)
	valgrind --leak-check=full --show-leak-kinds=all $(TARGET) test

.PHONY: all clean valgrind
