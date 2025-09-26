# Makefile for Welch's method implementation
CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99
LDFLAGS = -lm

# Target executable
TARGET = welchs_test

# Source files
SOURCES = welchs.c

# Object files
OBJECTS = $(SOURCES:.c=.o)

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

# Compile source files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET) *.txt

# Run the test
test: $(TARGET)
	./$(TARGET)

# Install dependencies (Ubuntu/Debian)
install-deps:
	@echo "No external dependencies required - using built-in FFT implementation"

# Install dependencies (CentOS/RHEL)
install-deps-centos:
	@echo "No external dependencies required - using built-in FFT implementation"

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build the executable (default)"
	@echo "  clean        - Remove build artifacts"
	@echo "  test         - Build and run the test"
	@echo "  install-deps - Check dependencies (none required)"
	@echo "  install-deps-centos - Check dependencies (none required)"
	@echo "  help         - Show this help message"

.PHONY: all clean test install-deps install-deps-centos help
