# Makefile for signal processing implementations
CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99
LDFLAGS = -lm

# Target executables
WELCHS_TARGET = welchs_test
PROPERTIES_TARGET = signal_properties_test

# Source files
WELCHS_SOURCES = welchs.c
PROPERTIES_SOURCES = signal_properties.c

# Object files
WELCHS_OBJECTS = $(WELCHS_SOURCES:.c=.o)
PROPERTIES_OBJECTS = $(PROPERTIES_SOURCES:.c=.o)

# Default target
all: $(WELCHS_TARGET) $(PROPERTIES_TARGET)

# Build Welch's method executable
$(WELCHS_TARGET): $(WELCHS_OBJECTS)
	$(CC) $(WELCHS_OBJECTS) -o $(WELCHS_TARGET) $(LDFLAGS)

# Build signal properties executable
$(PROPERTIES_TARGET): $(PROPERTIES_OBJECTS)
	$(CC) $(PROPERTIES_OBJECTS) -o $(PROPERTIES_TARGET) $(LDFLAGS)

# Compile source files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(WELCHS_OBJECTS) $(PROPERTIES_OBJECTS) $(WELCHS_TARGET) $(PROPERTIES_TARGET) *.txt

# Run Welch's test
test-welch: $(WELCHS_TARGET)
	./$(WELCHS_TARGET)

# Run signal properties test
test-properties: $(PROPERTIES_TARGET)
	./$(PROPERTIES_TARGET)

# Run all tests
test: test-welch test-properties

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
