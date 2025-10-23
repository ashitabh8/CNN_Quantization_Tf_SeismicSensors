CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O2
LDFLAGS = -lm
TARGET = calculate_features
SOURCE = calculate_features.c

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCE) $(LDFLAGS)

# Clean build artifacts
clean:
	rm -f $(TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Help target
help:
	@echo "Available targets:"
	@echo "  all     - Build the calculate_features executable"
	@echo "  run     - Build and run the program"
	@echo "  clean   - Remove build artifacts"
	@echo "  help    - Show this help message"

.PHONY: all clean run help
