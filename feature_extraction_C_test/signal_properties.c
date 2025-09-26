#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Configuration parameters
#define SIGNAL_LENGTH 1024
#define SAMPLING_RATE 512.0

// Structure to hold all signal properties
typedef struct {
    double signal_energy;
    double above_mean_density;
    int first_max_location;
    int last_max_location;
    double mean_change;
    double mean_abs_change;
    double mean_squared_change;
} signal_properties_t;

// Function prototypes
void calculate_signal_energy(double *signal, int length, double *energy);
void calculate_above_mean_density(double *signal, int length, double *density);
void find_max_locations(double *signal, int length, int *first_max, int *last_max);
void calculate_mean_change_properties(double *signal, int length, 
                                     double *mean_change, double *mean_abs_change, 
                                     double *mean_squared_change);
void calculate_all_properties(double *signal, int length, signal_properties_t *properties);
void print_properties(signal_properties_t *properties);
void save_properties_to_file(signal_properties_t *properties, const char *filename);
void generate_test_signals(double *signal, int length, int test_type);
void save_signal_to_file(double *signal, int length, const char *filename);

/**
 * Calculate signal energy (sum of squared values)
 * @param signal: Input signal array
 * @param length: Length of signal
 * @param energy: Output energy value
 */
void calculate_signal_energy(double *signal, int length, double *energy) {
    *energy = 0.0;
    for (int i = 0; i < length; i++) {
        *energy += signal[i] * signal[i];
    }
}

/**
 * Calculate above mean density (fraction of samples above the mean)
 * @param signal: Input signal array
 * @param length: Length of signal
 * @param density: Output density value (0.0 to 1.0)
 */
void calculate_above_mean_density(double *signal, int length, double *density) {
    // Calculate mean
    double sum = 0.0;
    for (int i = 0; i < length; i++) {
        sum += signal[i];
    }
    double mean = sum / length;
    
    // Count samples above mean
    int above_count = 0;
    for (int i = 0; i < length; i++) {
        if (signal[i] > mean) {
            above_count++;
        }
    }
    
    *density = (double)above_count / length;
}

/**
 * Find first and last locations of maximum value
 * @param signal: Input signal array
 * @param length: Length of signal
 * @param first_max: Output index of first occurrence of maximum
 * @param last_max: Output index of last occurrence of maximum
 */
void find_max_locations(double *signal, int length, int *first_max, int *last_max) {
    if (length == 0) {
        *first_max = -1;
        *last_max = -1;
        return;
    }
    
    double max_value = signal[0];
    *first_max = 0;
    *last_max = 0;
    
    // Find maximum value and first occurrence
    for (int i = 1; i < length; i++) {
        if (signal[i] > max_value) {
            max_value = signal[i];
            *first_max = i;
            *last_max = i;
        } else if (signal[i] == max_value) {
            *last_max = i;  // Update last occurrence
        }
    }
}

/**
 * Calculate mean change properties
 * @param signal: Input signal array
 * @param length: Length of signal
 * @param mean_change: Output mean of first differences
 * @param mean_abs_change: Output mean of absolute first differences
 * @param mean_squared_change: Output mean of squared first differences
 */
void calculate_mean_change_properties(double *signal, int length, 
                                     double *mean_change, double *mean_abs_change, 
                                     double *mean_squared_change) {
    if (length < 2) {
        *mean_change = 0.0;
        *mean_abs_change = 0.0;
        *mean_squared_change = 0.0;
        return;
    }
    
    double sum_change = 0.0;
    double sum_abs_change = 0.0;
    double sum_squared_change = 0.0;
    int num_changes = length - 1;
    
    for (int i = 0; i < num_changes; i++) {
        double change = signal[i + 1] - signal[i];
        sum_change += change;
        sum_abs_change += fabs(change);
        sum_squared_change += change * change;
    }
    
    *mean_change = sum_change / num_changes;
    *mean_abs_change = sum_abs_change / num_changes;
    *mean_squared_change = sum_squared_change / num_changes;
}

/**
 * Calculate all signal properties
 * @param signal: Input signal array
 * @param length: Length of signal
 * @param properties: Output structure containing all properties
 */
void calculate_all_properties(double *signal, int length, signal_properties_t *properties) {
    calculate_signal_energy(signal, length, &properties->signal_energy);
    calculate_above_mean_density(signal, length, &properties->above_mean_density);
    find_max_locations(signal, length, &properties->first_max_location, &properties->last_max_location);
    calculate_mean_change_properties(signal, length, 
                                   &properties->mean_change, 
                                   &properties->mean_abs_change, 
                                   &properties->mean_squared_change);
}

/**
 * Print all properties to console
 * @param properties: Properties structure to print
 */
void print_properties(signal_properties_t *properties) {
    printf("=== Signal Properties ===\n");
    printf("Signal Energy: %.6e\n", properties->signal_energy);
    printf("Above Mean Density: %.6f (%.2f%%)\n", 
           properties->above_mean_density, properties->above_mean_density * 100.0);
    printf("First Max Location: %d\n", properties->first_max_location);
    printf("Last Max Location: %d\n", properties->last_max_location);
    printf("Mean Change: %.6f\n", properties->mean_change);
    printf("Mean Absolute Change: %.6f\n", properties->mean_abs_change);
    printf("Mean Squared Change: %.6e\n", properties->mean_squared_change);
    printf("========================\n\n");
}

/**
 * Save properties to file
 * @param properties: Properties structure to save
 * @param filename: Output filename
 */
void save_properties_to_file(signal_properties_t *properties, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error: Could not open file %s for writing\n", filename);
        return;
    }
    
    fprintf(file, "Property,Value\n");
    fprintf(file, "signal_energy,%.6e\n", properties->signal_energy);
    fprintf(file, "above_mean_density,%.6f\n", properties->above_mean_density);
    fprintf(file, "first_max_location,%d\n", properties->first_max_location);
    fprintf(file, "last_max_location,%d\n", properties->last_max_location);
    fprintf(file, "mean_change,%.6f\n", properties->mean_change);
    fprintf(file, "mean_abs_change,%.6f\n", properties->mean_abs_change);
    fprintf(file, "mean_squared_change,%.6e\n", properties->mean_squared_change);
    
    fclose(file);
    printf("Properties saved to %s\n", filename);
}

/**
 * Save signal to file for analysis
 * @param signal: Signal array
 * @param length: Length of signal
 * @param filename: Output filename
 */
void save_signal_to_file(double *signal, int length, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error: Could not open file %s for writing\n", filename);
        return;
    }
    
    fprintf(file, "Index,Time (s),Amplitude\n");
    for (int i = 0; i < length; i++) {
        double time = i / SAMPLING_RATE;
        fprintf(file, "%d,%.6f,%.6f\n", i, time, signal[i]);
    }
    
    fclose(file);
    printf("Signal saved to %s\n", filename);
}

/**
 * Generate non-periodic test signals for better property testing
 * @param signal: Output signal array
 * @param length: Length of signal
 * @param test_type: Type of test signal (1-4)
 */
void generate_test_signals(double *signal, int length, int test_type) {
    double t;
    
    switch (test_type) {
        case 1: // Exponential decay with noise
            srand(42);
            for (int i = 0; i < length; i++) {
                t = i / SAMPLING_RATE;
                double decay = exp(-t * 2.0);
                double noise = ((double)rand() / RAND_MAX - 0.5) * 0.1;
                signal[i] = decay * sin(2.0 * M_PI * 20.0 * t) + noise;
            }
            break;
            
        case 2: // Chirp signal (frequency sweep)
            for (int i = 0; i < length; i++) {
                t = i / SAMPLING_RATE;
                double freq = 10.0 + 50.0 * t; // Frequency sweep from 10 to 60 Hz
                signal[i] = sin(2.0 * M_PI * freq * t) * exp(-t * 0.5);
            }
            break;
            
        case 3: // Step function with transitions
            for (int i = 0; i < length; i++) {
                t = i / SAMPLING_RATE;
                if (t < 0.5) {
                    signal[i] = 1.0;
                } else if (t < 1.0) {
                    signal[i] = -0.5;
                } else if (t < 1.5) {
                    signal[i] = 0.8;
                } else {
                    signal[i] = 0.0;
                }
                // Add some noise
                signal[i] += ((double)rand() / RAND_MAX - 0.5) * 0.05;
            }
            break;
            
        case 4: // Random walk with trend
            srand(123);
            signal[0] = 0.0;
            for (int i = 1; i < length; i++) {
                t = i / SAMPLING_RATE;
                double trend = 0.1 * t; // Linear trend
                double random_step = ((double)rand() / RAND_MAX - 0.5) * 0.2;
                signal[i] = signal[i-1] + random_step + trend;
            }
            break;
            
        default:
            // Default: simple sine wave
            for (int i = 0; i < length; i++) {
                t = i / SAMPLING_RATE;
                signal[i] = sin(2.0 * M_PI * 30.0 * t);
            }
            break;
    }
}

/**
 * Test function to validate the signal properties implementation
 */
void test_signal_properties() {
    printf("=== Testing Signal Properties Implementation ===\n");
    printf("Configuration:\n");
    printf("  Signal Length: %d samples\n", SIGNAL_LENGTH);
    printf("  Sampling Rate: %.1f Hz\n", SAMPLING_RATE);
    printf("  Duration: %.3f seconds\n", SIGNAL_LENGTH / SAMPLING_RATE);
    printf("\n");
    
    // Test cases with non-periodic signals
    const char* test_names[] = {
        "Exponential Decay with Noise",
        "Chirp Signal (Frequency Sweep)",
        "Step Function with Transitions",
        "Random Walk with Trend"
    };
    
    for (int test = 1; test <= 4; test++) {
        printf("Test %d: %s\n", test, test_names[test-1]);
        printf("--------------------------------------------------\n");
        
        // Allocate memory
        double *signal = (double*) malloc(sizeof(double) * SIGNAL_LENGTH);
        signal_properties_t properties;
        
        // Generate test signal
        generate_test_signals(signal, SIGNAL_LENGTH, test);
        
        // Calculate properties
        calculate_all_properties(signal, SIGNAL_LENGTH, &properties);
        
        // Print results
        print_properties(&properties);
        
        // Save results
        char signal_filename[100];
        char properties_filename[100];
        snprintf(signal_filename, sizeof(signal_filename), "test_signal_properties_%d.txt", test);
        snprintf(properties_filename, sizeof(properties_filename), "test_properties_%d.txt", test);
        
        save_signal_to_file(signal, SIGNAL_LENGTH, signal_filename);
        save_properties_to_file(&properties, properties_filename);
        
        // Clean up
        free(signal);
    }
    
    printf("=== Test completed ===\n");
    printf("Check the generated .txt files for detailed results.\n");
    printf("Use the Python validation script to compare with library implementations.\n");
}

int main() {
    printf("Signal Properties Implementation in C\n");
    printf("====================================\n\n");
    
    // Run tests
    test_signal_properties();
    
    return 0;
}
