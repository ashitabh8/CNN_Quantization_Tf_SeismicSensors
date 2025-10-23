#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../arduino_code_r1/preprocessing.h"

#define MAX_LINE_LENGTH 10000
#define SIGNAL_LENGTH 200
#define MAX_SAMPLES 10

// Structure to hold a sample
typedef struct {
    int sample_index;
    char class[20];
    double signal[SIGNAL_LENGTH];
} sample_t;

// Function to parse CSV line and extract signal data
int parse_csv_line(const char* line, sample_t* sample) {
    char* line_copy = strdup(line);
    char* token;
    int token_count = 0;
    int signal_index = 0;
    
    // Skip header line
    if (strstr(line, "sample_index") != NULL) {
        free(line_copy);
        return 0;
    }
    
    token = strtok(line_copy, ",");
    while (token != NULL && token_count < 3 + SIGNAL_LENGTH) {
        if (token_count == 0) {
            // sample_index
            sample->sample_index = atoi(token);
        } else if (token_count == 1) {
            // class
            strncpy(sample->class, token, sizeof(sample->class) - 1);
            sample->class[sizeof(sample->class) - 1] = '\0';
        } else if (token_count == 2) {
            // signal_length (skip this)
        } else if (token_count >= 3 && signal_index < SIGNAL_LENGTH) {
            // signal values
            sample->signal[signal_index] = atof(token);
            signal_index++;
        }
        token_count++;
        token = strtok(NULL, ",");
    }
    
    free(line_copy);
    return (signal_index == SIGNAL_LENGTH) ? 1 : 0;
}

// Function to calculate all features for a sample
void calculate_features_for_sample(sample_t* sample, double* features) {
    double freqs[NFFT/2 + 1];
    double psd[NFFT/2 + 1];
    int n_freqs;
    
    // Create a copy of the signal for processing (compute_psd modifies the input)
    double signal_copy[SIGNAL_LENGTH];
    memcpy(signal_copy, sample->signal, SIGNAL_LENGTH * sizeof(double));
    
    // Compute PSD
    compute_psd(signal_copy, SIGNAL_LENGTH, freqs, psd, &n_freqs);
    
    // Calculate features in the same order as the CSV output
    features[0] = calc_total_power(sample->signal, SIGNAL_LENGTH);  // total_power
    features[1] = calc_mean_change(psd, n_freqs);          // mean_change
    features[2] = calc_spectral_centroid(psd, freqs, n_freqs); // spectral_centroid
    features[3] = calc_psd_at_frequency(psd, freqs, n_freqs, 35.0); // psd_35hz
    features[4] = calc_psd_at_frequency(psd, freqs, n_freqs, 40.0); // psd_40hz
    features[5] = calc_psd_at_frequency(psd, freqs, n_freqs, 45.0); // psd_45hz
}

int main() {
    FILE* file;
    char line[MAX_LINE_LENGTH];
    sample_t samples[MAX_SAMPLES];
    int sample_count = 0;
    
    printf("=== C Feature Calculator ===\n");
    printf("Reading raw_samples.csv and calculating features...\n\n");
    
    // Open the CSV file
    file = fopen("raw_samples.csv", "r");
    if (file == NULL) {
        printf("Error: Could not open raw_samples.csv\n");
        return 1;
    }
    
    // Read and parse CSV file
    while (fgets(line, sizeof(line), file) && sample_count < MAX_SAMPLES) {
        if (parse_csv_line(line, &samples[sample_count])) {
            sample_count++;
        }
    }
    fclose(file);
    
    printf("Loaded %d samples from CSV file\n\n", sample_count);
    
    // Print header
    printf("%-12s %-10s %-12s %-12s %-15s %-10s %-10s %-10s\n", 
           "Sample", "Class", "Total_Power", "Mean_Change", "Spectral_Centroid", 
           "PSD_35Hz", "PSD_40Hz", "PSD_45Hz");
    printf("%-12s %-10s %-12s %-12s %-15s %-10s %-10s %-10s\n", 
           "------", "-----", "-----------", "-----------", "---------------", 
           "--------", "--------", "--------");
    
    // Calculate and print features for each sample
    for (int i = 0; i < sample_count; i++) {
        double features[6];
        calculate_features_for_sample(&samples[i], features);
        
        printf("%-12d %-10s %-12.4f %-12.4f %-15.4f %-10.4f %-10.4f %-10.4f\n",
               samples[i].sample_index,
               samples[i].class,
               features[0],  // total_power
               features[1],  // mean_change
               features[2],  // spectral_centroid
               features[3],  // psd_35hz
               features[4],  // psd_40hz
               features[5]   // psd_45hz
        );
    }
    
    printf("\n=== Comparison with Python Results ===\n");
    printf("Loading Python results from raw_features.csv for comparison...\n\n");
    
    // Load Python results for comparison
    FILE* python_file = fopen("raw_features.csv", "r");
    if (python_file == NULL) {
        printf("Warning: Could not open raw_features.csv for comparison\n");
        return 0;
    }
    
    // Skip header
    fgets(line, sizeof(line), python_file);
    
    printf("%-12s %-10s %-12s %-12s %-12s %-12s %-10s %-10s %-9s %-8s\n", 
           "Sample", "Class", "C_Total_Power", "Py_Total_Power", "C_Mean_Change", 
           "Py_Mean_Change", "C_PSD_35Hz", "Py_PSD_35Hz", "Diff_35Hz", "Rel_Error");
    printf("%-12s %-10s %-12s %-12s %-12s %-12s %-10s %-10s %-9s %-8s\n", 
           "------", "-----", "-------------", "--------------", "-------------", 
           "-------------", "----------", "-----------", "---------", "---------");
    
    // Compare results
    for (int i = 0; i < sample_count; i++) {
        double features[6];
        calculate_features_for_sample(&samples[i], features);
        
        // Read Python results
        double py_features[6];
        if (fgets(line, sizeof(line), python_file)) {
            char* line_copy = strdup(line);
            char* token = strtok(line_copy, ",");
            int token_count = 0;
            
            while (token != NULL && token_count < 8) {
                if (token_count >= 2) { // Skip sample_index and class
                    py_features[token_count - 2] = atof(token);
                }
                token_count++;
                token = strtok(NULL, ",");
            }
            free(line_copy);
        }
        
        // Calculate relative error for PSD_35Hz
        double diff_35hz = fabs(features[3] - py_features[3]);
        double rel_error = (py_features[3] != 0.0) ? (diff_35hz / fabs(py_features[3])) * 100.0 : 0.0;
        
        printf("%-12d %-10s %-12.4f %-12.4f %-12.4f %-12.4f %-10.4f %-10.4f %-9.4f %-7.2f%%\n",
               samples[i].sample_index,
               samples[i].class,
               features[0],  // C total_power
               py_features[0], // Python total_power
               features[1],  // C mean_change
               py_features[1], // Python mean_change
               features[3],  // C psd_35hz
               py_features[3], // Python psd_35hz
               diff_35hz,    // Absolute difference
               rel_error     // Relative error percentage
        );
    }
    
    fclose(python_file);
    
    printf("\n=== Feature Calculation Complete ===\n");
    printf("Features calculated using C implementation from preprocessing.h\n");
    printf("Results should be similar to Python implementation\n");
    
    return 0;
}
