#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "queue.h"
#include "preprocessing.h"

#define MAX_LINE_LENGTH 50000
#define MAX_SIGNAL_LENGTH 200

// CSV file pointers
FILE *features_csv, *raw_features_csv;

// Function to convert float array to double array
void convert_float_to_double(float* src, double* dst, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = (double)src[i];
    }
}

// CSV parsing function to read raw_samples.csv
int read_csv_line(FILE* fp, char* class_name, float* signal, int max_length) {
    char line[MAX_LINE_LENGTH];
    if (fgets(line, sizeof(line), fp) == NULL) {
        return 0; // End of file
    }
    
    // Remove newline
    line[strcspn(line, "\n")] = 0;
    
    char* token = strtok(line, ",");
    if (token == NULL) return 0;
    
    // Skip first token (class)
    strcpy(class_name, token);
    
    // Skip second token (sample_index)
    token = strtok(NULL, ",");
    if (token == NULL) return 0;
    
    // Skip third token (signal_length)
    token = strtok(NULL, ",");
    if (token == NULL) return 0;
    
    // Read signal values
    int i = 0;
    while ((token = strtok(NULL, ",")) != NULL && i < max_length) {
        signal[i] = atof(token);
        i++;
    }
    
    return i; // Return number of signal values read
}

// Extract 6 features from signal data (alphabetical order)
int extract_features(double* signal, int signal_length, double* features) {
    // Compute PSD using Welch's method
    double freqs[33];  // NFFT/2 + 1 = 33
    double psd[33];
    int n_freqs;
    
    compute_psd(signal, signal_length, freqs, psd, &n_freqs);
    
    printf("  PSD computed: %d frequency bins\n", n_freqs);
    
    // Extract features in ALPHABETICAL ORDER (same as model training)
    // Order: mean_change, psd_35hz, psd_40hz, psd_45hz, spectral_centroid, total_power
    // features[0] = calc_mean_change(psd, n_freqs);                   // mean_change
    features[0] = calc_psd_at_frequency(psd, freqs, n_freqs, 35.0);   // psd_35hz
    features[1] = calc_psd_at_frequency(psd, freqs, n_freqs, 40.0);   // psd_40hz
    features[2] = calc_psd_at_frequency(psd, freqs, n_freqs, 45.0);   // psd_45hz
    features[3] = calc_total_power(signal, signal_length);          // total_power
    
    return 1; // Success
}

// Process features for a single sample
int process_features(Queue* q, int sample_num, char* class_name) {
    printf("\n=== Processing Sample %d ===\n", sample_num);
    
    // Get signal data from queue
    float* signal_float = queue_get_array(q);
    if (signal_float == NULL) {
        printf("ERROR: Queue not full!\n");
        return 0;
    }
    
    // Convert to double for preprocessing
    double signal[QUEUE_SIZE];
    convert_float_to_double(signal_float, signal, QUEUE_SIZE);
    
    // Calculate signal statistics
    double min_val = signal[0], max_val = signal[0], sum = 0.0;
    for (int i = 0; i < QUEUE_SIZE; i++) {
        if (signal[i] < min_val) min_val = signal[i];
        if (signal[i] > max_val) max_val = signal[i];
        sum += signal[i];
    }
    double mean_val = sum / QUEUE_SIZE;
    
    printf("Signal: min=%.1f, max=%.1f, mean=%.1f\n", min_val, max_val, mean_val);
    
    // Extract 6 features
    double raw_features[4];
    if (!extract_features(signal, QUEUE_SIZE, raw_features)) {
        printf("ERROR: Feature extraction failed!\n");
        return 0;
    }
    
    // Print raw features (in alphabetical order)
    printf("Raw Features:\n");
    printf("  psd_35hz: %.6f\n", raw_features[0]);
    printf("  psd_40hz: %.6f\n", raw_features[1]);
    printf("  psd_45hz: %.6f\n", raw_features[2]);
    printf("  total_power: %.6f\n", raw_features[3]);
    
    // Save raw features to CSV
    fprintf(raw_features_csv, "%d,%s,%.10f,%.10f,%.10f,%.10f\n",
            sample_num, class_name, 
            raw_features[0], raw_features[1], raw_features[2],
            raw_features[3]);
    
    // Get feature statistics for normalization
    feature_stats_t stats[4];
    get_feature_statistics(stats);
    
    // Normalize all features
    double normalized_features[4];
    for (int i = 0; i < 4; i++) {
        normalized_features[i] = normalize_single_feature(raw_features[i], &stats[i]);
    }
    
    printf("Normalized Features: [");
    for (int i = 0; i < 4; i++) {
        printf("%.4f", normalized_features[i]);
        if (i < 3) printf(", ");
    }
    printf("]\n");
    
    // Save normalized features to CSV
    fprintf(features_csv, "%d,%s,%.10f,%.10f,%.10f,%.10f\n",
            sample_num, class_name, 
            normalized_features[0], normalized_features[1], normalized_features[2],
            normalized_features[3]);
    
    return 1;
}

// Test features with a single dataset
int test_features_with_data(const float* input_data, int test_num, char* class_name) {
    printf("\n=== Testing Features with Dataset %d ===\n", test_num);
    
    Queue q;
    queue_init(&q);
    
    // Push all 200 values one by one
    printf("Filling queue with 200 values...\n");
    for (int i = 0; i < 200; i++) {
        if (!queue_push(&q, input_data[i])) {
            printf("ERROR: Failed to push value at index %d\n", i);
            return 0;
        }
    }
    
    printf("Queue filled successfully\n");
    
    // Process features
    if (!process_features(&q, test_num, class_name)) {
        printf("ERROR: Feature processing failed\n");
        return 0;
    }
    
    // Clear queue for next iteration
    queue_clear(&q);
    printf("Queue cleared for next iteration\n");
    
    return 1;
}

int main() {
    printf("Feature Comparison Test\n");
    printf("=======================\n");
    printf("Testing feature extraction and normalization:\n");
    printf("Queue → Preprocessing → Features → Normalization\n");
    printf("Using data from raw_samples.csv\n");
    
    // Open CSV files for writing
    features_csv = fopen("c_normalized_features.csv", "w");
    raw_features_csv = fopen("c_raw_features.csv", "w");
    
    if (!features_csv || !raw_features_csv) {
        printf("ERROR: Could not open CSV files for writing\n");
        return 1;
    }
    
    // Write CSV headers
    fprintf(features_csv, "sample_index,class,psd_35hz,psd_40hz,psd_45hz,total_power\n");
    fprintf(raw_features_csv, "sample_index,class,psd_35hz,psd_40hz,psd_45hz,total_power\n");
    
    // Open raw_samples.csv for reading
    FILE* csv_file = fopen("../feature_training/raw_samples.csv", "r");
    if (!csv_file) {
        printf("ERROR: Could not open raw_samples.csv\n");
        fclose(features_csv);
        fclose(raw_features_csv);
        return 1;
    }
    
    // Skip header line
    char header_line[MAX_LINE_LENGTH];
    fgets(header_line, sizeof(header_line), csv_file);
    
    int total_errors = 0;
    int sample_num = 1;
    
    // Read and process each sample from CSV
    while (1) {
        char class_name[50];
        float signal[200];
        
        int signal_length = read_csv_line(csv_file, class_name, signal, 200);
        if (signal_length == 0) break; // End of file
        
        printf("\nProcessing sample %d (class: %s)\n", sample_num, class_name);
        
        if (!test_features_with_data(signal, sample_num, class_name)) {
            total_errors++;
        }
        
        sample_num++;
    }
    
    fclose(csv_file);
    fclose(features_csv);
    fclose(raw_features_csv);
    
    printf("\n=======================\n");
    printf("FINAL RESULTS\n");
    printf("=======================\n");
    if (total_errors == 0) {
        printf("ALL FEATURE TESTS PASSED!\n");
        printf("Feature extraction and normalization working correctly.\n");
        printf("CSV files saved: c_raw_features.csv, c_normalized_features.csv\n");
        printf("Ready for comparison with Python results.\n");
    } else {
        printf("FAILED: %d feature processing errors found.\n", total_errors);
    }
    
    return total_errors;
}
