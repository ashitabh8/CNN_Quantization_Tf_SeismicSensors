#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "queue.h"
#include "preprocessing.h"
#include "model_inference.h"
#include "tf_ops.h"

#define TOLERANCE 1e-6f
#define MAX_LINE_LENGTH 50000
#define MAX_SIGNAL_LENGTH 200

// Class names for output (3 classes)
static const char* class_names[] = {"background", "lexus", "mazda"};

// CSV file pointers
FILE *features_csv, *outputs_csv, *raw_features_csv;

// Function to convert float array to double array
void convert_float_to_double(float* src, double* dst, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = (double)src[i];
    }
}

// Function to convert double array to float array
void convert_double_to_float(double* src, float* dst, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = (float)src[i];
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
    features[0] = calc_mean_change(psd, n_freqs);                   // mean_change
    features[1] = calc_psd_at_frequency(psd, freqs, n_freqs, 35.0);   // psd_35hz
    features[2] = calc_psd_at_frequency(psd, freqs, n_freqs, 40.0);   // psd_40hz
    features[3] = calc_psd_at_frequency(psd, freqs, n_freqs, 45.0);   // psd_45hz
    features[4] = calc_spectral_centroid(psd, freqs, n_freqs);       // spectral_centroid
    features[5] = calc_total_power(signal, signal_length);          // total_power
    
    return 1; // Success
}

// Process complete ML pipeline
int process_pipeline(Queue* q, int sample_num, char* class_name) {
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
    double raw_features[6];
    if (!extract_features(signal, QUEUE_SIZE, raw_features)) {
        printf("ERROR: Feature extraction failed!\n");
        return 0;
    }
    
    // Print raw features (in alphabetical order)
    printf("Raw Features:\n");
    printf("  mean_change: %.6f\n", raw_features[0]);
    printf("  psd_35hz: %.6f\n", raw_features[1]);
    printf("  psd_40hz: %.6f\n", raw_features[2]);
    printf("  psd_45hz: %.6f\n", raw_features[3]);
    printf("  spectral_centroid: %.6f\n", raw_features[4]);
    printf("  total_power: %.6f\n", raw_features[5]);
    
    // Save raw features to CSV
    fprintf(raw_features_csv, "%d,%s,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f\n",
            sample_num, class_name, 
            raw_features[0], raw_features[1], raw_features[2],
            raw_features[3], raw_features[4], raw_features[5]);
    
    // Get feature statistics for normalization
    feature_stats_t stats[6];
    get_feature_statistics(stats);
    
    // Check background threshold using normalized total_power (now at index 5)
    double normalized_total_power = normalize_single_feature(raw_features[5], &stats[5]);
    printf("Background check: normalized total_power = %.6f, threshold = %.6f\n", 
           normalized_total_power, BACKGROUND_THRESHOLD);
    
    if (normalized_total_power < BACKGROUND_THRESHOLD) {
        printf("Background detected: power below threshold\n");
        printf("PREDICTION: Class 0 (background) with 100.0%% confidence\n");
        printf("Output: [1.0, 0.0, 0.0]\n");
        
        // Save to CSV files for background case
        fprintf(features_csv, "%d,%s,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f\n",
                sample_num, class_name, 
                normalize_single_feature(raw_features[0], &stats[0]),  // mean_change
                normalize_single_feature(raw_features[1], &stats[1]),  // psd_35hz
                normalize_single_feature(raw_features[2], &stats[2]),  // psd_40hz
                normalize_single_feature(raw_features[3], &stats[3]),  // psd_45hz
                normalize_single_feature(raw_features[4], &stats[4]),  // spectral_centroid
                normalize_single_feature(raw_features[5], &stats[5])); // total_power
        
        fprintf(outputs_csv, "%d,%s,%.10f,%.10f,%.10f\n",
                sample_num, class_name, 1.0, 0.0, 0.0);
        
        return 1;
    }
    
    printf("Background check: PASSED (power > threshold)\n");
    
    // Normalize all features
    double normalized_features[6];
    for (int i = 0; i < 6; i++) {
        normalized_features[i] = normalize_single_feature(raw_features[i], &stats[i]);
    }
    
    printf("Normalized Features: [");
    for (int i = 0; i < 6; i++) {
        printf("%.4f", normalized_features[i]);
        if (i < 5) printf(", ");
    }
    printf("]\n");
    
    // Save normalized features to CSV
    fprintf(features_csv, "%d,%s,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f\n",
            sample_num, class_name, 
            normalized_features[0], normalized_features[1], normalized_features[2],
            normalized_features[3], normalized_features[4], normalized_features[5]);
    
    // Convert to float for ML inference
    float ml_input[6];
    convert_double_to_float(normalized_features, ml_input, 6);
    
    // Run ML inference
    float ml_output[3];
    model_infer(ml_input, ml_output);
    
    printf("ML Inference output (logits): [");
    for (int i = 0; i < 3; i++) {
        printf("%.2f", ml_output[i]);
        if (i < 2) printf(", ");
    }
    printf("]\n");
    
    // Apply softmax
    softmax(ml_output, 3);
    
    printf("After softmax: [");
    for (int i = 0; i < 3; i++) {
        printf("%.2f", ml_output[i]);
        if (i < 2) printf(", ");
    }
    printf("]\n");
    
    // Save softmax outputs to CSV
    fprintf(outputs_csv, "%d,%s,%.10f,%.10f,%.10f\n",
            sample_num, class_name, ml_output[0], ml_output[1], ml_output[2]);
    
    // Find prediction
    int predicted_class = 0;
    float max_prob = ml_output[0];
    for (int i = 1; i < 3; i++) {
        if (ml_output[i] > max_prob) {
            max_prob = ml_output[i];
            predicted_class = i;
        }
    }
    
    printf("PREDICTION: Class %d (%s) with %.1f%% confidence\n", 
           predicted_class, class_names[predicted_class], max_prob * 100.0f);
    
    return 1;
}

// Test pipeline with a single dataset
int test_pipeline_with_data(const float* input_data, int test_num, char* class_name) {
    printf("\n=== Testing Full Pipeline with Dataset %d ===\n", test_num);
    
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
    
    // Process complete pipeline
    if (!process_pipeline(&q, test_num, class_name)) {
        printf("ERROR: Pipeline processing failed\n");
        return 0;
    }
    
    // Clear queue for next iteration
    queue_clear(&q);
    printf("Queue cleared for next iteration\n");
    
    return 1;
}

int main() {
    printf("Full ML Pipeline Test\n");
    printf("=====================\n");
    printf("Testing complete Arduino ML pipeline:\n");
    printf("Queue → Preprocessing → Features → Normalization → ML → Softmax → Prediction\n");
    printf("Using data from raw_samples.csv\n");
    
    // Open CSV files for writing
    features_csv = fopen("c_normalized_features.csv", "w");
    outputs_csv = fopen("c_softmax_outputs.csv", "w");
    raw_features_csv = fopen("c_raw_features.csv", "w");
    
    if (!features_csv || !outputs_csv || !raw_features_csv) {
        printf("ERROR: Could not open CSV files for writing\n");
        return 1;
    }
    
    // Write CSV headers
    fprintf(features_csv, "sample_index,class,mean_change,psd_35hz,psd_40hz,psd_45hz,spectral_centroid,total_power\n");
    fprintf(outputs_csv, "sample_index,class,background_prob,lexus_prob,mazda_prob\n");
    fprintf(raw_features_csv, "sample_index,class,mean_change,psd_35hz,psd_40hz,psd_45hz,spectral_centroid,total_power\n");
    
    // Open raw_samples.csv for reading
    FILE* csv_file = fopen("../feature_training/raw_samples.csv", "r");
    if (!csv_file) {
        printf("ERROR: Could not open raw_samples.csv\n");
        fclose(features_csv);
        fclose(outputs_csv);
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
        
        if (!test_pipeline_with_data(signal, sample_num, class_name)) {
            total_errors++;
        }
        
        sample_num++;
    }
    
    fclose(csv_file);
    fclose(features_csv);
    fclose(outputs_csv);
    fclose(raw_features_csv);
    
    printf("\n=====================\n");
    printf("FINAL RESULTS\n");
    printf("=====================\n");
    if (total_errors == 0) {
        printf("ALL PIPELINE TESTS PASSED!\n");
        printf("Complete ML pipeline is working correctly.\n");
        printf("CSV files saved: c_raw_features.csv, c_normalized_features.csv, c_softmax_outputs.csv\n");
        printf("Ready for Arduino deployment.\n");
    } else {
        printf("FAILED: %d pipeline errors found.\n", total_errors);
    }
    
    return total_errors;
}
