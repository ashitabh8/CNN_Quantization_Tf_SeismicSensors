#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../arduino_code_r1/preprocessing.h"

#define MAX_SIGNAL_LENGTH 1000
#define MAX_PSD_LENGTH 100

int main() {
    printf("C Welch Implementation Test\n");
    printf("==========================\n\n");
    
    // Read signal from CSV file
    double signal[MAX_SIGNAL_LENGTH];
    int signal_length = 0;
    
    FILE *signal_file = fopen("test_signal.csv", "r");
    if (signal_file == NULL) {
        printf("Error: Could not open test_signal.csv\n");
        return 1;
    }
    
    // Read signal values
    double value;
    while (fscanf(signal_file, "%lf", &value) == 1 && signal_length < MAX_SIGNAL_LENGTH) {
        signal[signal_length] = value;
        signal_length++;
    }
    fclose(signal_file);
    
    printf("Read signal with %d samples\n", signal_length);
    
    // Allocate arrays for PSD computation
    double freqs[MAX_PSD_LENGTH];
    double psd[MAX_PSD_LENGTH];
    int n_freqs;
    
    // Compute PSD using the fixed implementation
    compute_psd(signal, signal_length, freqs, psd, &n_freqs);
    
    printf("Computed PSD with %d frequency bins\n", n_freqs);
    printf("Frequency range: %.2f - %.2f Hz\n", freqs[0], freqs[n_freqs-1]);
    
    // Save PSD to CSV
    FILE *psd_file = fopen("test_psd_c.csv", "w");
    if (psd_file == NULL) {
        printf("Error: Could not create test_psd_c.csv\n");
        return 1;
    }
    
    for (int i = 0; i < n_freqs; i++) {
        fprintf(psd_file, "%.15e\n", psd[i]);
    }
    fclose(psd_file);
    
    // Save frequencies to CSV
    FILE *freqs_file = fopen("test_freqs_c.csv", "w");
    if (freqs_file == NULL) {
        printf("Error: Could not create test_freqs_c.csv\n");
        return 1;
    }
    
    for (int i = 0; i < n_freqs; i++) {
        fprintf(freqs_file, "%.15e\n", freqs[i]);
    }
    fclose(freqs_file);
    
    printf("Saved PSD and frequencies to CSV files\n");
    
    // Print some statistics
    double max_psd = 0.0;
    int max_idx = 0;
    for (int i = 0; i < n_freqs; i++) {
        if (psd[i] > max_psd) {
            max_psd = psd[i];
            max_idx = i;
        }
    }
    
    printf("Peak PSD: %.6e at %.2f Hz\n", max_psd, freqs[max_idx]);
    
    return 0;
}
