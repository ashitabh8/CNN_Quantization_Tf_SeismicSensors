#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <complex.h>
#include <stdbool.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Configuration parameters matching the TensorFlow implementation
#define SAMPLING_RATE 512.0
#define NFFT 32
#define NPERSEG 32
#define NOVERLAP 16
#define SCALING "density"
#define DETREND 0

// Function prototypes
void generate_sine_wave(double *signal, int length, double frequency, double amplitude, double phase);
void simple_fft(double complex *data, int n);
void welch_method(double *signal, int signal_length, double *psd, int *psd_length);
void print_psd(double *psd, int length, double sampling_rate);
void save_signal_to_file(double *signal, int length, const char *filename);
void save_psd_to_file(double *psd, int length, double sampling_rate, const char *filename);

/**
 * Generate a sine wave signal
 * @param signal: Output array to store the signal
 * @param length: Length of the signal
 * @param frequency: Frequency of the sine wave in Hz
 * @param amplitude: Amplitude of the sine wave
 * @param phase: Phase offset in radians
 */
void generate_sine_wave(double *signal, int length, double frequency, double amplitude, double phase) {
    for (int i = 0; i < length; i++) {
        signal[i] = amplitude * sin(2.0 * M_PI * frequency * i / SAMPLING_RATE + phase);
    }
}

/**
 * Simple FFT implementation using Cooley-Tukey algorithm
 * @param data: Complex data array (input/output)
 * @param n: Length of the data (must be power of 2)
 */
void simple_fft(double complex *data, int n) {
    if (n <= 1) return;
    
    // Divide
    double complex *even = (double complex*) malloc(n/2 * sizeof(double complex));
    double complex *odd = (double complex*) malloc(n/2 * sizeof(double complex));
    
    for (int i = 0; i < n/2; i++) {
        even[i] = data[2*i];
        odd[i] = data[2*i + 1];
    }
    
    // Conquer
    simple_fft(even, n/2);
    simple_fft(odd, n/2);
    
    // Combine
    for (int i = 0; i < n/2; i++) {
        double complex t = cexp(-2.0 * M_PI * I * i / n) * odd[i];
        data[i] = even[i] + t;
        data[i + n/2] = even[i] - t;
    }
    
    free(even);
    free(odd);
}

/**
 * Implement Welch's method for power spectral density estimation
 * This matches the TensorFlow implementation in dataset.py
 * Note: Data is pre-windowed, so no additional windowing or detrending is applied
 * @param signal: Input signal
 * @param signal_length: Length of the input signal
 * @param psd: Output array for PSD values
 * @param psd_length: Output length of PSD array
 */
void welch_method(double *signal, int signal_length, double *psd, int *psd_length) {
    int nperseg = NPERSEG;
    int noverlap = NOVERLAP;
    int nfft = NFFT;
    
    // Calculate number of segments
    int step = nperseg - noverlap;
    int nsegments = (signal_length - noverlap) / step;
    
    // Output PSD length (one-sided spectrum)
    *psd_length = nfft / 2 + 1;
    
    // Initialize PSD array
    for (int i = 0; i < *psd_length; i++) {
        psd[i] = 0.0;
    }
    
    // Allocate memory for FFT input (complex)
    double complex *fft_input = (double complex*) malloc(sizeof(double complex) * nfft);
    
    // Process each segment
    for (int seg = 0; seg < nsegments; seg++) {
        int start_idx = seg * step;
        
        // Prepare FFT input (zero-pad if necessary)
        for (int i = 0; i < nfft; i++) {
            if (i < nperseg && (start_idx + i) < signal_length) {
                fft_input[i] = signal[start_idx + i] + 0.0 * I;
            } else {
                fft_input[i] = 0.0 + 0.0 * I;
            }
        }
        
        // Perform FFT using our simple implementation
        simple_fft(fft_input, nfft);
        
        // Calculate power spectral density for this segment
        for (int i = 0; i < *psd_length; i++) {
            double magnitude_squared = creal(fft_input[i]) * creal(fft_input[i]) + 
                                     cimag(fft_input[i]) * cimag(fft_input[i]);
            psd[i] += magnitude_squared;
        }
    }
    
    // Average across segments (Welch's method characteristic)
    for (int i = 0; i < *psd_length; i++) {
        psd[i] /= nsegments;
    }
    
    // Apply scaling
    if (strcmp(SCALING, "density") == 0) {
        // Convert to power spectral density
        // PSD = Power / (sampling_rate * window_energy)
        // Since data is pre-windowed, we assume unit window energy
        for (int i = 0; i < *psd_length; i++) {
            psd[i] /= SAMPLING_RATE;
        }
    }
    
    // Clean up
    free(fft_input);
}

/**
 * Print PSD values with frequency information
 * @param psd: PSD array
 * @param length: Length of PSD array
 * @param sampling_rate: Sampling rate in Hz
 */
void print_psd(double *psd, int length, double sampling_rate) {
    printf("Frequency (Hz)\tPSD\n");
    printf("----------------\n");
    
    for (int i = 0; i < length; i++) {
        double frequency = i * sampling_rate / (2 * (length - 1));
        printf("%.2f\t\t%.6e\n", frequency, psd[i]);
    }
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
    
    fprintf(file, "Time (s)\tAmplitude\n");
    for (int i = 0; i < length; i++) {
        double time = i / SAMPLING_RATE;
        fprintf(file, "%.6f\t%.6f\n", time, signal[i]);
    }
    
    fclose(file);
    printf("Signal saved to %s\n", filename);
}

/**
 * Save PSD to file for analysis
 * @param psd: PSD array
 * @param length: Length of PSD array
 * @param sampling_rate: Sampling rate in Hz
 * @param filename: Output filename
 */
void save_psd_to_file(double *psd, int length, double sampling_rate, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error: Could not open file %s for writing\n", filename);
        return;
    }
    
    fprintf(file, "Frequency (Hz)\tPSD\n");
    for (int i = 0; i < length; i++) {
        double frequency = i * sampling_rate / (2 * (length - 1));
        fprintf(file, "%.6f\t%.6e\n", frequency, psd[i]);
    }
    
    fclose(file);
    printf("PSD saved to %s\n", filename);
}

/**
 * Test function to validate the Welch's method implementation
 */
void test_welch_implementation() {
    printf("=== Testing Welch's Method Implementation ===\n");
    printf("Configuration (matching dataset_config.yaml):\n");
    printf("  Sampling Rate: %.1f Hz\n", SAMPLING_RATE);
    printf("  NFFT: %d\n", NFFT);
    printf("  NPERSEG: %d\n", NPERSEG);
    printf("  NOVERLAP: %d\n", NOVERLAP);
    printf("  Window: pre-windowed (no additional windowing)\n");
    printf("  Scaling: %s\n", SCALING);
    printf("  Detrend: %s (disabled as per config)\n", DETREND ? "true" : "false");
    printf("\n");
    
    // Test 1: Single frequency sine wave
    printf("Test 1: Single frequency sine wave (50 Hz)\n");
    int signal_length = 1024;
    double *signal = (double*) malloc(sizeof(double) * signal_length);
    double *psd = (double*) malloc(sizeof(double) * (NFFT / 2 + 1));
    int psd_length;
    
    // Generate 50 Hz sine wave
    generate_sine_wave(signal, signal_length, 50.0, 1.0, 0.0);
    
    // Apply Welch's method
    welch_method(signal, signal_length, psd, &psd_length);
    
    // Save results
    save_signal_to_file(signal, signal_length, "test_signal_50hz.txt");
    save_psd_to_file(psd, psd_length, SAMPLING_RATE, "test_psd_50hz.txt");
    
    // Find peak frequency
    int peak_idx = 0;
    double max_psd = psd[0];
    for (int i = 1; i < psd_length; i++) {
        if (psd[i] > max_psd) {
            max_psd = psd[i];
            peak_idx = i;
        }
    }
    double peak_frequency = peak_idx * SAMPLING_RATE / (2 * (psd_length - 1));
    printf("Peak frequency: %.2f Hz (expected: 50.0 Hz)\n", peak_frequency);
    printf("Peak PSD value: %.6e\n", max_psd);
    printf("\n");
    
    // Test 2: Multiple frequency sine wave
    printf("Test 2: Multiple frequency sine wave (30 Hz + 80 Hz)\n");
    double *signal2 = (double*) malloc(sizeof(double) * signal_length);
    double *psd2 = (double*) malloc(sizeof(double) * (NFFT / 2 + 1));
    int psd2_length;
    
    // Generate combined sine wave
    for (int i = 0; i < signal_length; i++) {
        signal2[i] = sin(2.0 * M_PI * 30.0 * i / SAMPLING_RATE) + 
                     0.5 * sin(2.0 * M_PI * 80.0 * i / SAMPLING_RATE);
    }
    
    // Apply Welch's method
    welch_method(signal2, signal_length, psd2, &psd2_length);
    
    // Save results
    save_signal_to_file(signal2, signal_length, "test_signal_multi.txt");
    save_psd_to_file(psd2, psd2_length, SAMPLING_RATE, "test_psd_multi.txt");
    
    // Find peaks
    printf("Peak frequencies found:\n");
    for (int i = 1; i < psd2_length - 1; i++) {
        if (psd2[i] > psd2[i-1] && psd2[i] > psd2[i+1] && psd2[i] > max_psd * 0.1) {
            double freq = i * SAMPLING_RATE / (2 * (psd2_length - 1));
            printf("  %.2f Hz: PSD = %.6e\n", freq, psd2[i]);
        }
    }
    printf("\n");
    
    // Test 3: Noisy signal
    printf("Test 3: Noisy sine wave (40 Hz + noise)\n");
    double *signal3 = (double*) malloc(sizeof(double) * signal_length);
    double *psd3 = (double*) malloc(sizeof(double) * (NFFT / 2 + 1));
    int psd3_length;
    
    // Generate noisy sine wave
    srand(42); // Fixed seed for reproducibility
    for (int i = 0; i < signal_length; i++) {
        double noise = ((double)rand() / RAND_MAX - 0.5) * 0.2;
        signal3[i] = sin(2.0 * M_PI * 40.0 * i / SAMPLING_RATE) + noise;
    }
    
    // Apply Welch's method
    welch_method(signal3, signal_length, psd3, &psd3_length);
    
    // Save results
    save_signal_to_file(signal3, signal_length, "test_signal_noisy.txt");
    save_psd_to_file(psd3, psd3_length, SAMPLING_RATE, "test_psd_noisy.txt");
    
    // Find peak frequency
    peak_idx = 0;
    max_psd = psd3[0];
    for (int i = 1; i < psd3_length; i++) {
        if (psd3[i] > max_psd) {
            max_psd = psd3[i];
            peak_idx = i;
        }
    }
    peak_frequency = peak_idx * SAMPLING_RATE / (2 * (psd3_length - 1));
    printf("Peak frequency: %.2f Hz (expected: 40.0 Hz)\n", peak_frequency);
    printf("Peak PSD value: %.6e\n", max_psd);
    printf("\n");
    
    // Clean up
    free(signal);
    free(psd);
    free(signal2);
    free(psd2);
    free(signal3);
    free(psd3);
    
    printf("=== Test completed ===\n");
    printf("Check the generated .txt files for detailed results.\n");
    printf("Use the Python validation script to compare with TensorFlow implementation.\n");
}

int main() {
    printf("Welch's Method Implementation in C\n");
    printf("==================================\n\n");
    
    // Run tests
    test_welch_implementation();
    
    return 0;
}
