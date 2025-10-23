#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

// Constants matching feature_utils.py
#define SAMPLING_RATE 100.0
#define NPERSEG 50
#define NOVERLAP 25
#define NFFT 64
#define PI 3.14159265358979323846
#define BACKGROUND_THRESHOLD 0.0002488944592187181
#define STEP (NPERSEG - NOVERLAP)  // 25
#define N_BINS (NFFT / 2 + 1) 

// Structure for complex numbers
typedef struct {
    double real;
    double imag;
} complex_t;

// Feature statistics structure
typedef struct {
    double min;
    double max;
    double mean;
    double std;
    double range;
    int count;
} feature_stats_t;

// Function prototypes
void generate_hann_window(double* window, int n);
void dft(complex_t* input, complex_t* output, int n);
void compute_psd(double* signal, int signal_length, double* freqs, double* psd, int* n_freqs);
double calc_total_power(double* signal, int n);
double calc_above_mean_density(double* psd, int n);
double calc_mean_abs_change(double* psd, int n);
double calc_mean_change(double* psd, int n);
double calc_spectral_centroid(double* psd, double* freqs, int n);
double calc_psd_at_frequency(double* psd, double* freqs, int n, double target_freq);

// Normalization functions
double normalize_single_feature(double value, feature_stats_t* stats);
void normalize_all_features(double* features, feature_stats_t* stats_array, int n_features);
void get_feature_statistics(feature_stats_t* stats_array);


// ============================================================================
// Helper Functions
// ============================================================================

void generate_hann_window(double* window, int n) {
    for (int i = 0; i < n; i++) {
        window[i] = 0.5 * (1.0 - cos(2.0 * PI * i / (n - 1)));
    }
}

void dft(complex_t* input, complex_t* output, int n) {
    for (int k = 0; k < n; k++) {
        output[k].real = 0.0;
        output[k].imag = 0.0;
        
        for (int j = 0; j < n; j++) {
            double angle = -2.0 * PI * k * j / n;
            double cos_val = cos(angle);
            double sin_val = sin(angle);
            
            output[k].real += input[j].real * cos_val - input[j].imag * sin_val;
            output[k].imag += input[j].real * sin_val + input[j].imag * cos_val;
        }
    }
}

/**
 * Compute magnitude squared of complex number
 */
static inline double complex_mag_sq(complex_t c) {
    return c.real * c.real + c.imag * c.imag;
}

// ============================================================================
// Main Welch's Method Implementation
// ============================================================================

/**
 * Compute PSD using Welch's method with hardcoded parameters
 * 
 * Hardcoded parameters:
 *   - sampling_rate: 100 Hz
 *   - nperseg: 50
 *   - noverlap: 25
 *   - nfft: 64
 *   - window: Hann
 *   - detrend: constant (remove mean)
 *   - scaling: density (V^2/Hz)
 * 
 * Args:
 *     signal: Input signal array
 *     signal_length: Length of input signal
 *     freqs: Output array for frequency values (must be at least 33 elements)
 *     psd: Output array for PSD values (must be at least 33 elements)
 *     n_freqs: Output parameter - number of frequency bins (always 33)
 */
void compute_psd(double* signal, int signal_length, double* freqs, double* psd, int* n_freqs) {
    
    *n_freqs = N_BINS;
    
    // Initialize PSD accumulator
    double psd_accumulator[N_BINS];
    for (int i = 0; i < N_BINS; i++) {
        psd_accumulator[i] = 0.0;
    }
    
    // Generate Hann window
    double window[NPERSEG];
    generate_hann_window(window, NPERSEG);
    
    // Compute window normalization factor: U = (1/N) * sum(window^2)
    double U = 0.0;
    for (int i = 0; i < NPERSEG; i++) {
        U += window[i] * window[i];
    }
    U /= NPERSEG;
    
    // Create detrended signal copy (remove mean)
    double signal_copy[signal_length];
    double mean = 0.0;
    
    for (int i = 0; i < signal_length; i++) {
        signal_copy[i] = signal[i];
        mean += signal_copy[i];
    }
    mean /= signal_length;
    
    for (int i = 0; i < signal_length; i++) {
        signal_copy[i] -= mean;
    }
    
    // Calculate number of segments
    int nsegments = (signal_length - NOVERLAP) / STEP;
    
    // Working buffers
    double segment_windowed[NPERSEG];
    complex_t segment_padded[NFFT];
    complex_t dft_result[NFFT];
    
    // Process each segment
    for (int seg = 0; seg < nsegments; seg++) {
        int start_idx = seg * STEP;
        
        // Apply window to segment
        for (int i = 0; i < NPERSEG; i++) {
            segment_windowed[i] = signal_copy[start_idx + i] * window[i];
        }
        
        // Zero-pad: copy windowed segment and pad with zeros
        for (int i = 0; i < NPERSEG; i++) {
            segment_padded[i].real = segment_windowed[i];
            segment_padded[i].imag = 0.0;
        }
        for (int i = NPERSEG; i < NFFT; i++) {
            segment_padded[i].real = 0.0;
            segment_padded[i].imag = 0.0;
        }
        
        // Apply DFT
        dft(segment_padded, dft_result, NFFT);
        
        // Process positive frequencies only (one-sided spectrum)
        for (int k = 0; k < N_BINS; k++) {
            double mag_sq = complex_mag_sq(dft_result[k]);
            
            // Double the power for positive frequencies (except DC and Nyquist)
            if (k > 0 && k < NFFT / 2) {
                mag_sq *= 2.0;
            }
            
            psd_accumulator[k] += mag_sq;
        }
    }
    
    // Average across segments
    for (int k = 0; k < N_BINS; k++) {
        psd_accumulator[k] /= nsegments;
    }
    
    // Apply density scaling: 1 / (U * nfft * Δf)
    // where Δf = sampling_rate / nfft
    double freq_res = SAMPLING_RATE / NFFT;
    double scale = 1.0 / (U * NFFT * freq_res);
    
    for (int k = 0; k < N_BINS; k++) {
        psd[k] = psd_accumulator[k] * scale;
        freqs[k] = k * freq_res;
    }
}

// // Generate Hann window
// void generate_hann_window(double* window, int n) {
//     for (int i = 0; i < n; i++) {
//         window[i] = 0.5 * (1.0 - cos(2.0 * PI * i / (n - 1)));
//     }
// }

// // Simple DFT implementation (O(n²))
// void dft(complex_t* input, complex_t* output, int n) {
//     for (int k = 0; k < n; k++) {
//         output[k].real = 0.0;
//         output[k].imag = 0.0;
        
//         for (int j = 0; j < n; j++) {
//             double angle = -2.0 * PI * k * j / n;
//             double cos_val = cos(angle);
//             double sin_val = sin(angle);
            
//             output[k].real += input[j].real * cos_val - input[j].imag * sin_val;
//             output[k].imag += input[j].real * sin_val + input[j].imag * cos_val;
//         }
//     }
// }

// // Compute PSD using Welch's method
// void compute_psd(double* signal, int signal_length, double* freqs, double* psd, int* n_freqs) {
//     int hop_length = NPERSEG - NOVERLAP;
//     int n_segments = (signal_length - NOVERLAP) / hop_length;
    
//     // Calculate number of frequency bins (NFFT/2 + 1 for real signals)
//     *n_freqs = NFFT / 2 + 1;
    
//     // Initialize frequency bins
//     for (int i = 0; i < *n_freqs; i++) {
//         freqs[i] = i * SAMPLING_RATE / NFFT;
//     }
    
//     // Initialize PSD accumulator
//     for (int i = 0; i < *n_freqs; i++) {
//         psd[i] = 0.0;
//     }
    
//     // Apply detrending to entire signal first (like TensorFlow)
//     double signal_mean = 0.0;
//     for (int i = 0; i < signal_length; i++) {
//         signal_mean += signal[i];
//     }
//     signal_mean /= signal_length;
    
//     // Detrend the entire signal
//     for (int i = 0; i < signal_length; i++) {
//         signal[i] -= signal_mean;
//     }
    
//     // Generate Hann window
//     double hann_window[NPERSEG];
//     generate_hann_window(hann_window, NPERSEG);
    
//     // Calculate window energy (sum of squared window values)
//     double window_energy = 0.0;
//     for (int i = 0; i < NPERSEG; i++) {
//         window_energy += hann_window[i] * hann_window[i];
//     }
    
//     // Process each segment
//     for (int seg = 0; seg < n_segments; seg++) {
//         int start_idx = seg * hop_length;
        
//         // Extract segment and apply window (no additional detrending needed)
//         complex_t segment[NFFT];
//         for (int i = 0; i < NFFT; i++) {
//             if (i < NPERSEG) {
//                 segment[i].real = signal[start_idx + i] * hann_window[i];
//                 segment[i].imag = 0.0;
//             } else {
//                 // Zero-pad
//                 segment[i].real = 0.0;
//                 segment[i].imag = 0.0;
//             }
//         }
        
//         // Compute DFT
//         complex_t dft_output[NFFT];
//         dft(segment, dft_output, NFFT);
        
//         // Compute power spectrum and accumulate
//         // Apply moderate scaling to match FFT behavior
//         double scale_factor = sqrt(NFFT);  // Use sqrt(NFFT) for moderate scaling
//         for (int i = 0; i < *n_freqs; i++) {
//             double power = dft_output[i].real * dft_output[i].real + 
//                           dft_output[i].imag * dft_output[i].imag;
//             psd[i] += power * scale_factor;  // Apply moderate scaling
//         }
//     }
    
//     // Average across segments first (like TensorFlow)
//     for (int i = 0; i < *n_freqs; i++) {
//         psd[i] /= n_segments;
//     }
    
//     // Apply density scaling (like TensorFlow: psd_avg / (sampling_rate * window_energy))
//     double scale_factor = 1.0 / (SAMPLING_RATE * window_energy);
//     for (int i = 0; i < *n_freqs; i++) {
//         psd[i] *= scale_factor;
//     }
// }

// Calculate total power (sum of squared signal values - time domain energy)
double calc_total_power(double* signal, int n) {
    double total = 0.0;
    for (int i = 0; i < n; i++) {
        total += signal[i] * signal[i];
    }
    return total;
}

// Calculate proportion of PSD values above the mean
double calc_above_mean_density(double* psd, int n) {
    double mean = 0.0;
    for (int i = 0; i < n; i++) {
        mean += psd[i];
    }
    mean /= n;
    
    int above_count = 0;
    for (int i = 0; i < n; i++) {
        if (psd[i] > mean) {
            above_count++;
        }
    }
    
    return (double)above_count / n;
}

// Calculate mean of absolute PSD first-order differences
double calc_mean_abs_change(double* psd, int n) {
    if (n < 2) return 0.0;
    
    double total_abs_change = 0.0;
    for (int i = 1; i < n; i++) {
        double diff = psd[i] - psd[i-1];
        total_abs_change += fabs(diff);
    }
    
    return total_abs_change / (n - 1);
}

// Calculate mean of PSD first-order differences (signed)
double calc_mean_change(double* psd, int n) {
    if (n < 2) return 0.0;
    
    double total_change = 0.0;
    for (int i = 1; i < n; i++) {
        double diff = psd[i] - psd[i-1];
        total_change += diff;
    }
    
    return total_change / (n - 1);
}

// Calculate spectral centroid (weighted average frequency)
double calc_spectral_centroid(double* psd, double* freqs, int n) {
    double weighted_sum = 0.0;
    double total_power = 0.0;
    
    for (int i = 0; i < n; i++) {
        weighted_sum += freqs[i] * psd[i];
        total_power += psd[i];
    }
    
    if (total_power == 0.0) return 0.0;
    return weighted_sum / total_power;
}

// Get PSD value at or nearest to target frequency
double calc_psd_at_frequency(double* psd, double* freqs, int n, double target_freq) {
    int best_idx = 0;
    double min_diff = fabs(freqs[0] - target_freq);
    
    for (int i = 1; i < n; i++) {
        double diff = fabs(freqs[i] - target_freq);
        if (diff < min_diff) {
            min_diff = diff;
            best_idx = i;
        }
    }
    
    return psd[best_idx];
}

// Normalize a single feature value to [0, 1] using min-max normalization
double normalize_single_feature(double value, feature_stats_t* stats) {
    if (stats->range == 0.0) {
        // If range is zero, return 0.5 (middle of [0,1])
        return 0.5;
    }
    
    double normalized = (value - stats->min) / stats->range;
    
    // Clip to [0, 1] to handle numerical errors
    if (normalized < 0.0) normalized = 0.0;
    if (normalized > 1.0) normalized = 1.0;
    
    return normalized;
}

// Normalize all features in the array
void normalize_all_features(double* features, feature_stats_t* stats_array, int n_features) {
    for (int i = 0; i < n_features; i++) {
        features[i] = normalize_single_feature(features[i], &stats_array[i]);
    }
}

// Get feature statistics from training data (hardcoded from JSON)
// Features in alphabetical order: mean_change, psd_35hz, psd_40hz, psd_45hz, spectral_centroid, total_power
void get_feature_statistics(feature_stats_t* stats_array) {
    // mean_change (index 0)
    stats_array[0].min = -99955.7421875;
    stats_array[0].max = 2786.328369140625;
    stats_array[0].mean = -697.7705688476562;
    stats_array[0].std = 4226.8974609375;
    stats_array[0].range = 102742.0703125;
    stats_array[0].count = 3974;
    
    // psd_35hz (index 1)
    stats_array[1].min = 0.4829966127872467;
    stats_array[1].max = 298646.78125;
    stats_array[1].mean = 2361.765380859375;
    stats_array[1].std = 11707.154296875;
    stats_array[1].range = 298646.3125;
    stats_array[1].count = 3974;
    
    // psd_40hz (index 2)
    stats_array[2].min = 0.3288194239139557;
    stats_array[2].max = 61487.55078125;
    stats_array[2].mean = 993.2022094726562;
    stats_array[2].std = 3757.693603515625;
    stats_array[2].range = 61487.22265625;
    stats_array[2].count = 3974;
    
    // psd_45hz (index 3)
    stats_array[3].min = 0.23468634486198425;
    stats_array[3].max = 53115.98046875;
    stats_array[3].mean = 575.1829223632812;
    stats_array[3].std = 2195.572021484375;
    stats_array[3].range = 53115.74609375;
    stats_array[3].count = 3974;
    
    // spectral_centroid (index 4)
    stats_array[4].min = 0.5843378025420608;
    stats_array[4].max = 46.667872838668494;
    stats_array[4].mean = 17.55571547031415;
    stats_array[4].std = 7.55220006965818;
    stats_array[4].range = 46.083535036126435;
    stats_array[4].count = 3974;
    
    // total_power (index 5)
    stats_array[5].min = 612.0706176757812;
    stats_array[5].max = 8717437.0;
    stats_array[5].mean = 191002.09375;
    stats_array[5].std = 745345.75;
    stats_array[5].range = 8716825.0;
    stats_array[5].count = 3974;
}

#endif // PREPROCESSING_H
