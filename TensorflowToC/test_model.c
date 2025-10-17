// TensorflowToC/test_model.c
// Build:
//   gcc -O3 -ffast-math -std=c11 -I TensorflowToC -I TensorflowToC/test_model \
//       TensorflowToC/test_model.c TensorflowToC/test_model/*.c TensorflowToC/*.c \
//       -o TensorflowToC/test_model/run_c_model -lm
//
// Run:
//   ./TensorflowToC/test_model/run_c_model input.bin c_output.bin <INPUT_SIZE> <OUTPUT_SIZE>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "model_inference.h"  // must declare: void model_inference(const float* in, float* out);

static int read_floats(const char* path, float* buf, size_t count) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERR: cannot open %s for read\n", path);
        return -1;
    }
    size_t n = fread(buf, sizeof(float), count, f);
    fclose(f);
    if (n != count) {
        fprintf(stderr, "ERR: expected %zu floats, read %zu\n", count, n);
        return -1;
    }
    return 0;
}

static int write_floats(const char* path, const float* buf, size_t count) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "ERR: cannot open %s for write\n", path);
        return -1;
    }
    size_t n = fwrite(buf, sizeof(float), count, f);
    fclose(f);
    if (n != count) {
        fprintf(stderr, "ERR: expected to write %zu floats, wrote %zu\n", count, n);
        return -1;
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <input.bin> <output.bin> <INPUT_SIZE> <OUTPUT_SIZE>\n", argv[0]);
        return 2;
    }

    const char* in_path  = argv[1];
    const char* out_path = argv[2];
    char* endp = NULL;

    size_t input_size  = (size_t) strtoull(argv[3], &endp, 10);
    if (endp == argv[3] || input_size == 0) {
        fprintf(stderr, "ERR: bad INPUT_SIZE '%s'\n", argv[3]);
        return 2;
    }
    size_t output_size = (size_t) strtoull(argv[4], &endp, 10);
    if (endp == argv[4] || output_size == 0) {
        fprintf(stderr, "ERR: bad OUTPUT_SIZE '%s'\n", argv[4]);
        return 2;
    }

    float* input  = (float*) aligned_alloc(32, input_size  * sizeof(float));
    float* output = (float*) aligned_alloc(32, output_size * sizeof(float));
    if (!input || !output) {
        fprintf(stderr, "ERR: malloc failed\n");
        free(input); free(output);
        return 1;
    }

    if (read_floats(in_path, input, input_size) != 0) {
        free(input); free(output);
        return 1;
    }

    // Run the generated C model
    model_infer(input, output);

    if (write_floats(out_path, output, output_size) != 0) {
        free(input); free(output);
        return 1;
    }

    // Optional sanity print of first few outputs
    size_t show = output_size < 8 ? output_size : 8;
    fprintf(stdout, "C output first %zu vals:", show);
    for (size_t i = 0; i < show; ++i) fprintf(stdout, " %g", output[i]);
    fprintf(stdout, "\n");

    free(input);
    free(output);
    return 0;
}
