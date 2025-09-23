#include "model_inference.h"

int main() {
    // Import model geenrate random sample of 2.7.256 shape and get output
    float input[MODEL_INPUT_SIZE] = {0};
    float output[4] = {0};
    model_infer(input, output);
    return 0;
}