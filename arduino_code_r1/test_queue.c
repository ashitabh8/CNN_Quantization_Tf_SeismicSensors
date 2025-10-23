#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "queue.h"
#include "test_data.h"

#define TOLERANCE 1e-6f

// Test data arrays (from test_data.h)
extern const float test_data_1[200];
extern const float test_data_2[200];
extern const float test_data_3[200];
extern const float test_data_4[200];
extern const float test_data_5[200];

// Function to compare two float arrays
int compare_arrays(const float* arr1, const float* arr2, int size, const char* test_name) {
    int errors = 0;
    float max_diff = 0.0f;
    
    for (int i = 0; i < size; i++) {
        float diff = fabsf(arr1[i] - arr2[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
        if (diff > TOLERANCE) {
            if (errors < 5) {  // Only print first 5 errors
                printf("  Error at index %d: expected %.6f, got %.6f (diff: %.6f)\n", 
                       i, arr2[i], arr1[i], diff);
            }
            errors++;
        }
    }
    
    if (errors > 0) {
        printf("  %s: %d errors found, max difference: %.6f\n", test_name, errors, max_diff);
    } else {
        printf("  %s: PASSED (max difference: %.6f)\n", test_name, max_diff);
    }
    
    return errors;
}

// Test queue with a single dataset
int test_queue_with_data(const float* input_data, int test_num) {
    printf("\n=== Testing Queue with Dataset %d ===\n", test_num);
    
    Queue q;
    queue_init(&q);
    
    // Test 1: Push all 200 values one by one
    printf("Test 1: Pushing 200 values...\n");
    int push_success = 0;
    for (int i = 0; i < 200; i++) {
        if (queue_push(&q, input_data[i])) {
            push_success++;
        } else {
            printf("  ERROR: Failed to push value at index %d\n", i);
        }
    }
    
    printf("  Pushed %d/200 values successfully\n", push_success);
    
    // Test 2: Verify queue is full
    printf("Test 2: Checking if queue is full...\n");
    if (queue_is_full(&q)) {
        printf("  PASSED: Queue is full\n");
    } else {
        printf("  ERROR: Queue should be full but isn't\n");
        return 1;
    }
    
    // Test 3: Verify count is 200
    printf("Test 3: Checking queue count...\n");
    int count = queue_count(&q);
    if (count == 200) {
        printf("  PASSED: Queue count is %d\n", count);
    } else {
        printf("  ERROR: Expected count 200, got %d\n", count);
        return 1;
    }
    
    // Test 4: Get array and verify data integrity
    printf("Test 4: Verifying data integrity...\n");
    float* queue_data = queue_get_array(&q);
    if (queue_data == NULL) {
        printf("  ERROR: queue_get_array returned NULL\n");
        return 1;
    }
    
    int data_errors = compare_arrays(queue_data, input_data, 200, "Data integrity");
    if (data_errors > 0) {
        printf("  ERROR: Data integrity check failed with %d errors\n", data_errors);
        return 1;
    }
    
    // Test 5: Test individual pop operations
    printf("Test 5: Testing individual pop operations...\n");
    Queue q_copy = q;  // Make a copy for pop testing
    int pop_success = 0;
    for (int i = 0; i < 200; i++) {
        float popped_value;
        if (queue_pop(&q_copy, &popped_value)) {
            if (fabsf(popped_value - input_data[i]) < TOLERANCE) {
                pop_success++;
            } else {
                printf("  ERROR: Pop value mismatch at index %d\n", i);
            }
        } else {
            printf("  ERROR: Failed to pop value at index %d\n", i);
        }
    }
    printf("  Successfully popped %d/200 values correctly\n", pop_success);
    
    // Test 6: Clear queue
    printf("Test 6: Testing queue clear...\n");
    queue_clear(&q);
    if (queue_is_empty(&q) && queue_count(&q) == 0) {
        printf("  PASSED: Queue cleared successfully\n");
    } else {
        printf("  ERROR: Queue clear failed\n");
        return 1;
    }
    
    printf("=== Dataset %d: ALL TESTS PASSED ===\n", test_num);
    return 0;
}

// Test edge cases
int test_edge_cases() {
    printf("\n=== Testing Edge Cases ===\n");
    
    Queue q;
    queue_init(&q);
    
    // Test 1: Push to empty queue
    printf("Test 1: Push to empty queue...\n");
    if (queue_push(&q, 1.0f) && queue_count(&q) == 1) {
        printf("  PASSED: Push to empty queue\n");
    } else {
        printf("  ERROR: Push to empty queue failed\n");
        return 1;
    }
    
    // Test 2: Pop from queue with one element
    printf("Test 2: Pop from queue with one element...\n");
    float value;
    if (queue_pop(&q, &value) && fabsf(value - 1.0f) < TOLERANCE && queue_is_empty(&q)) {
        printf("  PASSED: Pop from single-element queue\n");
    } else {
        printf("  ERROR: Pop from single-element queue failed\n");
        return 1;
    }
    
    // Test 3: Pop from empty queue
    printf("Test 3: Pop from empty queue...\n");
    if (!queue_pop(&q, &value)) {
        printf("  PASSED: Pop from empty queue correctly failed\n");
    } else {
        printf("  ERROR: Pop from empty queue should have failed\n");
        return 1;
    }
    
    // Test 4: Push beyond capacity
    printf("Test 4: Push beyond capacity...\n");
    queue_clear(&q);
    for (int i = 0; i < QUEUE_SIZE; i++) {
        queue_push(&q, (float)i);
    }
    if (!queue_push(&q, 999.0f)) {
        printf("  PASSED: Push beyond capacity correctly failed\n");
    } else {
        printf("  ERROR: Push beyond capacity should have failed\n");
        return 1;
    }
    
    printf("=== Edge Cases: ALL TESTS PASSED ===\n");
    return 0;
}

int main() {
    printf("Arduino Queue System Test\n");
    printf("========================\n");
    printf("Testing circular queue with size %d\n", QUEUE_SIZE);
    printf("Using real nissan vehicle data from training dataset\n");
    
    int total_errors = 0;
    
    // Test with all 5 datasets
    const float* test_datasets[] = {
        test_data_1, test_data_2, test_data_3, test_data_4, test_data_5
    };
    
    for (int i = 0; i < NUM_TEST_SAMPLES; i++) {
        total_errors += test_queue_with_data(test_datasets[i], i + 1);
    }
    
    // Test edge cases
    total_errors += test_edge_cases();
    
    printf("\n========================\n");
    printf("FINAL RESULTS\n");
    printf("========================\n");
    if (total_errors == 0) {
        printf("ALL TESTS PASSED! Queue system is working correctly.\n");
        printf("Ready for Arduino deployment.\n");
    } else {
        printf("FAILED: %d test errors found.\n", total_errors);
    }
    
    return total_errors;
}
