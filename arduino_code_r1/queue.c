#include "queue.h"
#include <stddef.h>

void queue_init(Queue* q) {
    q->head = 0;
    q->count = 0;
    // Initialize data array to zeros (optional, but good practice)
    for (int i = 0; i < QUEUE_SIZE; i++) {
        q->data[i] = 0.0f;
    }
}

int queue_push(Queue* q, float value) {
    if (q->count >= QUEUE_SIZE) {
        return 0;  // Queue is full
    }
    
    q->data[q->head] = value;
    q->head = (q->head + 1) % QUEUE_SIZE;
    q->count++;
    return 1;  // Success
}

int queue_is_full(Queue* q) {
    return (q->count >= QUEUE_SIZE) ? 1 : 0;
}

int queue_is_empty(Queue* q) {
    return (q->count == 0) ? 1 : 0;
}

int queue_count(Queue* q) {
    return q->count;
}

float* queue_get_array(Queue* q) {
    if (q->count < QUEUE_SIZE) {
        return NULL;  // Queue not full yet
    }
    return q->data;
}

void queue_clear(Queue* q) {
    q->head = 0;
    q->count = 0;
    // Optionally clear data array
    for (int i = 0; i < QUEUE_SIZE; i++) {
        q->data[i] = 0.0f;
    }
}

int queue_pop(Queue* q, float* value) {
    if (q->count == 0) {
        return 0;  // Queue is empty
    }
    
    // Calculate the tail position (oldest element)
    int tail = (q->head - q->count + QUEUE_SIZE) % QUEUE_SIZE;
    *value = q->data[tail];
    q->count--;
    return 1;  // Success
}
