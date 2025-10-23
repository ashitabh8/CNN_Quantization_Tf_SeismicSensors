#ifndef QUEUE_ARDUINO_H
#define QUEUE_ARDUINO_H

// Queue size - matches preprocessing window size
#define QUEUE_SIZE 200

// Queue structure for circular buffer
typedef struct {
    float data[QUEUE_SIZE];  // Storage array for 200 float values
    int head;                // Current write position (circular)
    int count;               // Current number of elements in queue
} Queue;

/**
 * Initialize an empty queue
 * @param q Pointer to queue structure
 */
void queue_init(Queue* q) {
    q->head = 0;
    q->count = 0;
    // Initialize data array to zeros (optional, but good practice)
    for (int i = 0; i < QUEUE_SIZE; i++) {
        q->data[i] = 0.0f;
    }
}

/**
 * Push a single value into the queue
 * @param q Pointer to queue structure
 * @param value Float value to add
 * @return 1 on success, 0 if queue is full
 */
int queue_push(Queue* q, float value) {
    if (q->count >= QUEUE_SIZE) {
        return 0;  // Queue is full
    }
    
    q->data[q->head] = value;
    q->head = (q->head + 1) % QUEUE_SIZE;
    q->count++;
    return 1;  // Success
}

/**
 * Check if queue is full (has QUEUE_SIZE elements)
 * @param q Pointer to queue structure
 * @return 1 if full, 0 otherwise
 */
int queue_is_full(Queue* q) {
    return (q->count >= QUEUE_SIZE) ? 1 : 0;
}

/**
 * Check if queue is empty
 * @param q Pointer to queue structure
 * @return 1 if empty, 0 otherwise
 */
int queue_is_empty(Queue* q) {
    return (q->count == 0) ? 1 : 0;
}

/**
 * Get current number of elements in queue
 * @param q Pointer to queue structure
 * @return Number of elements (0 to QUEUE_SIZE)
 */
int queue_count(Queue* q) {
    return q->count;
}

/**
 * Get pointer to the complete array (only valid when queue is full)
 * This returns the data array in the order it was inserted
 * @param q Pointer to queue structure
 * @return Pointer to data array if full, NULL otherwise
 */
float* queue_get_array(Queue* q) {
    if (q->count < QUEUE_SIZE) {
        return NULL;  // Queue not full yet
    }
    return q->data;
}

/**
 * Clear the queue for next iteration
 * @param q Pointer to queue structure
 */
void queue_clear(Queue* q) {
    q->head = 0;
    q->count = 0;
    // Optionally clear data array
    for (int i = 0; i < QUEUE_SIZE; i++) {
        q->data[i] = 0.0f;
    }
}

/**
 * Get a single value from the queue (FIFO order)
 * Only use this if you need to pop individual values
 * @param q Pointer to queue structure
 * @param value Pointer to store the popped value
 * @return 1 on success, 0 if queue is empty
 */
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

#endif // QUEUE_ARDUINO_H
