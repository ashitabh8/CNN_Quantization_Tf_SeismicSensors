#ifndef QUEUE_H
#define QUEUE_H

#include <stdint.h>

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
void queue_init(Queue* q);

/**
 * Push a single value into the queue
 * @param q Pointer to queue structure
 * @param value Float value to add
 * @return 1 on success, 0 if queue is full
 */
int queue_push(Queue* q, float value);

/**
 * Check if queue is full (has QUEUE_SIZE elements)
 * @param q Pointer to queue structure
 * @return 1 if full, 0 otherwise
 */
int queue_is_full(Queue* q);

/**
 * Check if queue is empty
 * @param q Pointer to queue structure
 * @return 1 if empty, 0 otherwise
 */
int queue_is_empty(Queue* q);

/**
 * Get current number of elements in queue
 * @param q Pointer to queue structure
 * @return Number of elements (0 to QUEUE_SIZE)
 */
int queue_count(Queue* q);

/**
 * Get pointer to the complete array (only valid when queue is full)
 * This returns the data array in the order it was inserted
 * @param q Pointer to queue structure
 * @return Pointer to data array if full, NULL otherwise
 */
float* queue_get_array(Queue* q);

/**
 * Clear the queue for next iteration
 * @param q Pointer to queue structure
 */
void queue_clear(Queue* q);

/**
 * Get a single value from the queue (FIFO order)
 * Only use this if you need to pop individual values
 * @param q Pointer to queue structure
 * @param value Pointer to store the popped value
 * @return 1 on success, 0 if queue is empty
 */
int queue_pop(Queue* q, float* value);

#endif // QUEUE_H
