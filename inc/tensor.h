/*
Author: Kwangryeol Park
Created: 2024.07.31
Copyright 2024

This file is a header file for tensor data structure.
The ndim is the number of dimensions of the tensor.
The shape is the array of the size of each dimension.
The data is the array of the flatten tensor data.
*/
#ifndef _TENSOR_H
#define _TENSOR_H

#include <stdint.h>

extern uint64_t tensor_global_data_memory;   // Global variable to store the total memory allocated by tensor (bytes)

typedef enum {
    TENSOR_INT16,
    TENSOR_INT32,
    TENSOR_INT64,
    TENSOR_FLOAT32,
    TENSOR_FLOAT64
} tensor_type_t;

typedef union {
    int16_t int16;
    int32_t int32;
    int64_t int64;
    float float32;
    double float64;
} tensor_data_t;

typedef struct {
    tensor_type_t type;
    uint32_t ndim;
    uint32_t num_elements;
    uint32_t *shape;
    uint32_t *transpose;    // Transpose index. The original index is the key, and the value is the new index.
    tensor_data_t *data;
} tensor_t;

uint32_t tensor_get_data_memory(tensor_t *tensor);
uint64_t tensor_get_global_data_memory();

// Create and free functions for each tensor type
tensor_t *tensor_create(tensor_type_t type, uint32_t ndim, uint32_t *shape);
void tensor_free(tensor_t *tensor);

// Set and get functions for each tensor type
void tensor_data_set(tensor_t *tensor, tensor_data_t *data);

// Fill with
void tensor_fill_with(tensor_t *tensor, tensor_data_t data);

// Allocate tensor data address.
tensor_t *tensor_alloc_data_addr(tensor_t *tensor, tensor_data_t *data_addr);

// Convert n-d index to 1-d index
// uint32_t tensor_convert_1d_index_to_1d_index(tensor_t *tensor, uint32_t i);
// uint32_t tensor_convert_2d_index_to_1d_index(tensor_t *tensor, uint32_t i, uint32_t j);
// uint32_t tensor_convert_3d_index_to_1d_index(tensor_t *tensor, uint32_t i, uint32_t j, uint32_t k);
// uint32_t tensor_convert_4d_index_to_1d_index(tensor_t *tensor, uint32_t i, uint32_t j, uint32_t k, uint32_t l);
// uint32_t tensor_convert_5d_index_to_1d_index(tensor_t *tensor, uint32_t i, uint32_t j, uint32_t k, uint32_t l, uint32_t m);
uint32_t tensor_convert_nd_to_1d_index(tensor_t *tensor, uint32_t *indices);

// Print
void tensor_print_data(tensor_t *tensor);
void tensor_print_shape(tensor_t *tensor);
void tensor_print_dim(tensor_t *tensor);
void tensor_print_data_memory(tensor_t *tensor);
void tensor_print_global_data_memory();

// Shape transformation
tensor_t *tensor_unsqueeze(tensor_t *tensor, uint32_t axis);
tensor_t *tensor_squeeze(tensor_t *tensor, uint32_t axis);
tensor_t *tensor_transpose(tensor_t *tensor, uint32_t axis1, uint32_t axis2);
#endif // _TENSOR_H