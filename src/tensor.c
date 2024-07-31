#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef SWAP_int32_t
#define SWAP_int32_t(a, b) {int tmp = a; a = b; b = tmp;}
#endif

uint64_t tensor_global_data_memory = 0;  // Global variable to store the total memory allocated by tensor (bytes)

// Create and free functions for each tensor type
uint32_t tensor_get_data_memory(tensor_t *tensor) {
    uint32_t memory = 0;
    switch (tensor->type) {
        case TENSOR_INT16:
            memory = tensor->num_elements * sizeof(int16_t);
            break;
        case TENSOR_INT32:
            memory = tensor->num_elements * sizeof(int32_t);
            break;
        case TENSOR_INT64:
            memory = tensor->num_elements * sizeof(int64_t);
            break;
        case TENSOR_FLOAT32:
            memory = tensor->num_elements * sizeof(float);
            break;
        case TENSOR_FLOAT64:
            memory = tensor->num_elements * sizeof(double);
            break;
        default:
            printf(">> [%s][%s][%d] Error: Un-supported tensor type\r\n", __FILE__, __func__, __LINE__);
    }
    return memory;
}

uint64_t tensor_get_global_data_memory() {
    return tensor_global_data_memory;
}

tensor_t *tensor_create(tensor_type_t type, uint32_t ndim, uint32_t *shape) {
    tensor_t *tensor = (tensor_t *)malloc(sizeof(tensor_t));
    tensor->type = type;
    tensor->ndim = ndim;
    tensor->shape = (uint32_t *)malloc(ndim * sizeof(uint32_t));
    memcpy(tensor->shape, shape, ndim * sizeof(uint32_t));
    tensor->transpose = (uint32_t *)malloc(ndim * sizeof(uint32_t));
    for (int i = 0; i < ndim; i++)  tensor->transpose[i] = i;
    tensor->num_elements = 1;
    for (int i = 0; i < ndim; i++)  tensor->num_elements *= shape[i];
    tensor->data = (tensor_data_t *)malloc(tensor->num_elements * sizeof(tensor_data_t));
    tensor_global_data_memory += (uint64_t) tensor_get_data_memory(tensor);
    return tensor;
}

void tensor_free(tensor_t *tensor) {
    if (tensor_global_data_memory < (uint64_t) tensor_get_data_memory(tensor)) {
        printf(">> [%s][%s][%d] Error: tensor_global_data_memory is less than tensor memory\r\n", __FILE__, __func__, __LINE__);
        return;
    }
    tensor_global_data_memory -= (uint64_t) tensor_get_data_memory(tensor);
    free(tensor->shape);
    free(tensor->data);
    free(tensor);
}

// Set and get functions for each tensor type
void tensor_data_set(tensor_t *tensor, tensor_data_t *data) {
    memcpy(tensor->data, data, tensor->num_elements * sizeof(tensor_data_t));
}

// Fill with
void tensor_fill_with(tensor_t *tensor, tensor_data_t data) {
    for (int i = 0; i < tensor->num_elements; i++) {
        tensor->data[i] = data;
    }
}

// Allocate tensor data address.
// This function is useful when you already have a memory address for the tensor data.
// ex) weight tensor stored in the external memory
tensor_t *tensor_alloc_data_addr(tensor_t *tensor, tensor_data_t *data_addr) {
    tensor->data = data_addr;
    return tensor;
}

// Convert n-d index to 1-d index
// uint32_t tensor_convert_1d_index_to_1d_index(tensor_t *tensor, uint32_t i) {
//     if (tensor->ndim != 1) {
//         printf("Error: tensor ndim is not 1\r\n");
//         return -1;
//     }
//     if (i >= tensor->shape[0]) {
//         printf("Error: index is out of range\r\n");
//         return -1;
//     }
//     return i;
// }
// uint32_t tensor_convert_2d_index_to_1d_index(tensor_t *tensor, uint32_t i, uint32_t j) {
//     if (tensor->ndim != 2) {
//         printf(">> [%s][%s][%d] Error: tensor ndim is not 2\r\n", __FILE__, __func__, __LINE__);
//         return -1;
//     }
//     uint32_t *transpose = tensor->transpose;
//     if (i >= tensor->shape[0] || j >= tensor->shape[1]) {
//         printf(">> [%s][%s][%d] Error: index is out of range\r\n", __FILE__, __func__, __LINE__);
//         return -1;
//     }
//     // Map the original index to the new index
//     if (transpose[0] == 0 && transpose[1] == 1) return i * tensor->shape[1] + j;
//     else    return j * tensor->shape[0] + i;
// }
// uint32_t tensor_convert_3d_index_to_1d_index(tensor_t *tensor, uint32_t i, uint32_t j, uint32_t k) {
//     if (tensor->ndim != 3) {
//         printf("Error: tensor ndim is not 3\r\n");
//         return -1;
//     }
//     uint32_t *transpose = tensor->transpose;
//     if (i >= tensor->shape[transpose[0]] || j >= tensor->shape[transpose[1]] || k >= tensor->shape[transpose[2]]) {
//         printf(">> [%s][%s][%d] Error: index is out of range\r\n", __FILE__, __func__, __LINE__);
//         return -1;
//     }
//     uint32_t indics[3] = {i, j, k};
//     for (int i = 0; i < 3; i++) {
//         for (int j = i + 1; j < 3; j++) {
//             if (transpose[i] > transpose[j]) {
//                 SWAP_int32_t(indics[i], indics[j]);
//             }
//         }
//     }
//     return indics[0] * tensor->shape[transpose[1]] * tensor->shape[transpose[2]] + indics[1] * tensor->shape[transpose[2]] + indics[2];
// }
// uint32_t tensor_convert_4d_index_to_1d_index(tensor_t *tensor, uint32_t i, uint32_t j, uint32_t k, uint32_t l) {
//     if (tensor->ndim != 4) {
//         printf("Error: tensor ndim is not 4\r\n");
//         return -1;
//     }
//     uint32_t *transpose = tensor->transpose;
//     if (i >= tensor->shape[transpose[0]] || j >= tensor->shape[transpose[1]] || k >= tensor->shape[transpose[2]] || l >= tensor->shape[transpose[3]]) {
//         printf(">> [%s][%s][%d] Error: index is out of range\r\n", __FILE__, __func__, __LINE__);
//         return -1;
//     }
//     uint32_t indics[4] = {i, j, k, l};
//     for (int i = 0; i < 4; i++) {
//         for (int j = i + 1; j < 4; j++) {
//             if (transpose[i] > transpose[j]) {
//                 SWAP_int32_t(indics[i], indics[j]);
//             }
//         }
//     }
//     return indics[0] * tensor->shape[transpose[1]] * tensor->shape[transpose[2]] * tensor->shape[transpose[3]] + indics[1] * tensor->shape[transpose[2]] * tensor->shape[transpose[3]] + indics[2] * tensor->shape[transpose[3]] + indics[3];
// }
// uint32_t tensor_convert_5d_index_to_1d_index(tensor_t *tensor, uint32_t i, uint32_t j, uint32_t k, uint32_t l, uint32_t m) {
//     if (tensor->ndim != 5) {
//         printf("Error: tensor ndim is not 5\r\n");
//         return -1;
//     }
//     uint32_t *transpose = tensor->transpose;
//     if (i >= tensor->shape[transpose[0]] || j >= tensor->shape[transpose[1]] || k >= tensor->shape[transpose[2]] || l >= tensor->shape[transpose[3]] || m >= tensor->shape[transpose[4]]) {
//         printf(">> [%s][%s][%d] Error: index is out of range\r\n", __FILE__, __func__, __LINE__);
//         return -1;
//     }
//     uint32_t indics[5] = {i, j, k, l, m};
//     for (int i = 0; i < 5; i++) {
//         for (int j = i + 1; j < 5; j++) {
//             if (transpose[i] > transpose[j]) {
//                 SWAP_int32_t(indics[i], indics[j]);
//             }
//         }
//     }
//     return indics[0] * tensor->shape[transpose[1]] * tensor->shape[transpose[2]] * tensor->shape[transpose[3]] * tensor->shape[transpose[4]] + indics[1] * tensor->shape[transpose[2]] * tensor->shape[transpose[3]] * tensor->shape[transpose[4]] + indics[2] * tensor->shape[transpose[3]] * tensor->shape[transpose[4]] + indics[3] * tensor->shape[transpose[4]] + indics[4];
// }
uint32_t tensor_convert_nd_to_1d_index(tensor_t *tensor, uint32_t *indics) {
    uint32_t ndim = tensor->ndim;
    uint32_t *transpose = tensor->transpose;

    for (int i = 0; i < ndim; i++) {
        if (indics[i] >= tensor->shape[i]) {
            printf("[%s][%s][%d] Error: index is out of range\r\n", __FILE__, __func__, __LINE__);
            return -1;
        }
    }

    uint32_t indics_copy[ndim];
    memcpy(indics_copy, indics, ndim * sizeof(uint32_t));
    for (int i = 0; i < ndim; i++) {
        for (int j = i + 1; j < ndim; j++) {
            if (transpose[i] > transpose[j]) {
                SWAP_int32_t(indics_copy[i], indics_copy[j]);
            }
        }
    }

    uint32_t index = 0;
    uint32_t multiplier = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        index += indics_copy[i] * multiplier;
        multiplier *= tensor->shape[transpose[i]];
    }
    return index;
}

// Print
void tensor_print_data(tensor_t *tensor) {
    uint32_t num_elements = tensor->num_elements;
    printf(">> tensor data: [");
    char *type_str;
    switch (tensor->type) {
        case TENSOR_INT64:
            type_str = "%d, ";
            if (num_elements <= 20) {
                for (int i = 0; i < num_elements; i++) {
                    printf(type_str, tensor->data[i].int64);
                }
            } else {
                for (int i = 0; i < 10; i++) {
                    printf(type_str, tensor->data[i].int64);
                }
                printf("... ");
                for (int i = num_elements - 10; i < num_elements; i++) {
                    printf(type_str, tensor->data[i].int64);
                }
            }
            break;
        case TENSOR_FLOAT32:
            type_str = "%f, ";
            if (num_elements <= 20) {
                for (int i = 0; i < num_elements; i++) {
                    printf(type_str, tensor->data[i].float32);
                }
            } else {
                for (int i = 0; i < 10; i++) {
                    printf(type_str, tensor->data[i].float32);
                }
                printf("... ");
                for (int i = num_elements - 10; i < num_elements; i++) {
                    printf(type_str, tensor->data[i].float32);
                }
            }
            break;
        default:
            printf(">> [%s][%s][%d] Error: Un-supported tensor type\r\n", __FILE__, __func__, __LINE__);
    }
    printf("]\r\n");
}
void tensor_print_shape(tensor_t *tensor) {
    printf(">> tensor shape: (");
    for (int i = 0; i < tensor->ndim; i++) {
        printf("%d, ", tensor->shape[i]);
    }
    printf(")\r\n");
}
void tensor_print_dim(tensor_t *tensor) {
    printf(">> tensor dim: %d\r\n", tensor->ndim);
}
void tensor_print_data_memory(tensor_t *tensor) {
    printf(">> tensor data memory: %d bytes\r\n", tensor_get_data_memory(tensor));
}
void tensor_print_global_data_memory() {
    printf(">> tensor global data memory: %ld bytes\r\n", tensor_global_data_memory);
}


// Shape transformation
tensor_t *tensor_unsqueeze(tensor_t *tensor, uint32_t axis) {
    if (axis > tensor->ndim) {
        printf("[%s][%s][%d] axis is out of range\r\n", __FILE__, __func__, __LINE__);
        return NULL;
    }
    // Reallocate shape using realloc
    uint32_t *old_shape = (uint32_t *)malloc(tensor->ndim * sizeof(uint32_t));
    memcpy(old_shape, tensor->shape, tensor->ndim * sizeof(uint32_t));
    tensor->shape = (uint32_t *)realloc(tensor->shape, (tensor->ndim + 1) * sizeof(uint32_t));
    for (int i = tensor->ndim; i > axis; i--) {
        tensor->shape[i] = old_shape[i - 1];
    }

    tensor->shape[axis] = 1;
    tensor->ndim++;

    free(old_shape);
    return tensor;
}

// Squeeze
tensor_t *tensor_squeeze(tensor_t *tensor, uint32_t axis) {
    if (axis >= tensor->ndim) {
        printf("[%s][%s][%d] axis is out of range\r\n", __FILE__, __func__, __LINE__);
        return NULL;
    }
    if (tensor->shape[axis] != 1) {
        printf("[%s][%s][%d] The shape at the axis is not 1\r\n", __FILE__, __func__, __LINE__);
        return NULL;
    }
    // Reallocate shape using realloc
    uint32_t *old_shape = (uint32_t *)malloc(tensor->ndim * sizeof(uint32_t));
    memcpy(old_shape, tensor->shape, tensor->ndim * sizeof(uint32_t));
    tensor->shape = (uint32_t *)realloc(tensor->shape, (tensor->ndim - 1) * sizeof(uint32_t));
    for (int i = axis; i < tensor->ndim - 1; i++) {
        tensor->shape[i] = old_shape[i + 1];
    }

    tensor->ndim--;

    free(old_shape);
    return tensor;
}

// Tranpose.
// Tranpose operation does not change the location of data in memory.
// Instead, it changes the order of index to access the data.
// The information of tranposing is stored in tensor->transpose.
// The actual compute of index in done in tensor_convert_nd_to_1d_index function.
tensor_t *tensor_transpose(tensor_t *tensor, uint32_t axis1, uint32_t axis2) {
    if (axis1 >= tensor->ndim || axis2 >= tensor->ndim) {
        printf("[%s][%s][%d] axis is out of range\r\n", __FILE__, __func__, __LINE__);
        return NULL;
    }
    uint32_t tmp = tensor->shape[axis1];
    tensor->shape[axis1] = tensor->shape[axis2];
    tensor->shape[axis2] = tmp;

    tmp = tensor->transpose[axis1];
    tensor->transpose[axis1] = tensor->transpose[axis2];
    tensor->transpose[axis2] = tmp;
    return tensor;
}

tensor_t *tensor_reshape(tensor_t *tensor, uint32_t ndim, uint32_t *shape) {
    uint32_t num_elements = 1;
    for (int i = 0; i < ndim; i++) {
        num_elements *= shape[i];
    }
    if (num_elements != tensor->num_elements) {
        printf("[%s][%s][%d] Error: The number of elements is not matched\r\n", __FILE__, __func__, __LINE__);
        return NULL;
    }
    tensor->ndim = ndim;
    tensor->shape = (uint32_t *)realloc(tensor->shape, ndim * sizeof(uint32_t));
    memcpy(tensor->shape, shape, ndim * sizeof(uint32_t));
    return tensor;
}