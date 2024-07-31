#include "op_linear.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "tensor.h"

#ifndef NULL
#define NULL 0
#endif

linear_t *linear_create(tensor_t *weight, tensor_t *bias) {
    // weight: 2D tensor    (out_features x in_features)
    // bias: 1D tensor      (out_features)
    // output: 2D tensor    (batch_size x out_features)
    
    // Check shape
    if (weight->ndim != 2) {
        printf("[%s][%s][%d] Error: weight tensor must be 2D tensor\r\n", __FILE__, __func__, __LINE__);
        return NULL;
    }
    if (bias != (tensor_t *) NULL) {
        if (bias->ndim != 1) {
            printf("[%s][%s][%d] Error: bias tensor must be 1D tensor\r\n", __FILE__, __func__, __LINE__);
            return NULL;
        }
        // Type check
        if (weight->type != bias->type) {
            printf("[%s][%s][%d] Error: weight and bias must have the same type\r\n", __FILE__, __func__, __LINE__);
            return NULL;
        }
    }

    // Allocate linear_t
    linear_t *linear = (linear_t *)malloc(sizeof(linear_t));
    linear->weight = weight;
    linear->bias = bias;

    return linear;
}
void linear_free(linear_t *linear, uint8_t deep) {
    // deep: 0 - free only linear_t, 1 - free linear_t and weight, bias
    if (deep != 0) {
        if (linear->weight != (tensor_t *) NULL) {
            tensor_free(linear->weight);
        }
        if (linear->bias != (tensor_t *) NULL) {
            tensor_free(linear->bias);
        }
    } else {
        linear->weight = (tensor_t *) NULL;
        linear->bias = (tensor_t *) NULL;
    }
    free(linear);
}

tensor_t *linear(tensor_t *input, linear_t *linear_weight) {
    // output = input * weight.T + bias
    // input: 2D tensor or 1D tensor    (batch_size x in_features)
    // weight: 2D tensor    (out_features x in_features)
    // bias: 1D tensor      (out_features)
    // output: 2D tensor    (batch_size x out_features)
    
    // Check shape
    tensor_t *weight = linear_weight->weight;
    tensor_t *bias = linear_weight->bias;

    if (input->ndim == 1)   input = tensor_unsqueeze(input, 0);
    if (input->ndim != 2) {
        printf("[%s][%s][%d] Error: input tensor must be 2D tensor\r\n", __FILE__, __func__, __LINE__);
        return NULL;
    }
    if (weight->ndim != 2) {
        printf("[%s][%s][%d] Error: weight tensor must be 2D tensor\r\n", __FILE__, __func__, __LINE__);
        return NULL;
    }
    if (input->shape[1] != weight->shape[1]) {
        printf("[%s][%s][%d] Error: input shape[1] must be equal to weight shape[1]\r\n", __FILE__, __func__, __LINE__);
        return NULL;
    }
    if (bias != (tensor_t *) NULL) {
        if (bias->ndim != 1) {
            printf("[%s][%s][%d] Error: bias tensor must be 1D tensor\r\n", __FILE__, __func__, __LINE__);
            return NULL;
        }
        if (weight->shape[0] != bias->shape[0]) {
            printf("[%s][%s][%d] Error: weight shape[0] must be equal to bias shape[0]\r\n", __FILE__, __func__, __LINE__);
            return NULL;
        }
    }

    // Type check
    if (input->type != weight->type || input->type != bias->type) {
        printf("[%s][%s][%d] Error: input, weight, and bias must have the same type\r\n", __FILE__, __func__, __LINE__);
        return NULL;
    }

    // Allocate output tensor
    uint32_t shape[] = {input->shape[0], weight->shape[0]};
    tensor_t *output = tensor_create(input->type, 2, shape, (void *)0);

    // Calculate
    const tensor_data_t *input_data = input->data;
    const tensor_data_t *weight_data = weight->data;

    switch (input->type) {
        case TENSOR_INT64:
        for (int i = 0; i < input->shape[0]; i++) {
            for (int j = 0; j < weight->shape[0]; j++) {
                tensor_data_t sum = (tensor_data_t){.int64 = 0};
                for (int k = 0; k < input->shape[1]; k++) {
                    sum.int64 += input_data[tensor_convert_nd_to_1d_index(input, (uint32_t[]){i, k})].int64 * weight_data[tensor_convert_nd_to_1d_index(weight, (uint32_t[]){j, k})].int64;
                }
                if (bias != (tensor_t *) NULL) {
                    sum.int64 += bias->data[j].int64;
                }
                output->data[tensor_convert_nd_to_1d_index(output, (uint32_t[]){i, j})] = sum;
            }
        }
        break;
        case TENSOR_FLOAT32:
        for (int i = 0; i < input->shape[0]; i++) {
            for (int j = 0; j < weight->shape[0]; j++) {
                tensor_data_t sum = (tensor_data_t){.float32 = 0};
                for (int k = 0; k < input->shape[1]; k++) {
                    sum.float32 += input_data[tensor_convert_nd_to_1d_index(input, (uint32_t[]){i, k})].float32 * weight_data[tensor_convert_nd_to_1d_index(weight, (uint32_t[]){j, k})].float32;
                }
                if (bias != (tensor_t *) NULL) {
                    sum.float32 += bias->data[j].float32;
                }
                output->data[tensor_convert_nd_to_1d_index(output, (uint32_t[]){i, j})] = sum;
            }
        }
        break;
        
        case TENSOR_INT32:
            printf("[%s][%s][%d] Error: Un-supported tensor type. Supported tensor types are int64 or float32. Current: [int32]\r\n", __FILE__, __func__, __LINE__);
            return NULL;
        case TENSOR_INT16:
            printf("[%s][%s][%d] Error: Un-supported tensor type. Supported tensor types are int64 or float32. Current: [int16]\r\n", __FILE__, __func__, __LINE__);
            return NULL;
        case TENSOR_FLOAT64:
            printf("[%s][%s][%d] Error: Un-supported tensor type. Supported tensor types are int64 or float32. Current: [float64]\r\n", __FILE__, __func__, __LINE__);
            return NULL;
        default:
            printf("[%s][%s][%d] Error: Unknown tensor type. Supported tensor types are int64 or float32\r\n", __FILE__, __func__, __LINE__);
            return NULL;
    }

    return output;
}
