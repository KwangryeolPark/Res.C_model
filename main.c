/*
    Author: Kwangryeol Park
    Created: 2024.07.31
    
    PyTorch의 연산방법을 따름.
    // output = input * weight.T + bias
    // input: 2D tensor or 1D tensor    (batch_size x in_features)
    // weight: 2D tensor    (out_features x in_features)
    // bias: 1D tensor      (out_features)
    // output: 2D tensor    (batch_size x out_features)
*/

#include <stdio.h>
#include <stdint.h>
#include "tensor.h"
#include "op_linear.h"

int main() {
    printf(">> Demo: Create and free a tensor\r\n");
    tensor_t *input = tensor_create(TENSOR_FLOAT32, 2, (uint32_t[]){3, 5});
    tensor_t *weight = tensor_create(TENSOR_FLOAT32, 2, (uint32_t[]){2, 5});
    tensor_t *bias = tensor_create(TENSOR_FLOAT32, 1, (uint32_t[]){2});
    linear_t *linear_weight = linear_create(weight, bias);

    for (int i = 0; i < input->num_elements; i++) {
        input->data[i].float32 = (float)i;
    }
    for (int i = 0; i < weight->num_elements; i++) {
        weight->data[i].float32 = (float)i;
    }
    for (int i = 0; i < bias->num_elements; i++) {
        bias->data[i].float32 = (float)i;
    }

    tensor_t *output = linear(input, linear_weight);
    tensor_print_shape(output);
    tensor_print_data(output);

    // tensor_free(weight);
    // tensor_free(bias);
    // linear_free(linear_weight, 0);
    linear_free(linear_weight, 1);
    
    tensor_free(input);
    tensor_free(output);

    printf(">> Done\r\n");
    return 0;
}