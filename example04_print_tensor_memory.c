/*
    Author: Kwangryeol Park
    Created: 2024.07.31
    
    Tensor가 차지하는 RAM 메모리 크기를 계산하는 함수를 구현한다.
    이 메모리는 tensor_create할 때와 tensor_free할 때 계산된다.
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

    printf(">> Memory allocated by input tensor: %u bytes\r\n", tensor_get_data_memory(input));
    tensor_print_data_memory(input);
    printf(">> Memory allocated by weight tensor: %u bytes\r\n", tensor_get_data_memory(weight));
    tensor_print_data_memory(weight);
    printf(">> Memory allocated by bias tensor: %u bytes\r\n", tensor_get_data_memory(bias));
    tensor_print_data_memory(bias);

    for (int i = 0; i < input->num_elements; i++) {
        input->data[i].float32 = (float)i;
    }
    for (int i = 0; i < weight->num_elements; i++) {
        weight->data[i].float32 = (float)i;
    }
    for (int i = 0; i < bias->num_elements; i++) {
        bias->data[i].float32 = (float)i;
    }

    tensor_t *output = linear(input, weight, bias);
    printf(">> Memory allocated by output tensor: %u bytes\r\n", tensor_get_data_memory(output));
    tensor_print_data_memory(output);
    tensor_print_shape(output);
    tensor_print_data(output);

    printf(">> Memory allocated by tensor: %lu bytes\r\n", tensor_get_global_data_memory());
    printf(">> Free memory allocated by tensor\r\n");
    tensor_free(input);
    tensor_free(weight);
    tensor_free(bias);
    tensor_free(output);

    printf(">> Memory allocated by tensor: %lu bytes\r\n", tensor_get_global_data_memory());
    printf(">> Done\r\n");
    return 0;
}