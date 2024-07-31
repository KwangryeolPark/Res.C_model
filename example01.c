/*
    Author: Kwangryeol Park
    Created: 2024.07.31
    
    가장 기본적인 예제.
    Tensor를 만들고, 데이터를 채우고, 출력하는 예제.
    Tensor는 1D이며, tensor_convert_nd_index_to_1d_index 함수를 이용하여 N-Dimensional index를 1D index로 변환한다. 
*/
#include <stdio.h>
#include <stdint.h>
#include "tensor.h"

int main() {
    printf(">> Demo: Create and free a tensor\r\n");
    uint32_t shape[] = {2, 3, 4};
    tensor_t *tensor = tensor_create(TENSOR_INT16, 3, shape);
    tensor_fill_with(tensor, (tensor_data_t){.int16 = 0});


    tensor_print_data(tensor);
    tensor_print_shape(tensor);
    tensor_print_dim(tensor);

    for (int i = 0; i < tensor->shape[0]; i++) {
        for (int j = 0; j < tensor->shape[1]; j++) {
            for (int k = 0; k < tensor->shape[2]; k++) {
                tensor->data[tensor_convert_nd_index_to_1d_index(tensor, (uint32_t[]){i, j, k})] = (tensor_data_t) (i * 100 + j * 10 + k);
            }
        }
    }

    printf(">> element at (1, 2, 3) = %d\r\n", tensor->data[tensor_convert_nd_index_to_1d_index(tensor, (uint32_t[]){1, 2, 3})]);

    tensor_free(tensor);
    printf(">> Done\r\n");
    return 0;
}