/*
    Author: Kwangryeol Park
    Created: 2024.07.31
    
    Tensor의 Transpose 함수를 테스트하는 예제.
    2D tensor와 3D tensor에 대해 테스트한다.
    Tranpose는 Tensor 데이터 자체를 tranpose하는 것이 아닌, index를 접근하는데 있어서의 순서를 바꾸는 것이다.
*/
#include <stdio.h>
#include <stdint.h>
#include "tensor.h"
#include "op_linear.h"

int main() {
    printf(">> Demo: Create and free a tensor\r\n");

    printf(">> Transpose 2D tensor\r\n");
    tensor_t *tensor = tensor_create(TENSOR_INT32, 2, (uint32_t[]){3, 5});
    for (int i = 0; i < tensor->num_elements; i++) {
        tensor->data[i].int32 = i;
    }
    uint32_t row = 1;
    uint32_t col = 3;
    printf(">> Original tensor at %d, %d\r\n", row, col);
    tensor_print_shape(tensor);
    printf(">> %d\r\n", tensor->data[tensor_convert_nd_to_1d_index(tensor, (uint32_t[]){row, col})].int32);

    row = 3;
    col = 1;
    tensor_transpose(tensor, 1, 0); // Warning: it is the same
    printf(">> Transposed tensor at %d, %d\r\n", row, col);
    tensor_print_shape(tensor);
    printf(">> %d\r\n", tensor->data[tensor_convert_nd_to_1d_index(tensor, (uint32_t[]){row, col})].int32);

    tensor_free(tensor);






    printf(">> Transpose 3D tensor\r\n");
    tensor = tensor_create(TENSOR_INT32, 3, (uint32_t[]){3, 4, 5});
    for (int i = 0; i < tensor->num_elements; i++) {
        tensor->data[i].int32 = i;
    }
    row = 1;
    col = 3;
    uint32_t dpt = 2;
    printf(">> Original tensor at %d, %d, %d\r\n", row, col, dpt);
    tensor_print_shape(tensor);
    printf(">> %d\r\n", tensor->data[tensor_convert_nd_to_1d_index(tensor, (uint32_t[]){row, col, dpt})].int32);

    row = 3;
    col = 1;
    tensor_transpose(tensor, 1, 0); // Warning: it is the same
    printf(">> Original tensor at %d, %d, %d\r\n", row, col, dpt);
    tensor_print_shape(tensor);
    printf(">> %d\r\n", tensor->data[tensor_convert_nd_to_1d_index(tensor, (uint32_t[]){row, col, dpt})].int32);

    tensor_free(tensor);
    printf(">> Done\r\n");
    return 0;
}