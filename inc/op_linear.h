#ifndef _OP_LINEAR_H
#define _OP_LINEAR_H

#include "tensor.h"

typedef struct {
    tensor_t *weight;
    tensor_t *bias;
} linear_t;

linear_t *linear_create(tensor_t *weight, tensor_t *bias);
void linear_free(linear_t *linear, uint8_t deep);
tensor_t *linear(tensor_t *input, linear_t *linear_weight);

#endif // _OP_LINEAR_H