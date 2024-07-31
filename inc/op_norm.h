#ifndef _OP_NORM_H
#define _OP_NORM_H

#include "tensor.h"

typedef struct {
    tensor_t *mean;
    tensor_t *var;
    tensor_t *epsilon;
    tensor_t *gamma;
    tensor_t *beta;
} batch_norm_t;

batch_norm_t *batch_norm_create(tensor_t *mean, tensor_t *var, tensor_t *epsilon, tensor_t *gamma, tensor_t *beta);
void batch_free(batch_norm_t *batch_norm, uint8_t deep);

tensor_t *batch_norm_2d(tensor_t *input, batch_norm_t *batch_norm_weight);

#endif // _OP_NORM_H