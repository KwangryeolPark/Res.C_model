#ifndef _OP_LINEAR_H
#define _OP_LINEAR_H

#include "tensor.h"

tensor_t *linear(tensor_t *input, tensor_t *weight, tensor_t *bias);

#endif // _OP_LINEAR_H