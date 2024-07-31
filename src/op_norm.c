#include "op_norm.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "tensor.h"

#ifndef NULL
#define NULL 0
#endif

batch_norm_t *batch_norm_create(tensor_t *mean, tensor_t *var, tensor_t *epsilon, tensor_t *gamma, tensor_t *beta) {
    // mean: 1D tensor      (channels)
    // var: 1D tensor       (channels)
    // epsilon: 1D tensor   (channels)
    // gamma: 1D tensor     (channels)
    // beta: 1D tensor      (channels)
    // output: 4D tensor    (batch_size x channels x height x width)
    
    // Type check
    if (mean->type != var->type || mean->type != epsilon->type || mean->type != gamma->type || mean->type != beta->type) {
        printf("[%s][%s][%d] Error: mean, var, epsilon, gamma, and beta must have the same type (float32)\r\n", __FILE__, __func__, __LINE__);
        return NULL;
    }

    // Allocate batch_norm_t
    batch_norm_t *batch_norm_weight = (batch_norm_t *)malloc(sizeof(batch_norm_t));
    batch_norm_weight->mean = mean;
    batch_norm_weight->var = var;
    batch_norm_weight->epsilon = epsilon;
    batch_norm_weight->gamma = gamma;
    batch_norm_weight->beta = beta;
}

void batch_free(batch_norm_t *batch_norm, uint8_t deep) {
    // deep: 0 - free only batch_norm_t, 1 - free batch_norm_t and mean, var, epsilon, gamma, beta
    if (deep != 0) {
        if (batch_norm->mean != (tensor_t *) NULL) {
            tensor_free(batch_norm->mean);
        }
        if (batch_norm->var != (tensor_t *) NULL) {
            tensor_free(batch_norm->var);
        }
        if (batch_norm->epsilon != (tensor_t *) NULL) {
            tensor_free(batch_norm->epsilon);
        }
        if (batch_norm->gamma != (tensor_t *) NULL) {
            tensor_free(batch_norm->gamma);
        }
        if (batch_norm->beta != (tensor_t *) NULL) {
            tensor_free(batch_norm->beta);
        }
    } else {
        batch_norm->mean = (tensor_t *) NULL;
        batch_norm->var = (tensor_t *) NULL;
        batch_norm->epsilon = (tensor_t *) NULL;
        batch_norm->gamma = (tensor_t *) NULL;
        batch_norm->beta = (tensor_t *) NULL;
    }
    free(batch_norm);
}

tensor_t *batch_norm_2d(tensor_t *input, batch_norm_t *batch_norm_weight) {
    // output = (input - mean) / sqrt(var + epsilon) * gamma + beta
    // input: 4D tensor    (batch_size x channels x height x width)
    // mean: 1D tensor      (channels)
    // var: 1D tensor       (channels)
    // epsilon: 1D tensor   (channels)
    // gamma: 1D tensor     (channels)
    // beta: 1D tensor      (channels)
    // output: 4D tensor    (batch_size x channels x height x width)
    // All the types must be float32
    
    tensor_t *mean = batch_norm_weight->mean;
    tensor_t *var = batch_norm_weight->var;
    tensor_t *epsilon = batch_norm_weight->epsilon;
    tensor_t *gamma = batch_norm_weight->gamma;
    tensor_t *beta = batch_norm_weight->beta;
    tensor_t *output = tensor_create(input->type, input->ndim, input->shape, (void *)0);

    if (epsilon == (tensor_t *) NULL) { // epsilon is not given
        epsilon = tensor_create(input->type, 1, (uint32_t[]){mean->shape[0]}, (void *)0);
        for (int i = 0; i < epsilon->num_elements; i++) {
            epsilon->data[i].float32 = 1e-5f;   // default epsilon PyTorch
        }
    }
    
    // Check shape
    if (input->ndim != 4) {
        printf("[%s][%s][%d] Error: input tensor must be 4D tensor\r\n", __FILE__, __func__, __LINE__);
        return NULL;
    }
    if (mean->ndim != 1) {
        printf("[%s][%s][%d] Error: mean tensor must be 1D tensor\r\n", __FILE__, __func__, __LINE__);
        return NULL;
    }
    if (var->ndim != 1) {
        printf("[%s][%s][%d] Error: var tensor must be 1D tensor\r\n", __FILE__, __func__, __LINE__);
        return NULL;
    }
    if (epsilon->ndim != 1) {
        printf("[%s][%s][%d] Error: epsilon tensor must be 1D tensor\r\n", __FILE__, __func__, __LINE__);
        return NULL;
    }
    if (gamma->ndim != 1) {
        printf("[%s][%s][%d] Error: gamma tensor must be 1D tensor\r\n", __FILE__, __func__, __LINE__);
        return NULL;
    }
    if (beta->ndim != 1) {
        printf("[%s][%s][%d] Error: beta tensor must be 1D tensor\r\n", __FILE__, __func__, __LINE__);
        return NULL;
    }

    // Type check
    if (input->type != mean->type || input->type != var->type || input->type != epsilon->type || input->type != gamma->type || input->type != beta->type) {
        printf("[%s][%s][%d] Error: input, mean, var, epsilon, gamma, and beta must have the same type (float32)\r\n", __FILE__, __func__, __LINE__);
        return NULL;
    }

    tensor_t *input_coefficient = tensor_create(input->type, 1, (uint32_t[]){input->shape[1]}, (void *)0);
    tensor_t *bias = tensor_create(input->type, 1, (uint32_t[]){input->shape[1]}, (void *)0);
    for (int i = 0; i < input_coefficient->num_elements; i++) {
        input_coefficient->data[i].float32 = 1.0f / sqrt(var->data[i].float32 + epsilon->data[i].float32) * gamma->data[i].float32; // 1 / sqrt(var + epsilon) * gamma
        bias->data[i].float32 = -mean->data[i].float32 * input_coefficient->data[i].float32 + beta->data[i].float32;    // -mean / sqrt(var + epsilon) * gamma + beta
    }

    for (int c = 0; c < input->shape[1]; c++) {
        for (int i = 0; i < input->shape[0]; i++) {
            for (int j = 0; j < input->shape[2]; j++) {
                for (int k = 0; k < input->shape[3]; k++) {
                    output->data[tensor_convert_nd_to_1d_index(output, (uint32_t[]){i, c, j, k})].float32 = input->data[tensor_convert_nd_to_1d_index(input, (uint32_t[]){i, c, j, k})].float32 * input_coefficient->data[c].float32 + bias->data[c].float32;
                }
            }
        }
    }

    tensor_free(input_coefficient);
    tensor_free(bias);

    return output;
}