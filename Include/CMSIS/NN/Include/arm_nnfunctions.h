/*
 * Copyright (C) 2021-2022 Chair of Electronic Design Automation, TUM
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#ifndef _ARM_NNFUNCTIONS_H
#define _ARM_NNFUNCTIONS_H

#include "arm_nn_math_types.h"
#include "arm_nn_types.h"
#include "muriscv_nn_functions.h"

#ifdef __cplusplus
extern "C" {
#endif

#define arm_relu6_s8 muriscv_nn_relu6_s8

#define arm_elementwise_add_s8 muriscv_nn_elementwise_add_s8
#define arm_elementwise_add_s16 muriscv_nn_elementwise_add_s16

#define arm_elementwise_mul_s8 muriscv_nn_elementwise_mul_s8
#define arm_elementwise_mul_s16 muriscv_nn_elementwise_mul_s16

#define arm_convolve_1x1_s8_fast muriscv_nn_convolve_1x1_s8_fast
#define arm_convolve_1x1_s8_fast_get_buffer_size muriscv_nn_convolve_1x1_s8_fast_get_buffer_size

#define arm_convolve_s8 muriscv_nn_convolve_s8
#define arm_convolve_s8_get_buffer_size muriscv_nn_convolve_s8_get_buffer_size
#define arm_convolve_s16 muriscv_nn_convolve_s16
#define arm_convolve_s16_get_buffer_size muriscv_nn_convolve_s16_get_buffer_size

#define arm_convolve_wrapper_s8 muriscv_nn_convolve_wrapper_s8
#define arm_convolve_wrapper_s8_get_buffer_size muriscv_nn_convolve_wrapper_s8_get_buffer_size
#define arm_convolve_wrapper_s16 muriscv_nn_convolve_wrapper_s16
#define arm_convolve_wrapper_s16_get_buffer_size muriscv_nn_convolve_wrapper_s16_get_buffer_size

#define arm_depthwise_conv_s8 muriscv_nn_depthwise_conv_s8
#define arm_depthwise_conv_s8_get_buffer_size muriscv_nn_depthwise_conv_s8_get_buffer_size
#define arm_depthwise_conv_s16 muriscv_nn_depthwise_conv_s16
#define arm_depthwise_conv_s16_get_buffer_size muriscv_nn_depthwise_conv_s16_get_buffer_size

#define arm_depthwise_conv_wrapper_s8 muriscv_nn_depthwise_conv_wrapper_s8
#define arm_depthwise_conv_wrapper_s8_get_buffer_size muriscv_nn_depthwise_conv_wrapper_s8_get_buffer_size
#define arm_depthwise_conv_wrapper_s16 muriscv_nn_depthwise_conv_wrapper_s16
#define arm_depthwise_conv_wrapper_s16_get_buffer_size muriscv_nn_depthwise_conv_wrapper_s16_get_buffer_size

#define arm_fully_connected_s8 muriscv_nn_fully_connected_s8
#define arm_fully_connected_s8_get_buffer_size muriscv_nn_fully_connected_s8_get_buffer_size
#define arm_fully_connected_s16 muriscv_nn_fully_connected_s16
#define arm_fully_connected_s16_get_buffer_size muriscv_nn_fully_connected_s16_get_buffer_size

#define arm_avgpool_s8 muriscv_nn_avgpool_s8
#define arm_avgpool_s8_get_buffer_size muriscv_nn_avgpool_s8_get_buffer_size
#define arm_avgpool_s16 muriscv_nn_avgpool_s16
#define arm_avgpool_s16_get_buffer_size muriscv_nn_avgpool_s16_get_buffer_size

#define arm_max_pool_s8 muriscv_nn_maxpool_s8
#define arm_max_pool_s16 muriscv_nn_maxpool_s16

#define arm_softmax_s8 muriscv_nn_softmax_s8
#define arm_softmax_s8_s16 muriscv_nn_softmax_s8_s16
#define arm_softmax_s16 muriscv_nn_softmax_s16

#define arm_svdf_s8 muriscv_nn_svdf_s8
#define arm_svdf_state_s16_s8 muriscv_nn_svdf_state_s16_s8

#ifdef __cplusplus
}
#endif

#endif /* _ARM_NNFUNCTIONS_H */
