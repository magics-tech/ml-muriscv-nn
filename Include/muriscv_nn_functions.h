/*
 * Copyright (C) 2010-2022 Arm Limited or its affiliates.
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
 * Modifications copyright (C) 2021-2022 Chair of Electronic Design Automation, TUM
 */

/**
 * @defgroup groupNN Neural Network Functions
 * A collection of functions to perform basic operations for neural network layers. Functions with a _s8 suffix support
 * TensorFlow Lite framework.
 */

#ifndef _MURISCV_NN_FUNCTIONS_H
#define _MURISCV_NN_FUNCTIONS_H

#include "muriscv_nn_math_types.h"
#include "muriscv_nn_types.h"

// Include the Vicuna C runtime when running on Vicuna
#ifdef SIM_VICUNA
#include "crt/vicuna_crt.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup NNConv Convolution Functions
 *
 * Collection of convolution, depthwise convolution functions and their variants.
 *
 * The convolution is implemented in 2 steps: im2col and GEMM
 *
 * im2col is a process of converting each patch of image data into
 * a column. After im2col, the convolution is computed as matrix-matrix
 * multiplication.
 *
 * To reduce the memory footprint, the im2col is performed partially.
 * Each iteration, only a few column (i.e., patches) are generated and
 * computed with GEMM kernels similar to DSP muriscv_nn_mat_mult functions.
 *
 */

/**
 * @brief s8 convolution layer wrapper function with the main purpose to call the optimal kernel available in
 muriscv-nn
 *        to perform the convolution.
 *
 * @param[in, out] ctx            Function context that contains the additional buffer if required by the function.
                                  muriscv_nn_convolve_wrapper_s8_get_buffer_size will return the buffer_size if required
 * @param[in]      conv_params    Convolution parameters (e.g. strides, dilations, pads,...).
 *                                Range of conv_params->input_offset  : [-127, 128]
 *                                Range of conv_params->output_offset : [-128, 127]
 * @param[in]      quant_params   Per-channel quantization info.
 *                                It contains the multiplier and shift values to be applied to each output channel
 * @param[in]      input_dims     Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]      input_data     Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims    Filter tensor dimensions. Format: [C_OUT, HK, WK, C_IN] where HK and WK are the
 *                                spatial filter dimensions
 * @param[in]      filter_data    Filter data pointer. Data type: int8
 * @param[in]      bias_dims      Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data      Bias data pointer. Data type: int32
 * @param[in]      output_dims    Output tensor dimensions. Format: [N, H, W, C_OUT]
 * @param[out]     output_data    Output data pointer. Data type: int8
 *
 * @return     The function returns either
 *                  <code>MURISCV_NN_ARG_ERROR</code> if argument constraints fail. or,
 *                  <code>MURISCV_NN_SUCCESS</code> on successful completion.
 *
 */
muriscv_nn_status muriscv_nn_convolve_wrapper_s8(const muriscv_nn_context *ctx,
                                                 const muriscv_nn_conv_params *conv_params,
                                                 const muriscv_nn_per_channel_quant_params *quant_params,
                                                 const muriscv_nn_dims *input_dims,
                                                 const q7_t *input_data,
                                                 const muriscv_nn_dims *filter_dims,
                                                 const q7_t *filter_data,
                                                 const muriscv_nn_dims *bias_dims,
                                                 const int32_t *bias_data,
                                                 const muriscv_nn_dims *output_dims,
                                                 q7_t *output_data);

/**
 * @brief Get the required buffer size for muriscv_nn_convolve_wrapper_s8
 *
 * @param[in]      conv_params    Convolution parameters (e.g. strides, dilations, pads,...).
 *                                Range of conv_params->input_offset  : [-127, 128]
 *                                Range of conv_params->output_offset : [-128, 127]
 * @param[in]      input_dims     Input (activation) dimensions. Format: [N, H, W, C_IN]
 * @param[in]      filter_dims    Filter dimensions. Format: [C_OUT, HK, WK, C_IN] where HK and WK are the spatial
 *                                filter dimensions
 * @param[in]      output_dims    Output tensor dimensions. Format: [N, H, W, C_OUT]
 *
 * @return         The function returns  required buffer size(bytes)
 *
 */
int32_t muriscv_nn_convolve_wrapper_s8_get_buffer_size(const muriscv_nn_conv_params *conv_params,
                                                       const muriscv_nn_dims *input_dims,
                                                       const muriscv_nn_dims *filter_dims,
                                                       const muriscv_nn_dims *output_dims);

/**
 * @brief s16 convolution layer wrapper function with the main purpose to call the optimal kernel available in
 muriscv-nn
 *        to perform the convolution.
 *
 * @param[in, out] ctx            Function context that contains the additional buffer if required by the function.
                                  muriscv_nn_convolve_wrapper_s8_get_buffer_size will return the buffer_size if required
 * @param[in]      conv_params    Convolution parameters (e.g. strides, dilations, pads,...).
 *                                conv_params->input_offset  : Not used
 *                                conv_params->output_offset : Not used
 * @param[in]      quant_params   Per-channel quantization info.
 *                                It contains the multiplier and shift values to be applied to each output channel
 * @param[in]      input_dims     Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]      input_data     Input (activation) data pointer. Data type: int16
 * @param[in]      filter_dims    Filter tensor dimensions. Format: [C_OUT, HK, WK, C_IN] where HK and WK are the
 *                                spatial filter dimensions
 * @param[in]      filter_data    Filter data pointer. Data type: int8
 * @param[in]      bias_dims      Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data      Bias data pointer. Data type: int64
 * @param[in]      output_dims    Output tensor dimensions. Format: [N, H, W, C_OUT]
 * @param[out]     output_data    Output data pointer. Data type: int16
 *
 * @return     The function returns either
 *                  <code>MURISCV_NN_ARG_ERROR</code> if argument constraints fail. or,
 *                  <code>MURISCV_NN_SUCCESS</code> on successful completion.
 *
 */
muriscv_nn_status muriscv_nn_convolve_wrapper_s16(const muriscv_nn_context *ctx,
                                                  const muriscv_nn_conv_params *conv_params,
                                                  const muriscv_nn_per_channel_quant_params *quant_params,
                                                  const muriscv_nn_dims *input_dims,
                                                  const q15_t *input_data,
                                                  const muriscv_nn_dims *filter_dims,
                                                  const q7_t *filter_data,
                                                  const muriscv_nn_dims *bias_dims,
                                                  const int64_t *bias_data,
                                                  const muriscv_nn_dims *output_dims,
                                                  q15_t *output_data);

/**
 * @brief Get the required buffer size for muriscv_nn_convolve_wrapper_s16
 *
 * @param[in]      conv_params    Convolution parameters (e.g. strides, dilations, pads,...).
 *                                conv_params->input_offset  : Not used
 *                                conv_params->output_offset : Not used
 * @param[in]      input_dims     Input (activation) dimensions. Format: [N, H, W, C_IN]
 * @param[in]      filter_dims    Filter dimensions. Format: [C_OUT, HK, WK, C_IN] where HK and WK are the spatial
 *                                filter dimensions
 * @param[in]      output_dims    Output tensor dimensions. Format: [N, H, W, C_OUT]
 *
 * @return         The function returns  required buffer size(bytes)
 *
 */
int32_t muriscv_nn_convolve_wrapper_s16_get_buffer_size(const muriscv_nn_conv_params *conv_params,
                                                        const muriscv_nn_dims *input_dims,
                                                        const muriscv_nn_dims *filter_dims,
                                                        const muriscv_nn_dims *output_dims);

/**
 * @brief Basic s8 convolution function
 * @param[in, out] ctx            Function context that contains the additional buffer if required by the function.
                                  muriscv_nn_convolve_s8_get_buffer_size will return the buffer_size if required
 * @param[in]      conv_params    Convolution parameters (e.g. strides, dilations, pads,...).
 *                                Range of conv_params->input_offset  : [-127, 128]
 *                                Range of conv_params->output_offset : [-128, 127]
 * @param[in]      quant_params   Per-channel quantization info.
 *                                It contains the multiplier and shift values to be applied to each output channel
 * @param[in]      input_dims     Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]      input_data     Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims    Filter tensor dimensions. Format: [C_OUT, HK, WK, C_IN] where HK and WK are the
 *                                spatial filter dimensions
 * @param[in]      filter_data    Filter data pointer. Data type: int8
 * @param[in]      bias_dims      Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data      Optional bias data pointer. Data type: int32
 * @param[in]      output_dims    Output tensor dimensions. Format: [N, H, W, C_OUT]
 * @param[out]     output_data    Output data pointer. Data type: int8

 * @return     The function returns <code>MURISCV_NN_SUCCESS</code>
 *
 * @details
 *    1. Supported framework: TensorFlow Lite micro
 *    2. q7 is used as data type eventhough it is s8 data. It is done so to be consistent with existing APIs.
 *    3. Additional memory is required for optimization. Refer to argument 'ctx' for details.
 *
 */
muriscv_nn_status muriscv_nn_convolve_s8(const muriscv_nn_context *ctx,
                                         const muriscv_nn_conv_params *conv_params,
                                         const muriscv_nn_per_channel_quant_params *quant_params,
                                         const muriscv_nn_dims *input_dims,
                                         const q7_t *input_data,
                                         const muriscv_nn_dims *filter_dims,
                                         const q7_t *filter_data,
                                         const muriscv_nn_dims *bias_dims,
                                         const int32_t *bias_data,
                                         const muriscv_nn_dims *output_dims,
                                         q7_t *output_data);

/**
 * @brief Get the required buffer size for s8 convolution function
 *
 * @param[in]       input_dims            Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]       filter_dims           Filter tensor dimensions. Format: [C_OUT, HK, WK, C_IN] where HK and WK
 * are the spatial filter dimensions
 * @return          The function returns  required buffer size(bytes)
 *
 */
int32_t muriscv_nn_convolve_s8_get_buffer_size(const muriscv_nn_dims *input_dims, const muriscv_nn_dims *filter_dims);

/**
 * @brief Basic s16 convolution function
 * @param[in, out] ctx            Function context that contains the additional buffer if required by the function.
                                  muriscv_nn_convolve_s16_get_buffer_size will return the buffer_size if required
 * @param[in]      conv_params    Convolution parameters (e.g. strides, dilations, pads,...).
 *                                conv_params->input_offset  : Not used
 *                                conv_params->output_offset : Not used
 * @param[in]      quant_params   Per-channel quantization info.
 *                                It contains the multiplier and shift values to be applied to each output channel
 * @param[in]      input_dims     Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]      input_data     Input (activation) data pointer. Data type: int16
 * @param[in]      filter_dims    Filter tensor dimensions. Format: [C_OUT, HK, WK, C_IN] where HK and WK are the
 *                                spatial filter dimensions
 * @param[in]      filter_data    Filter data pointer. Data type: int8
 * @param[in]      bias_dims      Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data      Optional bias data pointer. Data type: int64
 * @param[in]      output_dims    Output tensor dimensions. Format: [N, H, W, C_OUT]
 * @param[out]     output_data    Output data pointer. Data type: int16

 * @return     The function returns <code>MURISCV_NN_SUCCESS</code>
 *
 * @details
 *    1. Supported framework: TensorFlow Lite micro
 *    2. q7/q15 is used as data type eventhough it is s8/s16 data. It is done so to be consistent with existing APIs.
 *    3. Additional memory is required for optimization. Refer to argument 'ctx' for details.
 *
 */
muriscv_nn_status muriscv_nn_convolve_s16(const muriscv_nn_context *ctx,
                                          const muriscv_nn_conv_params *conv_params,
                                          const muriscv_nn_per_channel_quant_params *quant_params,
                                          const muriscv_nn_dims *input_dims,
                                          const q15_t *input_data,
                                          const muriscv_nn_dims *filter_dims,
                                          const q7_t *filter_data,
                                          const muriscv_nn_dims *bias_dims,
                                          const int64_t *bias_data,
                                          const muriscv_nn_dims *output_dims,
                                          q15_t *output_data);

/**
 * @brief Get the required buffer size for s16 convolution function
 *
 * @param[in]       input_dims            Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]       filter_dims           Filter tensor dimensions. Format: [C_OUT, HK, WK, C_IN] where HK and WK
 * are the spatial filter dimensions
 * @return          The function returns  required buffer size(bytes)
 *
 */
int32_t muriscv_nn_convolve_s16_get_buffer_size(const muriscv_nn_dims *input_dims, const muriscv_nn_dims *filter_dims);

/**
 * @brief Optimized s16 convolution function
 * @param[in, out] ctx            Function context that contains the additional buffer if required by the function.
                                  muriscv_nn_convolve_fast_s16_get_buffer_size will return the buffer_size if required
 * @param[in]      conv_params    Convolution parameters (e.g. strides, dilations, pads,...).
 *                                conv_params->input_offset  : Not used
 *                                conv_params->output_offset : Not used
 * @param[in]      quant_params   Per-channel quantization info.
 *                                It contains the multiplier and shift values to be applied to each output channel
 * @param[in]      input_dims     Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]      input_data     Input (activation) data pointer. Data type: int16
 * @param[in]      filter_dims    Filter tensor dimensions. Format: [C_OUT, HK, WK, C_IN] where HK and WK are the
 *                                spatial filter dimensions. (filter_dims->w * filter_dims->h * input_dims->c) must not
 exceed 512
 * @param[in]      filter_data    Filter data pointer. Data type: int8
 * @param[in]      bias_dims      Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data      Optional bias data pointer. Data type: int64
 * @param[in]      output_dims    Output tensor dimensions. Format: [N, H, W, C_OUT]
 * @param[out]     output_data    Output data pointer. Data type: int16

 * @return     The function returns <code>MURISCV_NN_SUCCESS</code>
 *
 * @details
 *    1. Supported framework: TensorFlow Lite micro
 *    2. q7/q15 is used as data type eventhough it is s8/s16 data. It is done so to be consistent with existing APIs.
 *    3. Additional memory is required for optimization. Refer to argument 'ctx' for details.
 *    4. Implementation supports kernel volumes (filter width * filter height * input channels) < 512.
 *
 */
muriscv_nn_status muriscv_nn_convolve_fast_s16(const muriscv_nn_context *ctx,
                                               const muriscv_nn_conv_params *conv_params,
                                               const muriscv_nn_per_channel_quant_params *quant_params,
                                               const muriscv_nn_dims *input_dims,
                                               const q15_t *input_data,
                                               const muriscv_nn_dims *filter_dims,
                                               const q7_t *filter_data,
                                               const muriscv_nn_dims *bias_dims,
                                               const int64_t *bias_data,
                                               const muriscv_nn_dims *output_dims,
                                               q15_t *output_data);

/**
 * @brief Get the required buffer size for fast s16 convolution function
 *
 * @param[in]       input_dims            Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]       filter_dims           Filter tensor dimensions. Format: [C_OUT, HK, WK, C_IN] where HK and WK
 * are the spatial filter dimensions
 * @return          The function returns required buffer size(bytes)
 *
 */
int32_t muriscv_nn_convolve_fast_s16_get_buffer_size(const muriscv_nn_dims *input_dims,
                                                     const muriscv_nn_dims *filter_dims);

/**
 * @brief Fast s8 version for 1x1 convolution (non-square shape)
 *
 * @param[in, out] ctx            Function context that contains the additional buffer if required by the function.
                                  muriscv_nn_convolve_1x1_s8_fast_get_buffer_size will return the buffer_size if
 required
 * @param[in]      conv_params    Convolution parameters (e.g. strides, dilations, pads,...).
 *                                Range of conv_params->input_offset  : [-127, 128]
 *                                Range of conv_params->output_offset : [-128, 127]
 * @param[in]      quant_params   Per-channel quantization info.
 *                                It contains the multiplier and shift values to be applied to each output channel
 * @param[in]      input_dims     Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]      input_data     Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims    Filter tensor dimensions. Format: [C_OUT, 1, 1, C_IN]
 * @param[in]      filter_data    Filter data pointer. Data type: int8
 * @param[in]      bias_dims      Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data      Optional bias data pointer. Data type: int32
 * @param[in]      output_dims    Output tensor dimensions. Format: [N, H, W, C_OUT]
 * @param[out]     output_data    Output data pointer. Data type: int8
 *
 * @return     The function returns either
 *                  <code>MURISCV_NN_ARG_ERROR</code> if argument constraints fail. or,
 *                  <code>MURISCV_NN_SUCCESS</code> on successful completion.
 *
 * @details
 *   - Supported framework : TensorFlow Lite Micro
 *   - The following constrains on the arguments apply
 *      -# input_dims->c is a multiple of 4
 *      -# conv_params->padding.w = conv_params->padding.h = 0
 *      -# conv_params->stride.w = conv_params->stride.h = 1
 *
 */
muriscv_nn_status muriscv_nn_convolve_1x1_s8_fast(const muriscv_nn_context *ctx,
                                                  const muriscv_nn_conv_params *conv_params,
                                                  const muriscv_nn_per_channel_quant_params *quant_params,
                                                  const muriscv_nn_dims *input_dims,
                                                  const q7_t *input_data,
                                                  const muriscv_nn_dims *filter_dims,
                                                  const q7_t *filter_data,
                                                  const muriscv_nn_dims *bias_dims,
                                                  const int32_t *bias_data,
                                                  const muriscv_nn_dims *output_dims,
                                                  q7_t *output_data);

/**
 * @brief Get the required buffer size for muriscv_nn_convolve_1x1_s8_fast
 *
 * @param[in]       input_dims            Input (activation) dimensions
 * @return          The function returns the required buffer size in bytes
 *
 */
int32_t muriscv_nn_convolve_1x1_s8_fast_get_buffer_size(const muriscv_nn_dims *input_dims);

/**
 * @brief 1xn convolution
 *
 * @param[in, out] ctx            Function context that contains the additional buffer if required by the function.
                                  muriscv_nn_convolve_1_x_n_s8_get_buffer_size will return the buffer_size if required
 * @param[in]      conv_params    Convolution parameters (e.g. strides, dilations, pads,...).
 *                                Range of conv_params->input_offset  : [-127, 128]
 *                                Range of conv_params->output_offset : [-128, 127]
 * @param[in]      quant_params   Per-channel quantization info.
 *                                It contains the multiplier and shift values to be applied to each output channel
 * @param[in]      input_dims     Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]      input_data     Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims    Filter tensor dimensions. Format: [C_OUT, 1, WK, C_IN] where WK is the horizontal
 *                                spatial filter dimension
 * @param[in]      filter_data    Filter data pointer. Data type: int8
 * @param[in]      bias_dims      Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data      Optional bias data pointer. Data type: int32
 * @param[in]      output_dims    Output tensor dimensions. Format: [N, H, W, C_OUT]
 * @param[out]     output_data    Output data pointer. Data type: int8
 *
 * @return     The function returns either
 *                  <code>MURISCV_NN_ARG_ERROR</code> if argument constraints fail. or,
 *                  <code>MURISCV_NN_SUCCESS</code> on successful completion.
 *
 * @details
 *   - Supported framework : TensorFlow Lite Micro
 *   - The following constrains on the arguments apply
 *      -# input_dims->n equals 1
 *      -# ouput_dims->w is a multiple of 4
 *      -# Explicit constraints(since it is for 1xN convolution)
 *      -## input_dims->h equals 1
 *      -## output_dims->h equals 1
 *      -## filter_dims->h equals 1
 *@todo  Remove constraint on output_dims->w to make the function generic.
 *
 */
muriscv_nn_status muriscv_nn_convolve_1_x_n_s8(const muriscv_nn_context *ctx,
                                               const muriscv_nn_conv_params *conv_params,
                                               const muriscv_nn_per_channel_quant_params *quant_params,
                                               const muriscv_nn_dims *input_dims,
                                               const q7_t *input_data,
                                               const muriscv_nn_dims *filter_dims,
                                               const q7_t *filter_data,
                                               const muriscv_nn_dims *bias_dims,
                                               const int32_t *bias_data,
                                               const muriscv_nn_dims *output_dims,
                                               q7_t *output_data);

/**
 * @brief Get the required additional buffer size for 1xn convolution
 *
 * @param[in]       input_dims            Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]       filter_dims           Filter tensor dimensions. Format: [C_OUT, 1, WK, C_IN] where WK is the
 *                                        horizontal spatial filter dimension
 * @return          The function returns  required buffer size(bytes)
 *
 */
int32_t muriscv_nn_convolve_1_x_n_s8_get_buffer_size(const muriscv_nn_dims *input_dims,
                                                     const muriscv_nn_dims *filter_dims);

/**
 * @brief Wrapper function to pick the right optimized s8 depthwise convolution function
 *
 * @param[in, out] ctx            Function context (e.g. temporary buffer). Check the function
 *                                definition file to see if an additional buffer is required.
 *                                Optional function {API}_get_buffer_size() provides the buffer
 *                                size if required.
 * @param[in]      dw_conv_params Depthwise convolution parameters (e.g. strides, dilations, pads,...)
 *                                dw_conv_params->dilation is not used.
 *                                Range of dw_conv_params->input_offset : [-127, 128]
 *                                Range of dw_conv_params->output_offset : [-128, 127]
 * @param[in]      quant_params   Per-channel quantization info.
 *                               It contains the multiplier and shift values to be applied to each
 *                               output channel
 * @param[in]      input_dims     Input (activation) tensor dimensions. Format: [H, W, C_IN]
 *                                Batch argument N is not used and assumed to be 1.
 * @param[in]      input_data     Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims    Filter tensor dimensions. Format: [1, H, W, C_OUT]
 * @param[in]      filter_data    Filter data pointer. Data type: int8
 * @param[in]      bias_dims      Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data      Bias data pointer. Data type: int32
 * @param[in]      output_dims    Output tensor dimensions. Format: [1, H, W, C_OUT]
 * @param[in, out] output_data    Output data pointer. Data type: int8
 * @return     The function returns
 *                <code>MURISCV_NN_SUCCESS</code>   -  Successful completion.
 *
 * @details
 *    - Supported framework: TensorFlow Lite
 *    - Picks one of the the following functions
 *        -# muriscv_nn_depthwise_conv_s8()
 *        -# muriscv_nn_depthwise_conv_3x3_s8() - Cortex-M CPUs with DSP extension only
 *        -# muriscv_nn_depthwise_conv_s8_opt()
 *    - q7 is used as data type eventhough it is s8 data. It is done so to be consistent with existing APIs.
 *    - Check details of muriscv_nn_depthwise_conv_s8_opt() for potential data that can be accessed outside of the
 * boundary.
 */
muriscv_nn_status muriscv_nn_depthwise_conv_wrapper_s8(const muriscv_nn_context *ctx,
                                                       const muriscv_nn_dw_conv_params *dw_conv_params,
                                                       const muriscv_nn_per_channel_quant_params *quant_params,
                                                       const muriscv_nn_dims *input_dims,
                                                       const q7_t *input_data,
                                                       const muriscv_nn_dims *filter_dims,
                                                       const q7_t *filter_data,
                                                       const muriscv_nn_dims *bias_dims,
                                                       const int32_t *bias_data,
                                                       const muriscv_nn_dims *output_dims,
                                                       q7_t *output_data);

/**
 * @brief Get size of additional buffer required by muriscv_nn_depthwise_conv_wrapper_s8()
 *
 * @param[in]      dw_conv_params Depthwise convolution parameters (e.g. strides, dilations, pads,...)
 *                                Range of dw_conv_params->input_offset : [-127, 128]
 *                                Range of dw_conv_params->input_offset : [-128, 127]
 * @param[in]      input_dims     Input (activation) tensor dimensions. Format: [H, W, C_IN]
 *                                Batch argument N is not used and assumed to be 1.
 * @param[in]      filter_dims    Filter tensor dimensions. Format: [1, H, W, C_OUT]
 * @param[in]      output_dims    Output tensor dimensions. Format: [1, H, W, C_OUT]
 * @return                        Size of additional memory required for optimizations in bytes.
 *
 */
int32_t muriscv_nn_depthwise_conv_wrapper_s8_get_buffer_size(const muriscv_nn_dw_conv_params *dw_conv_params,
                                                             const muriscv_nn_dims *input_dims,
                                                             const muriscv_nn_dims *filter_dims,
                                                             const muriscv_nn_dims *output_dims);

/**
 * @brief Basic s8 depthwise convolution function that doesn't have any constraints on the input dimensions.
 *
 * @param[in, out] ctx            Function context (e.g. temporary buffer). Check the function
 *                                definition file to see if an additional buffer is required.
 *                                Optional function {API}_get_buffer_size() provides the buffer
 *                                size if an additional buffer is required.
 *                                exists if additional memory is.
 * @param[in]      dw_conv_params Depthwise convolution parameters (e.g. strides, dilations, pads,...)
 *                                dw_conv_params->dilation is not used.
 *                                Range of dw_conv_params->input_offset : [-127, 128]
 *                                Range of dw_conv_params->input_offset : [-128, 127]
 * @param[in]      quant_params   Per-channel quantization info.
 *                               It contains the multiplier and shift values to be applied to each
 *                               output channel
 * @param[in]      input_dims     Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 *                                Batch argument N is not used.
 * @param[in]      input_data     Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims    Filter tensor dimensions. Format: [1, H, W, C_OUT]
 * @param[in]      filter_data    Filter data pointer. Data type: int8
 * @param[in]      bias_dims      Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data      Bias data pointer. Data type: int32
 * @param[in]      output_dims    Output tensor dimensions. Format: [N, H, W, C_OUT]
 * @param[in, out] output_data    Output data pointer. Data type: int8
 * @return     The function returns <code>MURISCV_NN_SUCCESS</code>
 *
 * @details
 *    - Supported framework: TensorFlow Lite
 *    - q7 is used as data type eventhough it is s8 data. It is done so to be consistent with existing APIs.
 */
muriscv_nn_status muriscv_nn_depthwise_conv_s8(const muriscv_nn_context *ctx,
                                               const muriscv_nn_dw_conv_params *dw_conv_params,
                                               const muriscv_nn_per_channel_quant_params *quant_params,
                                               const muriscv_nn_dims *input_dims,
                                               const q7_t *input_data,
                                               const muriscv_nn_dims *filter_dims,
                                               const q7_t *filter_data,
                                               const muriscv_nn_dims *bias_dims,
                                               const int32_t *bias_data,
                                               const muriscv_nn_dims *output_dims,
                                               q7_t *output_data);

/**
 * @brief Basic s16 depthwise convolution function that doesn't have any constraints on the input dimensions.
 *
 * @param[in, out] ctx            Function context (e.g. temporary buffer). Check the function
 *                                definition file to see if an additional buffer is required.
 *                                Optional function {API}_get_buffer_size() provides the buffer
 *                                size if an additional buffer is required.
 *                                exists if additional memory is.
 * @param[in]      dw_conv_params Depthwise convolution parameters (e.g. strides, dilations, pads,...)
 *                                conv_params->input_offset  : Not used
 *                                conv_params->output_offset : Not used
 * @param[in]      quant_params   Per-channel quantization info.
 *                               It contains the multiplier and shift values to be applied to each
 *                               output channel
 * @param[in]      input_dims     Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 *                                Batch argument N is not used.
 * @param[in]      input_data     Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims    Filter tensor dimensions. Format: [1, H, W, C_OUT]
 * @param[in]      filter_data    Filter data pointer. Data type: int8
 * @param[in]      bias_dims      Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data      Bias data pointer. Data type: int64
 * @param[in]      output_dims    Output tensor dimensions. Format: [N, H, W, C_OUT]
 * @param[in, out] output_data    Output data pointer. Data type: int16
 * @return     The function returns <code>MURISCV_NN_SUCCESS</code>
 *
 * @details
 *    - Supported framework: TensorFlow Lite
 *    - q15 is used as data type eventhough it is s16 data. It is done so to be consistent with existing APIs.
 */
muriscv_nn_status muriscv_nn_depthwise_conv_s16(const muriscv_nn_context *ctx,
                                                const muriscv_nn_dw_conv_params *dw_conv_params,
                                                const muriscv_nn_per_channel_quant_params *quant_params,
                                                const muriscv_nn_dims *input_dims,
                                                const q15_t *input_data,
                                                const muriscv_nn_dims *filter_dims,
                                                const q7_t *filter_data,
                                                const muriscv_nn_dims *bias_dims,
                                                const int64_t *bias_data,
                                                const muriscv_nn_dims *output_dims,
                                                q15_t *output_data);

/**
 * @brief Wrapper function to pick the right optimized s16 depthwise convolution function
 *
 * @param[in, out] ctx            Function context (e.g. temporary buffer). Check the function
 *                                definition file to see if an additional buffer is required.
 *                                Optional function {API}_get_buffer_size() provides the buffer
 *                                size if required.
 * @param[in]      dw_conv_params Depthwise convolution parameters (e.g. strides, dilations, pads,...)
 *                                dw_conv_params->dilation is not used.
 *                                Range of dw_conv_params->input_offset : Not used
 *                                Range of dw_conv_params->output_offset : Not used
 * @param[in]      quant_params   Per-channel quantization info.
 *                                It contains the multiplier and shift values to be applied to each
 *                                output channel
 * @param[in]      input_dims     Input (activation) tensor dimensions. Format: [H, W, C_IN]
 *                                Batch argument N is not used and assumed to be 1.
 * @param[in]      input_data     Input (activation) data pointer. Data type: int16
 * @param[in]      filter_dims    Filter tensor dimensions. Format: [1, H, W, C_OUT]
 * @param[in]      filter_data    Filter data pointer. Data type: int8
 * @param[in]      bias_dims      Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data      Bias data pointer. Data type: int64
 * @param[in]      output_dims    Output tensor dimensions. Format: [1, H, W, C_OUT]
 * @param[in, out] output_data    Output data pointer. Data type: int16
 * @return     The function returns
 *                <code>MURISCV_NN_SUCCESS</code>   -  Successful completion.
 *
 * @details
 *    - Supported framework: TensorFlow Lite
 *    - Picks one of the the following functions
 *        -# muriscv_nn_depthwise_conv_s16()
 *        -# muriscv_nn_depthwise_conv_fast_s16()  - Cortex-M CPUs with DSP extension only
 *    - q7 is used as data type eventhough it is s8 data. It is done so to be consistent with existing APIs.
 */
muriscv_nn_status muriscv_nn_depthwise_conv_wrapper_s16(const muriscv_nn_context *ctx,
                                                        const muriscv_nn_dw_conv_params *dw_conv_params,
                                                        const muriscv_nn_per_channel_quant_params *quant_params,
                                                        const muriscv_nn_dims *input_dims,
                                                        const q15_t *input_data,
                                                        const muriscv_nn_dims *filter_dims,
                                                        const q7_t *filter_data,
                                                        const muriscv_nn_dims *bias_dims,
                                                        const int64_t *bias_data,
                                                        const muriscv_nn_dims *output_dims,
                                                        q15_t *output_data);

/**
 * @brief Get size of additional buffer required by muriscv_nn_depthwise_conv_wrapper_s16()
 *
 * @param[in]      dw_conv_params Depthwise convolution parameters (e.g. strides, dilations, pads,...)
 *                                Range of dw_conv_params->input_offset : Not used
 *                                Range of dw_conv_params->input_offset : Not used
 * @param[in]      input_dims     Input (activation) tensor dimensions. Format: [H, W, C_IN]
 *                                Batch argument N is not used and assumed to be 1.
 * @param[in]      filter_dims    Filter tensor dimensions. Format: [1, H, W, C_OUT]
 * @param[in]      output_dims    Output tensor dimensions. Format: [1, H, W, C_OUT]
 * @return                        Size of additional memory required for optimizations in bytes.
 *
 */
int32_t muriscv_nn_depthwise_conv_wrapper_s16_get_buffer_size(const muriscv_nn_dw_conv_params *dw_conv_params,
                                                              const muriscv_nn_dims *input_dims,
                                                              const muriscv_nn_dims *filter_dims,
                                                              const muriscv_nn_dims *output_dims);

/**
 * @brief Optimized s16 depthwise convolution function with constraint that in_channel equals out_channel.
 *        Refer muriscv_nn_depthwise_conv_s16() for function argument details.
 *
 * @return     The function returns one of the following
 *                <code>MURISCV_NN_ARG_ERROR</code> - ctx-buff == NULL and
 *                                                      muriscv_nn_depthwise_conv_fast_s16_get_buffer_size() > 0 or
 *                                                      input channel != output channel or
 *                                                      ch_mult != 1
 *
 *                <code>MURISCV_NN_SUCCESS</code> - Successful operation
 *
 * @details
 *    - Supported framework: TensorFlow Lite
 *    - The following constrains on the arguments apply
 *        -# Number of input channel equals number of output channels or ch_mult equals 1
 *    - q7 is used as data type eventhough it is s8 data. It is done so to be consistent with existing APIs.
 *    - Reccomended when number of channels is 4 or greater.
 *
 */
muriscv_nn_status muriscv_nn_depthwise_conv_fast_s16(const muriscv_nn_context *ctx,
                                                     const muriscv_nn_dw_conv_params *dw_conv_params,
                                                     const muriscv_nn_per_channel_quant_params *quant_params,
                                                     const muriscv_nn_dims *input_dims,
                                                     const q15_t *input_data,
                                                     const muriscv_nn_dims *filter_dims,
                                                     const q7_t *filter_data,
                                                     const muriscv_nn_dims *bias_dims,
                                                     const int64_t *bias_data,
                                                     const muriscv_nn_dims *output_dims,
                                                     q15_t *output_data);

/**
 * @brief Get the required buffer size for optimized s16 depthwise convolution
 * function with constraint that in_channel equals out_channel.
 * @param[in]       input_dims     Input (activation) tensor dimensions. Format: [1, H, W, C_IN]
 *                                 Batch argument N is not used.
 * @param[in]       filter_dims    Filter tensor dimensions. Format: [1, H, W, C_OUT]
 * @return          The function returns  required buffer size in bytes
 *
 */
int32_t muriscv_nn_depthwise_conv_fast_s16_get_buffer_size(const muriscv_nn_dims *input_dims,
                                                           const muriscv_nn_dims *filter_dims);

/**
 * @brief Optimized s8 depthwise convolution function for 3x3 kernel size with some constraints on
 *        the input arguments(documented below). Refer muriscv_nn_depthwise_conv_s8() for function
 *        argument details.
 *
 * @return     The function returns one of the following
 *                <code>MURISCV_NN_ARG_ERROR</code> - Unsupported dimension of tensors
 *                                                  - Unsupported pad size along the x axis
 *                <code>MURISCV_NN_SUCCESS</code> - Successful operation
 *
 * @details
 *   - Supported framework : TensorFlow Lite Micro
 *   - The following constrains on the arguments apply
 *      -# Number of input channel equals number of output channels
 *      -# Filter height and width equals 3
 *      -# Padding along x is either 0 or 1.
 *
 */
muriscv_nn_status muriscv_nn_depthwise_conv_3x3_s8(const muriscv_nn_context *ctx,
                                                   const muriscv_nn_dw_conv_params *dw_conv_params,
                                                   const muriscv_nn_per_channel_quant_params *quant_params,
                                                   const muriscv_nn_dims *input_dims,
                                                   const q7_t *input_data,
                                                   const muriscv_nn_dims *filter_dims,
                                                   const q7_t *filter_data,
                                                   const muriscv_nn_dims *bias_dims,
                                                   const int32_t *bias_data,
                                                   const muriscv_nn_dims *output_dims,
                                                   q7_t *output_data);

/**
 * @brief Optimized s8 depthwise convolution function with constraint that in_channel equals out_channel.
 *        Refer muriscv_nn_depthwise_conv_s8() for function argument details.
 *
 * @return     The function returns one of the following
 *                <code>MURISCV_NN_ARG_ERROR</code> - input channel != output channel or
 *                                                      ch_mult != 1
 *                <code>MURISCV_NN_SUCCESS</code> - Successful operation
 *
 * @note       If number of channels is not a multiple of 4, upto 3 elements outside the boundary will be read out
 *             for the following if the RISC-V vector extension is used.
 *               - Output shift
 *               - Output multiplier
 *               - Output bias
 *               - kernel
 * @details
 *    - Supported framework: TensorFlow Lite
 *    - The following constrains on the arguments apply
 *        -# Number of input channel equals number of output channels or ch_mult equals 1
 *    - q7 is used as data type eventhough it is s8 data. It is done so to be consistent with existing APIs.
 *    - Reccomended when number of channels is 4 or greater.
 *
 */
muriscv_nn_status muriscv_nn_depthwise_conv_s8_opt(const muriscv_nn_context *ctx,
                                                   const muriscv_nn_dw_conv_params *dw_conv_params,
                                                   const muriscv_nn_per_channel_quant_params *quant_params,
                                                   const muriscv_nn_dims *input_dims,
                                                   const q7_t *input_data,
                                                   const muriscv_nn_dims *filter_dims,
                                                   const q7_t *filter_data,
                                                   const muriscv_nn_dims *bias_dims,
                                                   const int32_t *bias_data,
                                                   const muriscv_nn_dims *output_dims,
                                                   q7_t *output_data);

/**
 * @brief Get the required buffer size for optimized s8 depthwise convolution
 * function with constraint that in_channel equals out_channel.
 * @param[in]       input_dims     Input (activation) tensor dimensions. Format: [1, H, W, C_IN]
 *                                 Batch argument N is not used.
 * @param[in]       filter_dims    Filter tensor dimensions. Format: [1, H, W, C_OUT]
 * @return          The function returns  required buffer size in bytes
 *
 */
int32_t muriscv_nn_depthwise_conv_s8_opt_get_buffer_size(const muriscv_nn_dims *input_dims,
                                                         const muriscv_nn_dims *filter_dims);

/**
 * @defgroup FC Fully-connected Layer Functions
 *
 * Collection of fully-connected and matrix multiplication functions.
 *
 * Fully-connected layer is basically a matrix-vector multiplication
 * with bias. The matrix is the weights and the input/output vectors
 * are the activation values. Supported {weight, activation} precisions
 * include {8-bit, 8-bit}, {16-bit, 16-bit}, and {8-bit, 16-bit}.
 *
 * Here we have two types of kernel functions. The basic function
 * implements the function using regular GEMV approach. The opt functions
 * operates with weights in interleaved formats.
 *
 */

/**
 * @brief Basic s8 Fully Connected function.
 *
 * @param[in, out] ctx            Function context (e.g. temporary buffer). Check the function
 *                                definition file to see if an additional buffer is required.
 *                                Optional function {API}_get_buffer_size() provides the buffer
 *                                size if an additional buffer is required.
 * @param[in]      fc_params      Fully Connected layer parameters.
 *                                Range of fc_params->input_offset  : [-127, 128]
 *                                fc_params->filter_offset : 0
 *                                Range of fc_params->output_offset : [-128, 127]
 * @param[in]      quant_params   Per-tensor quantization info.
 *                                It contains the multiplier and shift values to be applied to the output tensor.
 * @param[in]      input_dims     Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 *                                Input dimension is taken as Nx(H * W * C_IN)
 * @param[in]      input_data     Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims    Two dimensional filter dimensions. Format: [N, C]
 *                                N : accumulation depth and equals (H * W * C_IN) from input_dims
 *                                C : output depth and equals C_OUT in output_dims
 *                                H & W : Not used
 * @param[in]      filter_data    Filter data pointer. Data type: int8
 * @param[in]      bias_dims      Bias tensor dimensions. Format: [C_OUT]
 *                                N, H, W : Not used
 * @param[in]      bias_data      Bias data pointer. Data type: int32
 * @param[in]      output_dims    Output tensor dimensions. Format: [N, C_OUT]
 *                                N : Batches
 *                                C_OUT : Output depth
 *                                H & W : Not used.
 * @param[in, out] output_data    Output data pointer. Data type: int8
 * @return     The function returns <code>MURISCV_NN_SUCCESS</code>
 *
 * @details
 *    - Supported framework: TensorFlow Lite
 *    - q7 is used as data type eventhough it is s8 data. It is done so to be consistent with existing APIs.
 */
muriscv_nn_status muriscv_nn_fully_connected_s8(const muriscv_nn_context *ctx,
                                                const muriscv_nn_fc_params *fc_params,
                                                const muriscv_nn_per_tensor_quant_params *quant_params,
                                                const muriscv_nn_dims *input_dims,
                                                const q7_t *input_data,
                                                const muriscv_nn_dims *filter_dims,
                                                const q7_t *filter_data,
                                                const muriscv_nn_dims *bias_dims,
                                                const int32_t *bias_data,
                                                const muriscv_nn_dims *output_dims,
                                                q7_t *output_data);

/**
 * @brief Get the required buffer size for S8 basic fully-connected and
 * matrix multiplication layer function for TF Lite
 * @param[in]      filter_dims             dimension of filter
 * @return         The function returns    required buffer size in bytes
 *
 */
int32_t muriscv_nn_fully_connected_s8_get_buffer_size(const muriscv_nn_dims *filter_dims);

/**
 * @brief Basic s16 Fully Connected function.
 *
 * @param[in, out] ctx            Function context (e.g. temporary buffer). Check the function
 *                                definition file to see if an additional buffer is required.
 *                                Optional function {API}_get_buffer_size() provides the buffer
 *                                size if an additional buffer is required.
 * @param[in]      fc_params      Fully Connected layer parameters.
 *                                fc_params->input_offset  : 0
 *                                fc_params->filter_offset : 0
 *                                fc_params->output_offset : 0
 * @param[in]      quant_params   Per-tensor quantization info.
 *                                It contains the multiplier and shift values to be applied to the output tensor.
 * @param[in]      input_dims     Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 *                                Input dimension is taken as Nx(H * W * C_IN)
 * @param[in]      input_data     Input (activation) data pointer. Data type: int16
 * @param[in]      filter_dims    Two dimensional filter dimensions. Format: [N, C]
 *                                N : accumulation depth and equals (H * W * C_IN) from input_dims
 *                                C : output depth and equals C_OUT in output_dims
 *                                H & W : Not used
 * @param[in]      filter_data    Filter data pointer. Data type: int8
 * @param[in]      bias_dims      Bias tensor dimensions. Format: [C_OUT]
 *                                N, H, W : Not used
 * @param[in]      bias_data      Bias data pointer. Data type: int64
 * @param[in]      output_dims    Output tensor dimensions. Format: [N, C_OUT]
 *                                N : Batches
 *                                C_OUT : Output depth
 *                                H & W : Not used.
 * @param[in, out] output_data    Output data pointer. Data type: int16
 * @return     The function returns <code>MURISCV_NN_SUCCESS</code>
 *
 * @details
 *    - Supported framework: TensorFlow Lite
 *    - q15 is used as data type eventhough it is s16 data. It is done so to be consistent with existing APIs.
 */
muriscv_nn_status muriscv_nn_fully_connected_s16(const muriscv_nn_context *ctx,
                                                 const muriscv_nn_fc_params *fc_params,
                                                 const muriscv_nn_per_tensor_quant_params *quant_params,
                                                 const muriscv_nn_dims *input_dims,
                                                 const q15_t *input_data,
                                                 const muriscv_nn_dims *filter_dims,
                                                 const q7_t *filter_data,
                                                 const muriscv_nn_dims *bias_dims,
                                                 const int64_t *bias_data,
                                                 const muriscv_nn_dims *output_dims,
                                                 q15_t *output_data);

/**
 * @brief Get the required buffer size for S16 basic fully-connected and
 * matrix multiplication layer function for TF Lite
 * @param[in]      filter_dims             dimension of filter
 * @return         The function returns    required buffer size in bytes
 *
 */
int32_t muriscv_nn_fully_connected_s16_get_buffer_size(const muriscv_nn_dims *filter_dims);

/*
 *  Other functions
 *  These layers are typically not timing critical
 *  Basic implementation is supported here
 */

/**
 * @defgroup BasicMath Basic math functions
 *
 * Elementwise add and multiplication functions.
 *
 */

/**
 * @brief s8 elementwise add of two vectors
 * @param[in]       input_1_vect            pointer to input vector 1
 * @param[in]       input_2_vect            pointer to input vector 2
 * @param[in]       input_1_offset          offset for input 1. Range: -127 to 128
 * @param[in]       input_1_mult            multiplier for input 1
 * @param[in]       input_1_shift           shift for input 1
 * @param[in]       input_2_offset          offset for input 2. Range: -127 to 128
 * @param[in]       input_2_mult            multiplier for input 2
 * @param[in]       input_2_shift           shift for input 2
 * @param[in]       left_shift              input left shift
 * @param[in,out]   output                  pointer to output vector
 * @param[in]       out_offset              output offset.  Range: -128 to 127
 * @param[in]       out_mult                output multiplier
 * @param[in]       out_shift               output shift
 * @param[in]       out_activation_min      minimum value to clamp output to. Min: -128
 * @param[in]       out_activation_max      maximum value to clamp output to. Max: 127
 * @param[in]       block_size              number of samples
 * @return          The function returns    MURISCV_NN_SUCCESS
 */
muriscv_nn_status muriscv_nn_elementwise_add_s8(const int8_t *input_1_vect,
                                                const int8_t *input_2_vect,
                                                const int32_t input_1_offset,
                                                const int32_t input_1_mult,
                                                const int32_t input_1_shift,
                                                const int32_t input_2_offset,
                                                const int32_t input_2_mult,
                                                const int32_t input_2_shift,
                                                const int32_t left_shift,
                                                int8_t *output,
                                                const int32_t out_offset,
                                                const int32_t out_mult,
                                                const int32_t out_shift,
                                                const int32_t out_activation_min,
                                                const int32_t out_activation_max,
                                                const int32_t block_size);

/**
 * @brief s16 elementwise add of two vectors
 * @param[in]       input_1_vect            pointer to input vector 1
 * @param[in]       input_2_vect            pointer to input vector 2
 * @param[in]       input_1_offset          offset for input 1. Not used.
 * @param[in]       input_1_mult            multiplier for input 1
 * @param[in]       input_1_shift           shift for input 1
 * @param[in]       input_2_offset          offset for input 2. Not used.
 * @param[in]       input_2_mult            multiplier for input 2
 * @param[in]       input_2_shift           shift for input 2
 * @param[in]       left_shift              input left shift
 * @param[in,out]   output                  pointer to output vector
 * @param[in]       out_offset              output offset. Not used.
 * @param[in]       out_mult                output multiplier
 * @param[in]       out_shift               output shift
 * @param[in]       out_activation_min      minimum value to clamp output to. Min: -32768
 * @param[in]       out_activation_max      maximum value to clamp output to. Max: 32767
 * @param[in]       block_size              number of samples
 * @return          The function returns    MURISCV_NN_SUCCESS
 */
muriscv_nn_status muriscv_nn_elementwise_add_s16(const int16_t *input_1_vect,
                                                 const int16_t *input_2_vect,
                                                 const int32_t input_1_offset,
                                                 const int32_t input_1_mult,
                                                 const int32_t input_1_shift,
                                                 const int32_t input_2_offset,
                                                 const int32_t input_2_mult,
                                                 const int32_t input_2_shift,
                                                 const int32_t left_shift,
                                                 int16_t *output,
                                                 const int32_t out_offset,
                                                 const int32_t out_mult,
                                                 const int32_t out_shift,
                                                 const int32_t out_activation_min,
                                                 const int32_t out_activation_max,
                                                 const int32_t block_size);

/**
 * @brief s8 elementwise multiplication
 * @param[in]       input_1_vect            pointer to input vector 1
 * @param[in]       input_2_vect            pointer to input vector 2
 * @param[in]       input_1_offset          offset for input 1. Range: -127 to 128
 * @param[in]       input_2_offset          offset for input 2. Range: -127 to 128
 * @param[in,out]   output                  pointer to output vector
 * @param[in]       out_offset              output offset. Range: -128 to 127
 * @param[in]       out_mult                output multiplier
 * @param[in]       out_shift               output shift
 * @param[in]       out_activation_min      minimum value to clamp output to. Min: -128
 * @param[in]       out_activation_max      maximum value to clamp output to. Max: 127
 * @param[in]       block_size              number of samples
 * @return          The function returns    MURISCV_NN_SUCCESS
 *
 * @details   Supported framework: TensorFlow Lite micro
 */
muriscv_nn_status muriscv_nn_elementwise_mul_s8(const int8_t *input_1_vect,
                                                const int8_t *input_2_vect,
                                                const int32_t input_1_offset,
                                                const int32_t input_2_offset,
                                                int8_t *output,
                                                const int32_t out_offset,
                                                const int32_t out_mult,
                                                const int32_t out_shift,
                                                const int32_t out_activation_min,
                                                const int32_t out_activation_max,
                                                const int32_t block_size);

/**
 * @brief s16 elementwise multiplication
 * @param[in]       input_1_vect            pointer to input vector 1
 * @param[in]       input_2_vect            pointer to input vector 2
 * @param[in]       input_1_offset          offset for input 1. Not used.
 * @param[in]       input_2_offset          offset for input 2. Not used.
 * @param[in,out]   output                  pointer to output vector
 * @param[in]       out_offset              output offset. Not used.
 * @param[in]       out_mult                output multiplier
 * @param[in]       out_shift               output shift
 * @param[in]       out_activation_min      minimum value to clamp output to. Min: -32768
 * @param[in]       out_activation_max      maximum value to clamp output to. Max: 32767
 * @param[in]       block_size              number of samples
 * @return          The function returns    MURISCV_NN_SUCCESS
 *
 * @details   Supported framework: TensorFlow Lite micro
 */
muriscv_nn_status muriscv_nn_elementwise_mul_s16(const int16_t *input_1_vect,
                                                 const int16_t *input_2_vect,
                                                 const int32_t input_1_offset,
                                                 const int32_t input_2_offset,
                                                 int16_t *output,
                                                 const int32_t out_offset,
                                                 const int32_t out_mult,
                                                 const int32_t out_shift,
                                                 const int32_t out_activation_min,
                                                 const int32_t out_activation_max,
                                                 const int32_t block_size);

/**
 * @defgroup Acti Activation Functions
 *
 * Perform activation layers, including ReLU (Rectified Linear Unit),
 * sigmoid and tanh
 *
 */

/**
 * @brief Q7 RELU function
 * @param[in,out]   data        pointer to input
 * @param[in]       size        number of elements
 * @return none.
 */
void muriscv_nn_relu_q7(q7_t *data, uint16_t size);

/**
 * @brief s8 ReLU6 function
 * @param[in,out]   data        pointer to input
 * @param[in]       size        number of elements
 */
void muriscv_nn_relu6_s8(q7_t *data, uint16_t size);

/**
 * @defgroup Pooling Pooling Functions
 *
 * Perform pooling functions, including max pooling and average pooling
 *
 */

/**
 * @brief s8 average pooling function.
 *
 * @param[in, out] ctx            Function context (e.g. temporary buffer). Check the function
 *                                definition file to see if an additional buffer is required.
 *                                Optional function {API}_get_buffer_size() provides the buffer
 *                                size if an additional buffer is required.
 * @param[in]      pool_params    Pooling parameters
 * @param[in]      input_dims     Input (activation) tensor dimensions. Format: [H, W, C_IN]
 *                                Argument 'N' is not used.
 * @param[in]      input_data     Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims    Filter tensor dimensions. Format: [H, W]
 *                                Argument N and C are not used.
 * @param[in]      output_dims    Output tensor dimensions. Format: [H, W, C_OUT]
 *                                Argument N is not used.
 *                                C_OUT equals C_IN.
 * @param[in, out] output_data    Output data pointer. Data type: int8
 * @return                        The function returns
 *                                    <code>MURISCV_NN_SUCCESS</code> - Successful operation
 *
 * @details
 *    - Supported Framework: TensorFlow Lite
 *
 */
muriscv_nn_status muriscv_nn_avgpool_s8(const muriscv_nn_context *ctx,
                                        const muriscv_nn_pool_params *pool_params,
                                        const muriscv_nn_dims *input_dims,
                                        const q7_t *input_data,
                                        const muriscv_nn_dims *filter_dims,
                                        const muriscv_nn_dims *output_dims,
                                        q7_t *output_data);

/**
 * @brief Get the required buffer size for S8 average pooling function
 * @param[in]       dim_dst_width         output tensor dimension
 * @param[in]       ch_src                number of input tensor channels
 * @return          The function returns  required buffer size in bytes
 *
 */
int32_t muriscv_nn_avgpool_s8_get_buffer_size(const int dim_dst_width, const int ch_src);

/**
 * @brief s16 average pooling function.
 *
 * @param[in, out] ctx            Function context (e.g. temporary buffer). Check the function
 *                                definition file to see if an additional buffer is required.
 *                                Optional function {API}_get_buffer_size() provides the buffer
 *                                size if an additional buffer is required.
 * @param[in]      pool_params    Pooling parameters
 * @param[in]      input_dims     Input (activation) tensor dimensions. Format: [H, W, C_IN]
 *                                Argument 'N' is not used.
 * @param[in]      input_data     Input (activation) data pointer. Data type: int16
 * @param[in]      filter_dims    Filter tensor dimensions. Format: [H, W]
 *                                Argument N and C are not used.
 * @param[in]      output_dims    Output tensor dimensions. Format: [H, W, C_OUT]
 *                                Argument N is not used.
 *                                C_OUT equals C_IN.
 * @param[in, out] output_data    Output data pointer. Data type: int16
 * @return                        The function returns
 *                                    <code>MURISCV_NN_SUCCESS</code> - Successful operation
 *                                    <code>MURISCV_NN_ARG_ERROR</code> - In case of invalid arguments
 *
 * @details
 *    - Supported Framework: TensorFlow Lite
 *
 */
muriscv_nn_status muriscv_nn_avgpool_s16(const muriscv_nn_context *ctx,
                                         const muriscv_nn_pool_params *pool_params,
                                         const muriscv_nn_dims *input_dims,
                                         const int16_t *input_data,
                                         const muriscv_nn_dims *filter_dims,
                                         const muriscv_nn_dims *output_dims,
                                         int16_t *output_data);

/**
 * @brief Get the required buffer size for S16 average pooling function
 * @param[in]       dim_dst_width         output tensor dimension
 * @param[in]       ch_src                number of input tensor channels
 * @return          The function returns  required buffer size in bytes
 *
 */
int32_t muriscv_nn_avgpool_s16_get_buffer_size(const int dim_dst_width, const int ch_src);

/**
 * @brief s8 max pooling function.
 *
 * @param[in, out] ctx            Function context (e.g. temporary buffer). Check the function
 *                                definition file to see if an additional buffer is required.
 *                                Optional function {API}_get_buffer_size() provides the buffer
 *                                size if an additional buffer is required.
 * @param[in]      pool_params    Pooling parameters
 * @param[in]      input_dims     Input (activation) tensor dimensions. Format: [H, W, C_IN]
 *                                Argument 'N' is not used.
 * @param[in]      input_data     Input (activation) data pointer. The input tensor must not
 *                                overlap with the output tensor. Data type: int8
 * @param[in]      filter_dims    Filter tensor dimensions. Format: [H, W]
 *                                Argument N and C are not used.
 * @param[in]      output_dims    Output tensor dimensions. Format: [H, W, C_OUT]
 *                                Argument N is not used.
 *                                C_OUT equals C_IN.
 * @param[in, out] output_data    Output data pointer. Data type: int8
 * @return                        The function returns
 *                                    <code>MURISCV_NN_SUCCESS</code> - Successful operation
 *
 * @details
 *    - Supported Framework: TensorFlow Lite
 *    - (fabianpedd): Name changed to 'maxpool' instead of 'max_pool' in order to be more consistent (see 'avgpool')
 *
 */
muriscv_nn_status muriscv_nn_maxpool_s8(const muriscv_nn_context *ctx,
                                        const muriscv_nn_pool_params *pool_params,
                                        const muriscv_nn_dims *input_dims,
                                        const q7_t *input_data,
                                        const muriscv_nn_dims *filter_dims,
                                        const muriscv_nn_dims *output_dims,
                                        q7_t *output_data);

/**
 * @brief s16 max pooling function.
 *
 * @param[in, out] ctx            Function context (e.g. temporary buffer). Check the function
 *                                definition file to see if an additional buffer is required.
 *                                Optional function {API}_get_buffer_size() provides the buffer
 *                                size if an additional buffer is required.
 * @param[in]      pool_params    Pooling parameters
 * @param[in]      input_dims     Input (activation) tensor dimensions. Format: [H, W, C_IN]
 *                                Argument 'N' is not used.
 * @param[in]      src            Input (activation) data pointer. The input tensor must not
 *                                overlap with the output tensor. Data type: int16
 * @param[in]      filter_dims    Filter tensor dimensions. Format: [H, W]
 *                                Argument N and C are not used.
 * @param[in]      output_dims    Output tensor dimensions. Format: [H, W, C_OUT]
 *                                Argument N is not used.
 *                                C_OUT equals C_IN.
 * @param[in, out] dst            Output data pointer. Data type: int16
 * @return                        The function returns
 *                                    <code>MURISCV_NN_SUCCESS</code> - Successful operation
 *
 * @details
 *    - Supported Framework: TensorFlow Lite
 *    - (fabianpedd): Name changed to 'maxpool' instead of 'max_pool' in order to be more consistent (see 'avgpool')
 *
 */
muriscv_nn_status muriscv_nn_maxpool_s16(const muriscv_nn_context *ctx,
                                         const muriscv_nn_pool_params *pool_params,
                                         const muriscv_nn_dims *input_dims,
                                         const int16_t *src,
                                         const muriscv_nn_dims *filter_dims,
                                         const muriscv_nn_dims *output_dims,
                                         int16_t *dst);

/**
 * @defgroup Softmax Softmax Functions
 *
 * EXP(2) based softmax functions.
 *
 */

/**
 * @brief Q7 softmax function
 * @param[in]       vec_in      pointer to input vector
 * @param[in]       dim_vec     input vector dimension
 * @param[out]      p_out       pointer to output vector
 *
 * @note This function is an optimized version which is not bit-accurate with
 *       TensorFlow Lite's kernel
 *
 */
void muriscv_nn_softmax_q7(const q7_t *vec_in, const uint16_t dim_vec, q7_t *p_out);

/**
 * @brief S8 softmax function
 * @param[in]  input     Pointer to the input tensor
 * @param[in]  num_rows  Number of rows in the input tensor
 * @param[in]  row_size  Number of elements in each input row
 * @param[in]  mult      Input quantization multiplier
 * @param[in]  shift     Input quantization shift within the range [0, 31]
 * @param[in]  diff_min  Minimum difference with max in row. Used to check if
 *                       the quantized exponential operation can be performed
 * @param[out] output    Pointer to the output tensor
 *
 * @note Supported framework: TensorFlow Lite micro (bit-accurate)
 *
 */
void muriscv_nn_softmax_s8(const int8_t *input,
                           const int32_t num_rows,
                           const int32_t row_size,
                           const int32_t mult,
                           const int32_t shift,
                           const int32_t diff_min,
                           int8_t *output);

/**
 * @brief S8 to s16 softmax function
 * @param[in]  input     Pointer to the input tensor
 * @param[in]  num_rows  Number of rows in the input tensor
 * @param[in]  row_size  Number of elements in each input row
 * @param[in]  mult      Input quantization multiplier
 * @param[in]  shift     Input quantization shift within the range [0, 31]
 * @param[in]  diff_min  Minimum difference with max in row. Used to check if
 *                       the quantized exponential operation can be performed
 * @param[out] output    Pointer to the output tensor
 *
 * @note Supported framework: TensorFlow Lite micro (bit-accurate)
 *
 */
void muriscv_nn_softmax_s8_s16(const int8_t *input,
                               const int32_t num_rows,
                               const int32_t row_size,
                               const int32_t mult,
                               const int32_t shift,
                               const int32_t diff_min,
                               int16_t *output);

/**
 * @brief S16 softmax function
 * @param[in]  input           Pointer to the input tensor
 * @param[in]  num_rows        Number of rows in the input tensor
 * @param[in]  row_size        Number of elements in each input row
 * @param[in]  mult            Input quantization multiplier
 * @param[in]  shift           Input quantization shift within the range [0, 31]
 * @param[in]  softmax_params  Softmax s16 layer parameters with two pointers to LUTs speficied below.
 *                             For indexing the high 9 bits are used and 7 remaining for interpolation.
 *                             That means 512 entries for the 9-bit indexing and 1 extra for interpolation, i.e. 513
 *                             values for each LUT.
 *                             - Lookup table for exp(x), where x uniform distributed between [-10.0 , 0.0]
 *                             - Lookup table for 1 / (1 + x), where x uniform distributed between [0.0 , 1.0]
 * @param[out] output          Pointer to the output tensor
 * @return                        The function returns
 *                                    <code>MURISCV_NN_ARG_ERROR</code> Argument error check failed
 *                                    <code>MURISCV_NN_SUCCESS</code> - Successful operation
 *
 * @note Supported framework: TensorFlow Lite micro (bit-accurate)
 *
 */
muriscv_nn_status muriscv_nn_softmax_s16(const int16_t *input,
                                         const int32_t num_rows,
                                         const int32_t row_size,
                                         const int32_t mult,
                                         const int32_t shift,
                                         const muriscv_nn_softmax_lut_s16 *softmax_params,
                                         int16_t *output);

/**
 * @brief U8 softmax function
 * @param[in]  input     Pointer to the input tensor
 * @param[in]  num_rows  Number of rows in the input tensor
 * @param[in]  row_size  Number of elements in each input row
 * @param[in]  mult      Input quantization multiplier
 * @param[in]  shift     Input quantization shift within the range [0, 31]
 * @param[in]  diff_min  Minimum difference with max in row. Used to check if
 *                       the quantized exponential operation can be performed
 * @param[out] output    Pointer to the output tensor
 *
 * @note Supported framework: TensorFlow Lite micro (bit-accurate)
 *
 */
void muriscv_nn_softmax_u8(const uint8_t *input,
                           const int32_t num_rows,
                           const int32_t row_size,
                           const int32_t mult,
                           const int32_t shift,
                           const int32_t diff_min,
                           uint8_t *output);

/**
 * @defgroup Reshape Reshape Functions
 *
 */

/**
 * @brief Reshape a s8 vector into another with different shape
 * @param[in]  input      points to the s8 input vector
 * @param[out] output     points to the s8 output vector
 * @param[in]  total_size total size of the input and output vectors in bytes
 *
 * @note The output is expected to be in a memory area that does not overlap with the input's
 *
 */
void muriscv_nn_reshape_s8(const int8_t *input, int8_t *output, const uint32_t total_size);

/**
 * @defgroup Concatenation Concatenation Functions
 *
 */

/**
 * @brief int8/uint8 concatenation function to be used for concatenating N-tensors along the X axis
 *        This function should be called for each input tensor to concatenate. The argument offset_x
 *        will be used to store the input tensor in the correct position in the output tensor
 *
 *        i.e.    offset_x = 0
 *                for(i = 0 i < num_input_tensors; ++i)
 *                {
 *                    muriscv_nn_concatenation_s8_x(&input[i], ..., &output, ..., ..., offset_x)
 *                    offset_x += input_x[i]
 *                }
 *
 *        This function assumes that the output tensor has:
 *        -# The same height of the input tensor
 *        -# The same number of channels of the input tensor
 *        -# The same batch size of the input tensor
 *
 *        Unless specified otherwise, arguments are mandatory.
 *
 * @note This function, data layout independent, can be used to concatenate either int8 or uint8 tensors because it
 *      does not involve any arithmetic operation
 *
 * @param[in]  input    Pointer to input tensor. Input tensor must not overlap with the output tensor.
 * @param[in]  input_x  Width of input tensor
 * @param[in]  input_y  Height of input tensor
 * @param[in]  input_z  Channels in input tensor
 * @param[in]  input_w  Batch size in input tensor
 * @param[out] output   Pointer to output tensor. Expected to be at least
 *                          (input_x * input_y * input_z * input_w) + offset_x
 *                      bytes.
 * @param[in]  output_x Width of output tensor
 * @param[in]  offset_x The offset (in number of elements) on the X axis to start concatenating the input tensor
 *                      It is user responsibility to provide the correct value
 *
 * <b> Input constraints</b>
 * offset_x is less than output_x
 *
 */
void muriscv_nn_concatenation_s8_x(const int8_t *input,
                                   const uint16_t input_x,
                                   const uint16_t input_y,
                                   const uint16_t input_z,
                                   const uint16_t input_w,
                                   int8_t *output,
                                   const uint16_t output_x,
                                   const uint32_t offset_x);

/**
 * @brief int8/uint8 concatenation function to be used for concatenating N-tensors along the Y axis
 *        This function should be called for each input tensor to concatenate. The argument offset_y
 *        will be used to store the input tensor in the correct position in the output tensor
 *
 *        i.e.    offset_y = 0
 *                for(i = 0 i < num_input_tensors; ++i)
 *                {
 *                    muriscv_nn_concatenation_s8_y(&input[i], ..., &output, ..., ..., offset_y)
 *                    offset_y += input_y[i]
 *                }
 *
 *        This function assumes that the output tensor has:
 *        -# The same width of the input tensor
 *        -# The same number of channels of the input tensor
 *        -# The same batch size of the input tensor
 *
 *        Unless specified otherwise, arguments are mandatory.
 *
 * @note This function, data layout independent, can be used to concatenate either int8 or uint8 tensors because it
 *       does not involve any arithmetic operation
 *
 * @param[in]  input    Pointer to input tensor. Input tensor must not overlap with the output tensor.
 * @param[in]  input_x  Width of input tensor
 * @param[in]  input_y  Height of input tensor
 * @param[in]  input_z  Channels in input tensor
 * @param[in]  input_w  Batch size in input tensor
 * @param[out] output   Pointer to output tensor. Expected to be at least
 *                          (input_z * input_w * input_x * input_y) + offset_y
 *                      bytes.
 * @param[in]  output_y Height of output tensor
 * @param[in]  offset_y The offset on the Y axis to start concatenating the input tensor
 *                      It is user responsibility to provide the correct value
 *
 * <b> Input constraints</b>
 * offset_y is less than output_y
 *
 */
void muriscv_nn_concatenation_s8_y(const int8_t *input,
                                   const uint16_t input_x,
                                   const uint16_t input_y,
                                   const uint16_t input_z,
                                   const uint16_t input_w,
                                   int8_t *output,
                                   const uint16_t output_y,
                                   const uint32_t offset_y);

/**
 * @brief int8/uint8 concatenation function to be used for concatenating N-tensors along the Z axis
 *        This function should be called for each input tensor to concatenate. The argument offset_z
 *        will be used to store the input tensor in the correct position in the output tensor
 *
 *        i.e.    offset_z = 0
 *                for(i = 0 i < num_input_tensors; ++i)
 *                {
 *                    muriscv_nn_concatenation_s8_z(&input[i], ..., &output, ..., ..., offset_z)
 *                    offset_z += input_z[i]
 *                }
 *
 *        This function assumes that the output tensor has:
 *        -# The same width of the input tensor
 *        -# The same height of the input tensor
 *        -# The same batch size of the input tensor
 *
 *        Unless specified otherwise, arguments are mandatory.
 *
 * @note This function, data layout independent, can be used to concatenate either int8 or uint8 tensors because it
 *       does not involve any arithmetic operation
 *
 * @param[in]  input    Pointer to input tensor. Input tensor must not overlap with output tensor.
 * @param[in]  input_x  Width of input tensor
 * @param[in]  input_y  Height of input tensor
 * @param[in]  input_z  Channels in input tensor
 * @param[in]  input_w  Batch size in input tensor
 * @param[out] output   Pointer to output tensor. Expected to be at least
 *                          (input_x * input_y * input_z * input_w) + offset_z
 *                      bytes.
 * @param[in]  output_z Channels in output tensor
 * @param[in]  offset_z The offset on the Z axis to start concatenating the input tensor
 *                      It is user responsibility to provide the correct value
 *
 * <b> Input constraints</b>
 * offset_z is less than output_z
 *
 */
void muriscv_nn_concatenation_s8_z(const int8_t *input,
                                   const uint16_t input_x,
                                   const uint16_t input_y,
                                   const uint16_t input_z,
                                   const uint16_t input_w,
                                   int8_t *output,
                                   const uint16_t output_z,
                                   const uint32_t offset_z);

/**
 * @brief int8/uint8 concatenation function to be used for concatenating N-tensors along the W axis (Batch size)
 *        This function should be called for each input tensor to concatenate. The argument offset_w
 *        will be used to store the input tensor in the correct position in the output tensor
 *
 *        i.e.    offset_w = 0
 *                for(i = 0 i < num_input_tensors; ++i)
 *                {
 *                    muriscv_nn_concatenation_s8_w(&input[i], ..., &output, ..., ..., offset_w)
 *                    offset_w += input_w[i]
 *                }
 *
 *        This function assumes that the output tensor has:
 *        -# The same width of the input tensor
 *        -# The same height of the input tensor
 *        -# The same number o channels of the input tensor
 *
 *        Unless specified otherwise, arguments are mandatory.
 *
 * @note This function, data layout independent, can be used to concatenate either int8 or uint8 tensors because it
 *       does not involve any arithmetic operation
 *
 * @param[in]  input    Pointer to input tensor
 * @param[in]  input_x  Width of input tensor
 * @param[in]  input_y  Height of input tensor
 * @param[in]  input_z  Channels in input tensor
 * @param[in]  input_w  Batch size in input tensor
 * @param[out] output   Pointer to output tensor. Expected to be at least
 *                          input_x * input_y * input_z * input_w
 *                      bytes.
 * @param[in]  offset_w The offset on the W axis to start concatenating the input tensor
 *                      It is user responsibility to provide the correct value
 *
 */
void muriscv_nn_concatenation_s8_w(const int8_t *input,
                                   const uint16_t input_x,
                                   const uint16_t input_y,
                                   const uint16_t input_z,
                                   const uint16_t input_w,
                                   int8_t *output,
                                   const uint32_t offset_w);
/**
 * @defgroup SVDF SVDF Layer Functions
 *
 */

/**
 * @brief s8 SVDF function with 8 bit state tensor and 8 bit time weights
 *
 * @param[in]   input_ctx Temporary scratch buffer
 * @param[in]   output_ctx Temporary output scratch buffer
 * @param[in]   svdf_params SVDF Parameters
 *              Range of svdf_params->input_offset  : [-128, 127]
 *              Range of svdf_params->output_offset  : [-128, 127]
 * @param[in]   input_quant_params Input quantization parameters
 * @param[in]   output_quant_params Output quantization parameters
 * @param[in]   input_dims Input tensor dimensions
 * @param[in]   input_data Pointer to input tensor
 * @param[in]   state_dims State tensor dimensions
 * @param[in]   state_data Pointer to state tensor
 * @param[in]   weights_feature_dims Weights (feature) tensor dimensions
 * @param[in]   weights_feature_data Pointer to the weights (feature) tensor
 * @param[in]   weights_time_dims Weights (time) tensor dimensions
 * @param[in]   weights_time_data Pointer to the weights (time) tensor
 * @param[in]   bias_dims Bias tensor dimensions
 * @param[in]   bias_data Pointer to bias tensor
 * @param[in]   output_dims Output tensor dimensions
 * @param[out]  output_data Pointer to the output tensor
 *
 * @return     The function returns <code>MURISCV_NN_SUCCESS</code>
 *
 * @details
 *    1. Supported framework: TensorFlow Lite micro
 *    2. q7 is used as data type eventhough it is s8 data. It is done so to be consistent with existing APIs.
 *
 */
muriscv_nn_status muriscv_nn_svdf_s8(const muriscv_nn_context *input_ctx,
                                     const muriscv_nn_context *output_ctx,
                                     const muriscv_nn_svdf_params *svdf_params,
                                     const muriscv_nn_per_tensor_quant_params *input_quant_params,
                                     const muriscv_nn_per_tensor_quant_params *output_quant_params,
                                     const muriscv_nn_dims *input_dims,
                                     const q7_t *input_data,
                                     const muriscv_nn_dims *state_dims,
                                     q7_t *state_data,
                                     const muriscv_nn_dims *weights_feature_dims,
                                     const q7_t *weights_feature_data,
                                     const muriscv_nn_dims *weights_time_dims,
                                     const q7_t *weights_time_data,
                                     const muriscv_nn_dims *bias_dims,
                                     const q31_t *bias_data,
                                     const muriscv_nn_dims *output_dims,
                                     q7_t *output_data);

/**
 * @brief s8 SVDF function with 16 bit state tensor and 16 bit time weights
 *
 * @param[in]   input_ctx Temporary scratch buffer
 * @param[in]   output_ctx Temporary output scratch buffer
 * @param[in]   svdf_params SVDF Parameters
 *              Range of svdf_params->input_offset  : [-128, 127]
 *              Range of svdf_params->output_offset  : [-128, 127]
 * @param[in]   input_quant_params Input quantization parameters
 * @param[in]   output_quant_params Output quantization parameters
 * @param[in]   input_dims Input tensor dimensions
 * @param[in]   input_data Pointer to input tensor
 * @param[in]   state_dims State tensor dimensions
 * @param[in]   state_data Pointer to state tensor
 * @param[in]   weights_feature_dims Weights (feature) tensor dimensions
 * @param[in]   weights_feature_data Pointer to the weights (feature) tensor
 * @param[in]   weights_time_dims Weights (time) tensor dimensions
 * @param[in]   weights_time_data Pointer to the weights (time) tensor
 * @param[in]   bias_dims Bias tensor dimensions
 * @param[in]   bias_data Pointer to bias tensor
 * @param[in]   output_dims Output tensor dimensions
 * @param[out]  output_data Pointer to the output tensor
 *
 * @return     The function returns <code>MURISCV_NN_SUCCESS</code>
 *
 * @details
 *    1. Supported framework: TensorFlow Lite micro
 *    2. q7 is used as data type eventhough it is s8 data. It is done so to be consistent with existing APIs.
 *
 */
muriscv_nn_status muriscv_nn_svdf_state_s16_s8(const muriscv_nn_context *input_ctx,
                                               const muriscv_nn_context *output_ctx,
                                               const muriscv_nn_svdf_params *svdf_params,
                                               const muriscv_nn_per_tensor_quant_params *input_quant_params,
                                               const muriscv_nn_per_tensor_quant_params *output_quant_params,
                                               const muriscv_nn_dims *input_dims,
                                               const q7_t *input_data,
                                               const muriscv_nn_dims *state_dims,
                                               q15_t *state_data,
                                               const muriscv_nn_dims *weights_feature_dims,
                                               const q7_t *weights_feature_data,
                                               const muriscv_nn_dims *weights_time_dims,
                                               const q15_t *weights_time_data,
                                               const muriscv_nn_dims *bias_dims,
                                               const q31_t *bias_data,
                                               const muriscv_nn_dims *output_dims,
                                               q7_t *output_data);

#ifdef __cplusplus
}
#endif

#endif /* _MURISCV_NN_FUNCTIONS_H */
