/*
 * Copyright (C) 2021-2022 Chair of Electronic Design Automation, TUM.
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
 */

#include <muriscv_nn_functions.h>
#include <stdio.h>
#include <stdlib.h>
#include <unity.h>

#include "../../TestData/test_muriscv_nn_mul_s16/test_data.h"
#include "../../Utils/validate.h"

#define TOLERANCE 0

void setUp(void)
{ /* set stuff up here */
}

void tearDown(void)
{ /* clean stuff up here */
}

void test_muriscv_nn_mul_1_s16(void)
{
    int16_t output[MUL_1_BLOCK_SIZE] = {0};
    muriscv_nn_status result = muriscv_nn_elementwise_mul_s16(mul_1_input_1,
                                                              mul_1_input_2,
                                                              MUL_1_INPUT_1_OFFSET,
                                                              MUL_1_INPUT_2_OFFSET,
                                                              output,
                                                              MUL_1_OUTPUT_OFFSET,
                                                              MUL_1_OUTPUT_MULT,
                                                              MUL_1_OUTPUT_SHIFT,
                                                              MUL_1_OUT_ACTIVATION_MIN,
                                                              MUL_1_OUT_ACTIVATION_MAX,
                                                              MUL_1_BLOCK_SIZE);

    TEST_ASSERT_EQUAL(result, MURISCV_NN_SUCCESS);
    TEST_ASSERT_TRUE(validate_s16(output, mul_1_output_ref, MUL_1_BLOCK_SIZE, TOLERANCE));
}

void test_muriscv_nn_mul_2_s16(void)
{
    int16_t output[MUL_2_BLOCK_SIZE] = {0};
    muriscv_nn_status result = muriscv_nn_elementwise_mul_s16(mul_2_input_1,
                                                              mul_2_input_2,
                                                              MUL_2_INPUT_1_OFFSET,
                                                              MUL_2_INPUT_2_OFFSET,
                                                              output,
                                                              MUL_2_OUTPUT_OFFSET,
                                                              MUL_2_OUTPUT_MULT,
                                                              MUL_2_OUTPUT_SHIFT,
                                                              MUL_2_OUT_ACTIVATION_MIN,
                                                              MUL_2_OUT_ACTIVATION_MAX,
                                                              MUL_2_BLOCK_SIZE);

    TEST_ASSERT_EQUAL(result, MURISCV_NN_SUCCESS);
    TEST_ASSERT_TRUE(validate_s16(output, mul_2_output_ref, MUL_2_BLOCK_SIZE, TOLERANCE));
}

int main(void)
{
    UNITY_BEGIN();

    RUN_TEST(test_muriscv_nn_mul_1_s16);
    RUN_TEST(test_muriscv_nn_mul_2_s16);

#if defined(__riscv) || defined(__riscv__)
    /* If an error occurred make sure the simulator fails so CTest can detect that. */
    int failures = UNITY_END();
    if (failures != 0)
    {
        __asm__ volatile("unimp");
    }
    return failures;
#else
    return UNITY_END();
#endif
}