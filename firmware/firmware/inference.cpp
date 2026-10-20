#include "inference.h"
#include <math.h>
#include <string.h>

#if __has_include("weights.h")
#include "weights.h"
#define HAVE_WEIGHTS_H 1
#else
#define HAVE_WEIGHTS_H 0
#endif

#if defined(ARDUINO)
#include <Arduino.h>
#else
#include <stdio.h>
#endif

/**
 * ============================================================================
 * EMBEDDED INT8 MULTI-SCALE 1D-CNN INFERENCE ENGINE
 * Target Hardware: Arduino Giga R1 WiFi (STM32H747XI Dual Core Cortex-M7 @ 480 MHz)
 * 
 * Clinical Baseline: MIT-BIH Arrhythmia Database, AAMI EC57 5-Class Standard
 * Class Indices:
 *   0: Supraventricular (S)
 *   1: Ventricular (V)
 *   2: Bundle Branch Block (B)
 *   3: Paced / Other (Q)
 *   4: Normal (N)
 * ============================================================================
 */

#define NUM_BRANCHES 3
#define NUM_CHANNELS 32
#define POOL_SIZE 16
#define BRANCH_OUT_DIM (NUM_CHANNELS * 2 * POOL_SIZE) // 32 * 32 = 1024
#define CONCAT_DIM (NUM_BRANCHES * BRANCH_OUT_DIM)     // 3 * 1024 = 3072
#define FC1_IN_DIM (CONCAT_DIM + 2)                    // 3072 + 2 = 3074
#define FC1_OUT_DIM 64
#define FC2_OUT_DIM 5
#define SE_MID_DIM 8

// Kernel sizes for the 3 parallel branches
static const int KERNEL_SIZES[NUM_BRANCHES] = {3, 5, 7};

// Static scratchpad buffers in internal SRAM (zero dynamic heap allocation)
static float s_x[MODEL_IN_CHANNELS * MODEL_INPUT_WINDOW_SIZE];
static float s_conv1_buf[NUM_CHANNELS][MODEL_INPUT_WINDOW_SIZE];
static float s_conv2_buf[NUM_CHANNELS][MODEL_INPUT_WINDOW_SIZE];
static float s_concat_features[FC1_IN_DIM];
static float s_fc1_out[FC1_OUT_DIM];

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

bool inference_init() {
#if HAVE_WEIGHTS_H
#if defined(ARDUINO)
    Serial.println("# INFERENCE: Engine initialized with Brevitas QAT weights.h.");
#else
    printf("# INFERENCE: Engine initialized with Brevitas QAT weights.h.\n");
#endif
    return true;
#else
#if defined(ARDUINO)
    Serial.println("# INFERENCE: Running in standalone fallback mode (weights.h pending QAT export).");
#else
    printf("# INFERENCE: Running in standalone fallback mode (weights.h pending QAT export).\n");
#endif
    return true;
#endif
}

/**
 * Execute a single 1D Conv Branch with Double Conv1D + BatchNorm + ReLU + SE Attention + Dual Pool
 */
static void run_conv_branch(
    int branch_idx,
    const float *input_signal,
    float *branch_out_features
) {
    int k = KERNEL_SIZES[branch_idx];
    int pad = k / 2;

#if HAVE_WEIGHTS_H
    // Pointer lookup based on branch index
    const int8_t *w_conv1 = nullptr;
    const float *b_conv1 = nullptr;
    float scale_conv1 = 1.0f;

    const float *bn1_gamma = nullptr;
    const float *bn1_beta = nullptr;
    const float *bn1_mean = nullptr;
    const float *bn1_var = nullptr;

    const int8_t *w_conv2 = nullptr;
    const float *b_conv2 = nullptr;
    float scale_conv2 = 1.0f;

    const float *bn2_gamma = nullptr;
    const float *bn2_beta = nullptr;
    const float *bn2_mean = nullptr;
    const float *bn2_var = nullptr;

    const float *w_se0 = nullptr;
    const float *w_se2 = nullptr;

    if (branch_idx == 0) {
        w_conv1 = layer_branches_0_conv1_weights;
        scale_conv1 = layer_branches_0_conv1_scale;
        b_conv1 = layer_branches_0_conv1_bias;
        bn1_gamma = layer_branches_0_bn1_gamma;
        bn1_beta = layer_branches_0_bn1_beta;
        bn1_mean = layer_branches_0_bn1_mean;
        bn1_var = layer_branches_0_bn1_var;

        w_conv2 = layer_branches_0_conv2_weights;
        scale_conv2 = layer_branches_0_conv2_scale;
        b_conv2 = layer_branches_0_conv2_bias;
        bn2_gamma = layer_branches_0_bn2_gamma;
        bn2_beta = layer_branches_0_bn2_beta;
        bn2_mean = layer_branches_0_bn2_mean;
        bn2_var = layer_branches_0_bn2_var;

        w_se0 = layer_branches_0_se_excite_0_weights;
        w_se2 = layer_branches_0_se_excite_2_weights;
    } else if (branch_idx == 1) {
        w_conv1 = layer_branches_1_conv1_weights;
        scale_conv1 = layer_branches_1_conv1_scale;
        b_conv1 = layer_branches_1_conv1_bias;
        bn1_gamma = layer_branches_1_bn1_gamma;
        bn1_beta = layer_branches_1_bn1_beta;
        bn1_mean = layer_branches_1_bn1_mean;
        bn1_var = layer_branches_1_bn1_var;

        w_conv2 = layer_branches_1_conv2_weights;
        scale_conv2 = layer_branches_1_conv2_scale;
        b_conv2 = layer_branches_1_conv2_bias;
        bn2_gamma = layer_branches_1_bn2_gamma;
        bn2_beta = layer_branches_1_bn2_beta;
        bn2_mean = layer_branches_1_bn2_mean;
        bn2_var = layer_branches_1_bn2_var;

        w_se0 = layer_branches_1_se_excite_0_weights;
        w_se2 = layer_branches_1_se_excite_2_weights;
    } else {
        w_conv1 = layer_branches_2_conv1_weights;
        scale_conv1 = layer_branches_2_conv1_scale;
        b_conv1 = layer_branches_2_conv1_bias;
        bn1_gamma = layer_branches_2_bn1_gamma;
        bn1_beta = layer_branches_2_bn1_beta;
        bn1_mean = layer_branches_2_bn1_mean;
        bn1_var = layer_branches_2_bn1_var;

        w_conv2 = layer_branches_2_conv2_weights;
        scale_conv2 = layer_branches_2_conv2_scale;
        b_conv2 = layer_branches_2_conv2_bias;
        bn2_gamma = layer_branches_2_bn2_gamma;
        bn2_beta = layer_branches_2_bn2_beta;
        bn2_mean = layer_branches_2_bn2_mean;
        bn2_var = layer_branches_2_bn2_var;

        w_se0 = layer_branches_2_se_excite_0_weights;
        w_se2 = layer_branches_2_se_excite_2_weights;
    }

    // --- 1. Conv1 (in_channels=MODEL_IN_CHANNELS, out_channels=32, kernel=k) + BN1 + ReLU ---
    for (int c = 0; c < NUM_CHANNELS; c++) {
        float b = b_conv1 ? b_conv1[c] : 0.0f;
        float inv_std = 1.0f / sqrtf(bn1_var[c] + 1e-5f);
        float g = bn1_gamma[c] * inv_std;
        float eff_scale = scale_conv1 * g;
        float eff_bias = b * g + (bn1_beta[c] - bn1_mean[c] * g);
        const int8_t *w_c = w_conv1 + c * (MODEL_IN_CHANNELS * k);

        for (int t = 0; t < MODEL_INPUT_WINDOW_SIZE; t++) {
            float sum = 0.0f;
            for (int in_c = 0; in_c < MODEL_IN_CHANNELS; in_c++) {
                const int8_t *w_ptr = w_c + in_c * k;
                const float *in_ptr = &input_signal[in_c * MODEL_INPUT_WINDOW_SIZE];
                for (int j = 0; j < k; j++) {
                    int in_idx = t + j - pad;
                    if ((unsigned)in_idx < (unsigned)MODEL_INPUT_WINDOW_SIZE) {
                        sum += (float)w_ptr[j] * in_ptr[in_idx];
                    }
                }
            }
            float val = sum * eff_scale + eff_bias;
            s_conv1_buf[c][t] = val > 0.0f ? val : 0.0f; // ReLU
        }
    }

    // --- 2. Conv2 (in_channels=32, out_channels=32, kernel=k) + BN2 + ReLU ---
    for (int c = 0; c < NUM_CHANNELS; c++) {
        float b = b_conv2 ? b_conv2[c] : 0.0f;
        float inv_std = 1.0f / sqrtf(bn2_var[c] + 1e-5f);
        float g = bn2_gamma[c] * inv_std;
        float eff_scale = scale_conv2 * g;
        float eff_bias = b * g + (bn2_beta[c] - bn2_mean[c] * g);
        const int8_t *w_c = w_conv2 + c * NUM_CHANNELS * k;

        for (int t = 0; t < MODEL_INPUT_WINDOW_SIZE; t++) {
            float sum = 0.0f;
            for (int in_c = 0; in_c < NUM_CHANNELS; in_c++) {
                const int8_t *w_ptr = w_c + in_c * k;
                const float *in_ptr = s_conv1_buf[in_c];
                for (int j = 0; j < k; j++) {
                    int in_idx = t + j - pad;
                    if ((unsigned)in_idx < (unsigned)MODEL_INPUT_WINDOW_SIZE) {
                        sum += (float)w_ptr[j] * in_ptr[in_idx];
                    }
                }
            }
            float val = sum * eff_scale + eff_bias;
            s_conv2_buf[c][t] = val > 0.0f ? val : 0.0f; // ReLU
        }
    }

    // --- 3. Squeeze-and-Excitation (SE Attention) ---
    float squeeze[NUM_CHANNELS];
    for (int c = 0; c < NUM_CHANNELS; c++) {
        float sum = 0.0f;
        for (int t = 0; t < MODEL_INPUT_WINDOW_SIZE; t++) {
            sum += s_conv2_buf[c][t];
        }
        squeeze[c] = sum / (float)MODEL_INPUT_WINDOW_SIZE;
    }

    // SE FC1: 32 -> 8 with ReLU
    float se_mid[SE_MID_DIM];
    for (int m = 0; m < SE_MID_DIM; m++) {
        float sum = 0.0f;
        for (int c = 0; c < NUM_CHANNELS; c++) {
            sum += w_se0[m * NUM_CHANNELS + c] * squeeze[c];
        }
        se_mid[m] = sum > 0.0f ? sum : 0.0f;
    }

    // SE FC2: 8 -> 32 with Sigmoid
    float excite[NUM_CHANNELS];
    for (int c = 0; c < NUM_CHANNELS; c++) {
        float sum = 0.0f;
        for (int m = 0; m < SE_MID_DIM; m++) {
            sum += w_se2[c * SE_MID_DIM + m] * se_mid[m];
        }
        excite[c] = sigmoid(sum);
    }

    // Apply excitation scale
    for (int c = 0; c < NUM_CHANNELS; c++) {
        float s = excite[c];
        for (int t = 0; t < MODEL_INPUT_WINDOW_SIZE; t++) {
            s_conv2_buf[c][t] *= s;
        }
    }

    // --- 4. Dual Pooling: MaxPool(16) and AvgPool(16) ---
    // Output size = 16 intervals, each of length 256 / 16 = 16 samples
    const int bin_size = MODEL_INPUT_WINDOW_SIZE / POOL_SIZE; // 16
    for (int c = 0; c < NUM_CHANNELS; c++) {
        for (int p = 0; p < POOL_SIZE; p++) {
            int start_idx = p * bin_size;
            float max_val = s_conv2_buf[c][start_idx];
            float sum_val = 0.0f;

            for (int s = 0; s < bin_size; s++) {
                float v = s_conv2_buf[c][start_idx + s];
                if (v > max_val) max_val = v;
                sum_val += v;
            }

            // Dual pool layout: [C, 2 * pool_size]
            // First 16 are MaxPool, next 16 are AvgPool
            branch_out_features[c * (2 * POOL_SIZE) + p] = max_val;
            branch_out_features[c * (2 * POOL_SIZE) + POOL_SIZE + p] = sum_val / (float)bin_size;
        }
    }
#else
    // Fallback stub: zero fill features
    for (int i = 0; i < BRANCH_OUT_DIM; i++) {
        branch_out_features[i] = 0.0f;
    }
#endif
}

bool inference_run(
    const int8_t *input_window,
    float pre_rr,
    float post_rr,
    float *output_logits,
    uint8_t *predicted_class
) {
    if (input_window == nullptr || output_logits == nullptr) {
        return false;
    }

    // 1. Convert input window to float (normalize INT8 scale ~ 32.0f)
    for (int i = 0; i < MODEL_IN_CHANNELS * MODEL_INPUT_WINDOW_SIZE; i++) {
        s_x[i] = (float)input_window[i] / 32.0f;
    }

#if HAVE_WEIGHTS_H
    // 2. Execute parallel branches
    for (int b = 0; b < NUM_BRANCHES; b++) {
        run_conv_branch(b, s_x, &s_concat_features[b * BRANCH_OUT_DIM]);
    }

    // 3. Append RR-interval timing features
    s_concat_features[CONCAT_DIM]     = pre_rr;
    s_concat_features[CONCAT_DIM + 1] = post_rr;

    // 4. FC1 (3074 -> 64) + ReLU
    for (int i = 0; i < FC1_OUT_DIM; i++) {
        float sum = layer_fc1_bias ? layer_fc1_bias[i] : 0.0f;
        int row_offset = i * FC1_IN_DIM;
        for (int j = 0; j < FC1_IN_DIM; j++) {
            sum += ((float)layer_fc1_weights[row_offset + j] * layer_fc1_scale) * s_concat_features[j];
        }
        s_fc1_out[i] = sum > 0.0f ? sum : 0.0f; // ReLU
    }

    // 5. FC2 (64 -> 5)
    for (int k = 0; k < FC2_OUT_DIM; k++) {
        float sum = layer_fc2_bias ? layer_fc2_bias[k] : 0.0f;
        int row_offset = k * FC1_OUT_DIM;
        for (int i = 0; i < FC1_OUT_DIM; i++) {
            sum += ((float)layer_fc2_weights[row_offset + i] * layer_fc2_scale) * s_fc1_out[i];
        }
        output_logits[k] = sum;
    }

    // 6. Argmax Prediction
    uint8_t best_c = 0;
    float best_logit = output_logits[0];
    for (uint8_t k = 1; k < MODEL_NUM_CLASSES; k++) {
        if (output_logits[k] > best_logit) {
            best_logit = output_logits[k];
            best_c = k;
        }
    }

    if (predicted_class != nullptr) {
        *predicted_class = best_c;
    }

#else
    // Fallback mode: Set Normal class
    for (int i = 0; i < MODEL_NUM_CLASSES; i++) {
        output_logits[i] = 0.0f;
    }
    output_logits[CLASS_IDX_NORMAL] = 1.0f;
    if (predicted_class != nullptr) {
        *predicted_class = CLASS_IDX_NORMAL;
    }
#endif

    return true;
}
