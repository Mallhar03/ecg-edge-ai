#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdint.h>
#include <stdbool.h>

/**
 * ============================================================================
 * INT8 CNN INFERENCE ENGINE STUB FOR ARDUINO GIGA R1
 * Target Hardware: Arduino Giga R1 WiFi (STM32H747XI)
 * ============================================================================
 * 
 * TODO: Implement INT8 Multi-Scale 1D-CNN inference pipeline once front-end
 * ADC signal acquisition distortion (Aug 3 EOD report) is fully resolved.
 * 
 * Target Model Specs:
 * - Window Size : 256 samples (~711 ms @ 360 Hz)
 * - Input Type  : INT8 normalized ECG window
 * - Class Output: 5 classes (ST_segment, QT_interval, P_wave, BBB, Normal)
 */

#define MODEL_INPUT_WINDOW_SIZE 256
#define MODEL_NUM_CLASSES 5

// Class indices (IMMUTABLE order matching config.yaml)
#define CLASS_IDX_ST_SEGMENT          0
#define CLASS_IDX_QT_INTERVAL         1
#define CLASS_IDX_P_WAVE              2
#define CLASS_IDX_BUNDLE_BRANCH_BLOCK 3
#define CLASS_IDX_NORMAL              4

/**
 * Initialize INT8 CNN inference engine (load weights from weights.h).
 * @return true if initialized successfully, false otherwise.
 */
bool inference_init();

/**
 * Execute forward pass of INT8 Multi-Scale 1D-CNN model on a 256-sample window.
 * 
 * TODO: This is a stub function. Full INT8 MAC matrix multiplication, ReLU, 
 * AdaptiveAvgPool1d, and Dense linear layers will be added here.
 * 
 * @param input_window Pointer to array of 256 INT8 normalized samples.
 * @param output_logits Output array of 5 floats for raw logits.
 * @return true if inference executed successfully, false otherwise.
 */
bool inference_run(const int8_t *input_window, float *output_logits);

#endif // INFERENCE_H
