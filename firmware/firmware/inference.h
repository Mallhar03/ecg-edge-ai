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
#define MODEL_IN_CHANNELS       2
#define MODEL_NUM_CLASSES       5

// Class indices matching config.yaml (AAMI EC57 Standard)
#define CLASS_IDX_SUPRAVENTRICULAR    0 // S: Atrial / junctional premature ectopy
#define CLASS_IDX_VENTRICULAR         1 // V: Premature ventricular contractions (PVCs)
#define CLASS_IDX_BUNDLE_BRANCH_BLOCK 2 // B: Left / Right Bundle Branch Block
#define CLASS_IDX_PACED_OTHER         3 // Q: Pacemaker rhythms / unclassifiable
#define CLASS_IDX_NORMAL              4 // N: Normal sinus rhythm

// Human-readable class names
static const char * const CLASS_NAMES[MODEL_NUM_CLASSES] = {
    "Supraventricular",
    "Ventricular",
    "Bundle_Branch_Block",
    "Paced_Other",
    "Normal"
};

/**
 * Initialize INT8 CNN inference engine.
 * @return true if initialized successfully, false otherwise.
 */
bool inference_init();

/**
 * Execute forward pass of INT8 Multi-Scale 1D-CNN model on a 256-sample window
 * with auxiliary RR-interval timing ratio features.
 * 
 * @param input_window Pointer to array of 256 INT8 normalized samples.
 * @param pre_rr Local median-normalized pre-RR timing ratio (R_i - R_{i-1}) / RR_med.
 * @param post_rr Local median-normalized post-RR timing ratio (R_{i+1} - R_i) / RR_med.
 * @param output_logits Output array of 5 floats for raw logits.
 * @param predicted_class Output pointer receiving the argmax / thresholded class index.
 * @return true if inference executed successfully, false otherwise.
 */
bool inference_run(const int8_t *input_window, float pre_rr, float post_rr, float *output_logits, uint8_t *predicted_class);

#endif // INFERENCE_H
