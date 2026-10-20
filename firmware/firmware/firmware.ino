/**
 * ============================================================================
 * ECG ARRHYTHMIA ACCELERATOR — ARDUINO GIGA R1 FIRMWARE
 * Main Arduino Sketch: 16-Bit ADC Sampling, Real-Time QRS Detection,
 * Window Extraction, and INT8 CNN Inference Execution
 * Target Hardware: Arduino Giga R1 WiFi (STM32H747XI Dual Core ARM Cortex-M7 @ 480 MHz)
 * ============================================================================
 */

#include "adc_acquisition.h"
#include "signal_diagnostics.h"
#include "inference.h"
#include <math.h>

// Periodic Diagnostic Summary Interval (every 10 seconds)
static const uint32_t SUMMARY_INTERVAL_MS = 10000;
static uint32_t last_summary_ms = 0;

// Circular Sample Buffer
#define CIRCULAR_BUFFER_SIZE 512
#define WINDOW_PRE_R_SAMPLES 90
#define WINDOW_POST_R_SAMPLES 165
#define WINDOW_TOTAL_SAMPLES 256 // 90 + 1 + 165 = 256

static int16_t s_raw_buffer[CIRCULAR_BUFFER_SIZE];
static uint32_t s_sample_count = 0;

// RR-Interval Tracking for Timing Ratios
#define RR_HISTORY_SIZE 8
static uint32_t s_last_r_sample_idx = 0;
static uint32_t s_rr_intervals[RR_HISTORY_SIZE] = {360, 360, 360, 360, 360, 360, 360, 360}; // Default ~1s (60 bpm)
static uint8_t s_rr_idx = 0;

// Pending Beat Inference State Machine
static bool s_beat_pending = false;
static uint32_t s_pending_r_sample_idx = 0;
static float s_pending_pre_rr_ratio = 1.0f;

// Real-Time Pan-Tompkins Style Derivative & Moving Window QRS Detector
static int16_t s_prev_sample_1 = 0;
static int16_t s_prev_sample_2 = 0;
static uint32_t s_mwi_sum = 0;
#define MWI_WINDOW_SIZE 16
static uint32_t s_mwi_buffer[MWI_WINDOW_SIZE] = {0};
static uint8_t s_mwi_idx = 0;

// Adaptive Peak Thresholds
static uint32_t s_signal_level = 50000;
static uint32_t s_noise_level = 10000;
static uint32_t s_last_peak_sample_idx = 0;
static const uint32_t REFRACTORY_PERIOD_SAMPLES = 72; // ~200 ms @ 360 Hz

/**
 * Compute running median of RR intervals for local heart rate baseline normalization.
 */
static float get_median_rr() {
    uint32_t sorted[RR_HISTORY_SIZE];
    for (int i = 0; i < RR_HISTORY_SIZE; i++) {
        sorted[i] = s_rr_intervals[i];
    }
    // Bubble sort 8 elements
    for (int i = 0; i < RR_HISTORY_SIZE - 1; i++) {
        for (int j = 0; j < RR_HISTORY_SIZE - i - 1; j++) {
            if (sorted[j] > sorted[j + 1]) {
                uint32_t tmp = sorted[j];
                sorted[j] = sorted[j + 1];
                sorted[j + 1] = tmp;
            }
        }
    }
    return (float)(sorted[3] + sorted[4]) / 2.0f;
}

/**
 * Process single incoming ADC sample through real-time QRS peak detection.
 * Returns true if an R-peak was identified.
 */
static bool detect_r_peak(int16_t sample, uint32_t current_idx) {
    // 1. Derivative filter: d = x[n] - x[n-2]
    int32_t diff = (int32_t)sample - (int32_t)s_prev_sample_2;
    s_prev_sample_2 = s_prev_sample_1;
    s_prev_sample_1 = sample;

    // 2. Square: non-linear amplitude amplification
    uint32_t squared = (uint32_t)(diff * diff);

    // 3. Moving Window Integrator (MWI)
    s_mwi_sum -= s_mwi_buffer[s_mwi_idx];
    s_mwi_buffer[s_mwi_idx] = squared;
    s_mwi_sum += squared;
    s_mwi_idx = (s_mwi_idx + 1) % MWI_WINDOW_SIZE;
    uint32_t mwi_val = s_mwi_sum / MWI_WINDOW_SIZE;

    // Dynamic threshold: Thresh = Noise + 0.25 * (Signal - Noise)
    uint32_t threshold = s_noise_level + ((s_signal_level - s_noise_level) >> 2);

    // Refractory blanking check
    if (current_idx - s_last_peak_sample_idx < REFRACTORY_PERIOD_SAMPLES) {
        return false;
    }

    if (mwi_val > threshold) {
        // Confirmed R-peak candidate
        s_signal_level = (s_signal_level * 7 + mwi_val) / 8;
        s_last_peak_sample_idx = current_idx;
        return true;
    } else {
        // Track ambient noise level
        s_noise_level = (s_noise_level * 31 + mwi_val) / 32;
        return false;
    }
}

/**
 * Extract 256-sample window centered at R-peak, perform Z-score normalization,
 * quantize to INT8, and execute CNN inference.
 */
static void process_beat_inference(uint32_t r_peak_idx, float pre_rr, float post_rr) {
    int8_t window_int8[WINDOW_TOTAL_SAMPLES];

    // Compute mean
    float sum = 0.0f;
    for (int i = 0; i < WINDOW_TOTAL_SAMPLES; i++) {
        int buf_idx = (int)(r_peak_idx - WINDOW_PRE_R_SAMPLES + i) % CIRCULAR_BUFFER_SIZE;
        if (buf_idx < 0) buf_idx += CIRCULAR_BUFFER_SIZE;
        sum += (float)s_raw_buffer[buf_idx];
    }
    float mean = sum / (float)WINDOW_TOTAL_SAMPLES;

    // Compute variance & std
    float var_sum = 0.0f;
    for (int i = 0; i < WINDOW_TOTAL_SAMPLES; i++) {
        int buf_idx = (int)(r_peak_idx - WINDOW_PRE_R_SAMPLES + i) % CIRCULAR_BUFFER_SIZE;
        if (buf_idx < 0) buf_idx += CIRCULAR_BUFFER_SIZE;
        float diff = (float)s_raw_buffer[buf_idx] - mean;
        var_sum += diff * diff;
    }
    float std_dev = sqrtf(var_sum / (float)WINDOW_TOTAL_SAMPLES);
    if (std_dev < 1.0f) std_dev = 1.0f; // Prevent div by zero

    // Scale to INT8 [-128, 127] with scale 32.0f
    for (int i = 0; i < WINDOW_TOTAL_SAMPLES; i++) {
        int buf_idx = (int)(r_peak_idx - WINDOW_PRE_R_SAMPLES + i) % CIRCULAR_BUFFER_SIZE;
        if (buf_idx < 0) buf_idx += CIRCULAR_BUFFER_SIZE;
        float norm_val = ((float)s_raw_buffer[buf_idx] - mean) / std_dev;
        int32_t q = (int32_t)roundf(norm_val * 32.0f);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        window_int8[i] = (int8_t)q;
    }

    // Execute Inference & Measure Latency
    float logits[MODEL_NUM_CLASSES];
    uint8_t predicted_class = CLASS_IDX_NORMAL;

    uint32_t t_start = micros();
    bool success = inference_run(window_int8, pre_rr, post_rr, logits, &predicted_class);
    uint32_t t_latency_us = micros() - t_start;

    if (success) {
        uint32_t timestamp_ms = (r_peak_idx * 1000UL) / SAMPLE_RATE_HZ;
        float hr_bpm = (pre_rr > 0.01f) ? (60.0f * (float)SAMPLE_RATE_HZ / (pre_rr * get_median_rr())) : 0.0f;

        Serial.print("[BEAT] T=");
        Serial.print(timestamp_ms);
        Serial.print("ms | Class=");
        Serial.print(CLASS_NAMES[predicted_class]);
        Serial.print(" | HR=");
        Serial.print((int)hr_bpm);
        Serial.print(" bpm | pre_RR=");
        Serial.print(pre_rr, 2);
        Serial.print(" | Latency=");
        Serial.print((float)t_latency_us / 1000.0f, 2);
        Serial.println("ms");

        // High-severity alarm notification for ectopic beats
        if (predicted_class == CLASS_IDX_VENTRICULAR) {
            Serial.println("  >>> ALERT: Premature Ventricular Contraction (PVC) Detected! <<<");
#if defined(LED_RED)
            digitalWrite(LED_RED, LOW); // Active LOW on Giga R1
            delayMicroseconds(50);
            digitalWrite(LED_RED, HIGH);
#endif
        } else if (predicted_class == CLASS_IDX_SUPRAVENTRICULAR) {
            Serial.println("  >>> ALERT: Supraventricular Ectopic Beat (PAC/SVEB) Detected! <<<");
        }
    }
}

void setup() {
    Serial.begin(115200);

    uint32_t serial_timeout = millis() + 3000;
    while (!Serial && millis() < serial_timeout) {
        delay(10);
    }

    Serial.println("\n========================================================");
    Serial.println("  ECG-EDGE ACCELERATOR — ARDUINO GIGA R1 FIRMWARE");
    Serial.println("  Dual-Core Cortex-M7 Real-Time INT8 Arrhythmia Engine");
    Serial.println("  Standard: AAMI EC57 (S, V, B, Q, N) @ 360 Hz");
    Serial.println("========================================================\n");

#if defined(LED_RED)
    pinMode(LED_RED, OUTPUT);
    digitalWrite(LED_RED, HIGH); // Off
#endif

    // 1. Initialize 16-bit ADC Hardware Peripheral at 360 Hz
    if (!adc_acquisition_init()) {
        Serial.println("ERROR: ADC Hardware Initialization Failed!");
    } else {
        Serial.println("SUCCESS: 16-Bit ADC Initialized at 360 Hz (Settling: 810.5 cycles).");
    }

    // 2. Initialize Signal Quality Inspection Utilities
    signal_diagnostics_init();

    // 3. Initialize INT8 CNN Inference Engine
    inference_init();

    last_summary_ms = millis();
}

void loop() {
    AdcSample sample;

    // Read non-blocking sample at 360 Hz (~2.777 ms interval)
    if (adc_acquisition_read(sample)) {
        // Store in circular buffer (centered around 1.65V baseline ~32768)
        int16_t centered_val = (int16_t)((int32_t)sample.raw_value - 32768);
        s_raw_buffer[s_sample_count % CIRCULAR_BUFFER_SIZE] = centered_val;

        // Run real-time QRS detection
        if (detect_r_peak(centered_val, s_sample_count)) {
            uint32_t current_r_idx = s_sample_count;

            if (s_last_r_sample_idx > 0) {
                uint32_t rr = current_r_idx - s_last_r_sample_idx;
                s_rr_intervals[s_rr_idx] = rr;
                s_rr_idx = (s_rr_idx + 1) % RR_HISTORY_SIZE;

                float median_rr = get_median_rr();
                float pre_rr_ratio = (float)rr / median_rr;

                // Schedule beat window extraction once 166 post-R samples arrive
                s_beat_pending = true;
                s_pending_r_sample_idx = current_r_idx;
                s_pending_pre_rr_ratio = pre_rr_ratio;
            }
            s_last_r_sample_idx = current_r_idx;
        }

        // Check if scheduled post-R samples have arrived for pending beat
        if (s_beat_pending && (s_sample_count >= s_pending_r_sample_idx + WINDOW_POST_R_SAMPLES)) {
            float post_rr_ratio = 1.0f; // Nominal post-RR until next beat arrives
            process_beat_inference(s_pending_r_sample_idx, s_pending_pre_rr_ratio, post_rr_ratio);
            s_beat_pending = false;
        }

        s_sample_count++;
    }

    // Periodically print diagnostic summary report over Serial (every 10 seconds)
    if (millis() - last_summary_ms >= SUMMARY_INTERVAL_MS) {
        signal_diagnostics_print_summary();
        last_summary_ms = millis();
    }
}
