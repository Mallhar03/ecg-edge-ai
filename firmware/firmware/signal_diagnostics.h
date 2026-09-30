#ifndef SIGNAL_DIAGNOSTICS_H
#define SIGNAL_DIAGNOSTICS_H

#include <Arduino.h>
#include "adc_acquisition.h"

/**
 * ============================================================================
 * SIGNAL DIAGNOSTICS & LOGGING UTILITIES
 * Target Hardware: Arduino Giga R1 WiFi (STM32H747XI)
 * ============================================================================
 * Utilities to stream raw ADC samples over Serial in CSV format (timestamp,raw_value)
 * and analyze signal statistics (min, max, mean, peak-to-peak, clipping alerts)
 * to assist in debugging front-end signal distortion (Aug 3 EOD Report).
 */

struct DiagnosticStats {
    uint32_t total_samples;
    uint16_t min_raw;
    uint16_t max_raw;
    uint32_t sum_raw;
    uint32_t clipping_high_count; // Samples at ADC_MAX_RAW_VALUE (saturation)
    uint32_t clipping_low_count;  // Samples at 0 (cutoff)
};

/**
 * Initialize signal diagnostics module.
 */
void signal_diagnostics_init();

/**
 * Process a sample and stream its values out over Serial in CSV format:
 * timestamp_ms,raw_value
 * 
 * @param sample Reference to the acquired AdcSample.
 */
void signal_diagnostics_log_csv(const AdcSample &sample);

/**
 * Reset accumulated signal statistics counters.
 */
void signal_diagnostics_reset_stats();

/**
 * Update diagnostic statistical trackers with a new sample.
 * @param sample Reference to current sample.
 */
void signal_diagnostics_update_stats(const AdcSample &sample);

/**
 * Print diagnostic summary report over Serial.
 */
void signal_diagnostics_print_summary();

#endif // SIGNAL_DIAGNOSTICS_H
