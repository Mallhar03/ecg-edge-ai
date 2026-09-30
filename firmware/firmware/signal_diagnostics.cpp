#include "signal_diagnostics.h"

static DiagnosticStats stats;

void signal_diagnostics_init() {
    signal_diagnostics_reset_stats();
    // Print CSV header over Serial for diagnostic plotting tools
    // Simple format required by spec: timestamp,raw_value
    Serial.println("timestamp,raw_value");
}

void signal_diagnostics_log_csv(const AdcSample &sample) {
    // Format: timestamp_ms,raw_value
    Serial.print(sample.timestamp_ms);
    Serial.print(",");
    Serial.println(sample.raw_value);

    // Update internal quality metrics
    signal_diagnostics_update_stats(sample);
}

void signal_diagnostics_reset_stats() {
    stats.total_samples = 0;
    stats.min_raw = 65535;
    stats.max_raw = 0;
    stats.sum_raw = 0;
    stats.clipping_high_count = 0;
    stats.clipping_low_count = 0;
}

void signal_diagnostics_update_stats(const AdcSample &sample) {
    stats.total_samples++;
    stats.sum_raw += sample.raw_value;

    if (sample.raw_value < stats.min_raw) {
        stats.min_raw = sample.raw_value;
    }
    if (sample.raw_value > stats.max_raw) {
        stats.max_raw = sample.raw_value;
    }

    // Check for high clipping (saturation at 16-bit max)
    if (sample.raw_value >= (ADC_MAX_RAW_VALUE - 10)) {
        stats.clipping_high_count++;
    }
    // Check for low clipping (cutoff at 0)
    if (sample.raw_value <= 10) {
        stats.clipping_low_count++;
    }
}

void signal_diagnostics_print_summary() {
    if (stats.total_samples == 0) {
        Serial.println("# DIAGNOSTICS: No samples collected yet.");
        return;
    }

    uint32_t avg_raw = stats.sum_raw / stats.total_samples;
    uint32_t p2p_raw = stats.max_raw - stats.min_raw;

    Serial.println("\n--- ECG SIGNAL DIAGNOSTICS SUMMARY ---");
    Serial.print("Total Samples    : "); Serial.println(stats.total_samples);
    Serial.print("Min Raw Value    : "); Serial.print(stats.min_raw); Serial.print(" ("); Serial.print(adc_raw_to_mv(stats.min_raw)); Serial.println(" mV)");
    Serial.print("Max Raw Value    : "); Serial.print(stats.max_raw); Serial.print(" ("); Serial.print(adc_raw_to_mv(stats.max_raw)); Serial.println(" mV)");
    Serial.print("Mean Raw Value   : "); Serial.print(avg_raw); Serial.print(" ("); Serial.print(adc_raw_to_mv(avg_raw)); Serial.println(" mV)");
    Serial.print("Peak-to-Peak     : "); Serial.print(p2p_raw); Serial.print(" ("); Serial.print(adc_raw_to_mv(p2p_raw)); Serial.println(" mV)");
    Serial.print("High Saturation  : "); Serial.print(stats.clipping_high_count); Serial.println(" samples");
    Serial.print("Low Cutoff       : "); Serial.print(stats.clipping_low_count); Serial.println(" samples");
    
    if (stats.clipping_high_count > 0 || stats.clipping_low_count > 0) {
        Serial.println("WARNING: Signal clipping/saturation detected! Check front-end op-amp gain and DC offset.");
    }
    Serial.println("--------------------------------------\n");
}
