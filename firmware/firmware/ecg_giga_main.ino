/**
 * ============================================================================
 * ECG ARRHYTHMIA ACCELERATOR — ARDUINO GIGA R1 FIRMWARE
 * Main Arduino Sketch: 16-Bit ADC Acquisition & Diagnostic Serial Streaming
 * Target Hardware: Arduino Giga R1 WiFi (STM32H747XI Dual Core)
 * ============================================================================
 */

#include "adc_acquisition.h"
#include "signal_diagnostics.h"
#include "inference.h"

// Periodic Summary Interval (every 10 seconds = 3600 samples @ 360 Hz)
static const uint32_t SUMMARY_INTERVAL_MS = 10000;
static uint32_t last_summary_ms = 0;

void setup() {
    // 1. Initialize High-Speed USB Serial Interface @ 115200 Baud
    Serial.begin(115200);

    // Wait up to 3 seconds for Serial Monitor connection (non-blocking for standalone boot)
    uint32_t serial_timeout = millis() + 3000;
    while (!Serial && millis() < serial_timeout) {
        delay(10);
    }

    Serial.println("\n==================================================");
    Serial.println("  ECG-FPGA ACCELERATOR -> ARDUINO GIGA R1 FIRMWARE");
    Serial.println("  16-Bit ADC Sampling @ 360 Hz (PhysioNet MIT-BIH)");
    Serial.println("==================================================\n");

    // 2. Initialize 16-bit ADC Hardware Peripheral at 360 Hz
    if (!adc_acquisition_init()) {
        Serial.println("ERROR: ADC Hardware Initialization Failed!");
    } else {
        Serial.println("SUCCESS: 16-Bit ADC Initialized at 360 Hz.");
    }

    // 3. Initialize Signal Quality Inspection Utilities
    signal_diagnostics_init();

    // 4. Initialize INT8 CNN Inference Engine Stub
    inference_init();

    last_summary_ms = millis();
}

void loop() {
    AdcSample sample;

    // Read non-blocking sample at 360 Hz (~2.777 ms interval)
    if (adc_acquisition_read(sample)) {
        // Stream raw ADC value over Serial in CSV format: timestamp_ms,raw_value
        signal_diagnostics_log_csv(sample);
    }

    // Periodically print diagnostic summary report over Serial (every 10 seconds)
    if (millis() - last_summary_ms >= SUMMARY_INTERVAL_MS) {
        signal_diagnostics_print_summary();
        last_summary_ms = millis();
    }
}
