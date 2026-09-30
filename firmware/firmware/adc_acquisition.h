#ifndef ADC_ACQUISITION_H
#define ADC_ACQUISITION_H

#include <Arduino.h>

/**
 * ============================================================================
 * ADC ACQUISITION CONFIGURATION & PARAMETERS
 * Target Hardware: Arduino Giga R1 WiFi (STM32H747XI Dual Cortex-M7/M4)
 * Library: Arduino_AdvancedAnalog
 * ============================================================================
 */

// Explicit ADC Resolution (Bits) - Configurable, Default: 16-bit
#ifndef ADC_RESOLUTION_BITS
#define ADC_RESOLUTION_BITS 16
#endif

// Maximum Raw ADC Count based on resolution (65535 for 16-bit)
#define ADC_MAX_RAW_VALUE ((1UL << ADC_RESOLUTION_BITS) - 1UL)

/**
 * CLINICAL MANDATE: SAMPLE_RATE_HZ = 360
 * The INT8 CNN classifier is trained on the PhysioNet MIT-BIH Arrhythmia Database,
 * which is sampled natively at 360 Hz (~2.777 ms sampling period).
 * 
 * CRITICAL WARNING: Modifying this sample rate will distort the temporal length
 * of ECG waveforms (P-wave, QRS complex, ST segment, T-wave), corrupting cardiac
 * feature extraction and invalidating model predictions.
 */
static const uint32_t SAMPLE_RATE_HZ = 360;
static const uint32_t SAMPLE_PERIOD_US = 1000000UL / SAMPLE_RATE_HZ; // ~2777 microseconds

// Analog Pin for ECG Front-End ADC Input
#ifndef ADC_INPUT_PIN
#define ADC_INPUT_PIN A0
#endif

// Full-scale ADC reference voltage in millivolts (3.3V = 3300 mV)
#ifndef ADC_VREF_MV
#define ADC_VREF_MV 3300.0f
#endif

/**
 * Data structure representing a single ECG ADC sample.
 */
struct AdcSample {
    uint32_t timestamp_ms; // Milliseconds since MCU boot
    uint32_t timestamp_us; // Microseconds timestamp for precise jitter analysis
    uint16_t raw_value;    // 16-bit raw ADC reading (0 - 65535)
    float    voltage_mv;   // Calibrated input voltage in millivolts
};

/**
 * Initialize the 16-bit ADC peripheral on the Arduino Giga R1 at 360 Hz.
 * @return true if initialization succeeded, false on error.
 */
bool adc_acquisition_init();

/**
 * Non-blocking check and read of the latest ADC sample.
 * @param sample Output reference struct to populate with sample data.
 * @return true if a new sample was acquired, false otherwise.
 */
bool adc_acquisition_read(AdcSample &sample);

/**
 * Convert raw 16-bit integer ADC value to millivolts.
 * @param raw_val 16-bit raw ADC code (0 - 65535).
 * @return Analog voltage in mV.
 */
float adc_raw_to_mv(uint16_t raw_val);

#endif // ADC_ACQUISITION_H
