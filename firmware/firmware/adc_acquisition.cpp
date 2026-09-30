#include "adc_acquisition.h"
#include <Arduino_AdvancedAnalog.h>

/**
 * ============================================================================
 * DIAGNOSTIC-FIRST ADC ACQUISITION IMPLEMENTATION
 * Target Hardware: Arduino Giga R1 WiFi (STM32H747XI)
 * ============================================================================
 *
 * HARDWARE DEBUGGING NOTES:
 * ----------------------------------------------------------------------------
 * 1. ADC PERIPHERAL — LIBRARY-MANAGED INTERNALS:
 *    - STM32H747XI features 3 independent 16-bit SAR ADCs (ADC1, ADC2, ADC3).
 *    - Clock prescaler, DMA channel selection, and peripheral clock routing are
 *      configured internally by the Arduino_AdvancedAnalog library. They are
 *      NOT user-configurable through this API. No register-level prescaler
 *      calls are made by this code.
 *    - NOTE (Aug 3 EOD Report signal distortion): If high source impedance from
 *      the ECG front-end op-amp is causing S/H capacitor undercharge, the only
 *      lever exposed by the API is `sample_time` (the last argument to begin()).
 *      AN_ADC_SAMPLETIME_387_5 is used here as a starting point; also available
 *      are AN_ADC_SAMPLETIME_64_5 and AN_ADC_SAMPLETIME_810_5. Extend the
 *      sample time first if distortion persists. This is the ONLY supported
 *      software tuning point for the S/H settling window.
 *
 * 2. SAMPLING TIME — API-EXPOSED ONLY:
 *    - Sampling time (S/H capacitor settling duration) is passed as the
 *      `sample_time` argument to adc.begin(). This code uses
 *      AN_ADC_SAMPLETIME_387_5 (387.5 ADC clock cycles) as a conservative
 *      starting point for ECG front-end source impedances.
 *    - Exact clock-cycle-to-nanosecond mapping depends on the ADC kernel
 *      clock selected by the library internally, which is not exposed by
 *      this API. Do not guess at it from register fields — measure the actual
 *      settling waveform on a scope instead.
 *
 * 3. VOLTAGE REFERENCE (VREF) — HARDWARE ONLY, NOT SOFTWARE-CONFIGURABLE:
 *    - VREF+ on Giga R1 is tied to 3.3 V (VDDA) at the board level.
 *    - This code does not configure it. A 100 nF decoupling cap on VREF+/VDDA
 *      is a layout requirement, not a firmware concern.
 *
 * 4. 360 HZ TIMING & BUFFER SIZING:
 *    - AdvancedAnalog drives sampling via a hardware timer + DMA internally.
 *      The 360 Hz rate is passed to begin() and enforced by the library.
 *    - n_samples = 32: each DMABuffer holds 32 raw samples. At 360 Hz a new
 *      buffer becomes available approximately every 89 ms (~32/360 s).
 *    - n_buffers = 4: queue depth of 4. Memory footprint = 4×32×2 = 256 bytes.
 *      Provides enough headroom to survive one loop() iteration delay without
 *      dropping a buffer.
 *
 * 5. BUFFER LIFETIME & release():
 *    - SampleBuffer is a typedef for DMABuffer<Sample>& — a REFERENCE, not a
 *      value type. It cannot be stored in a static variable.
 *    - We hold the live buffer as DMABuffer<Sample> (the actual value type),
 *      which is default-constructible (pool=nullptr, ptr=nullptr) and can be
 *      stored statically.
 *    - buf.release() MUST be called once a buffer is fully drained to return
 *      it to the library's DMA write queue. Failing to call release() will
 *      stall the queue: adc.available() will stop returning true after
 *      n_buffers iterations.
 * ============================================================================
 */

// AdvancedAnalog object bound to input pin A0
static AdvancedADC adc(ADC_INPUT_PIN);

// --- Internal buffer-draining state ---
// SampleBuffer (the typedef) is DMABuffer<Sample>& — a reference — and cannot
// be stored in a static. We store the concrete DMABuffer<Sample> value instead.
// DMABuffer<Sample> is default-constructible (ptr == nullptr, pool == nullptr).
static DMABuffer<Sample> current_buf;   // holds the in-flight DMA buffer
static bool   buf_valid     = false;    // true while current_buf is live
static size_t buf_index     = 0;        // next sample index to drain
// Must match n_samples passed to begin() — see note 4 above.
static const size_t N_SAMPLES_PER_BUF  = 32;

static uint32_t last_sample_us = 0;
static bool     is_initialized = false;

// Whether the AdvancedADC hardware path started successfully.
static bool     hw_adc_ok      = false;

float adc_raw_to_mv(uint16_t raw_val) {
    // Linear scaling: (raw_val / 65535.0) * 3300.0 mV
    return ((float)raw_val / (float)ADC_MAX_RAW_VALUE) * ADC_VREF_MV;
}

bool adc_acquisition_init() {
    // 1. Set resolution — also used by the software analogRead() fallback path.
    analogReadResolution(ADC_RESOLUTION_BITS);

    // 2. Start AdvancedADC DMA pipeline.
    //    Signature: begin(resolution, sample_rate_hz, n_samples, n_buffers,
    //                     start=true, sample_time=AN_ADC_SAMPLETIME_8_5)
    //
    //    n_samples = 32, n_buffers = 4  — see note 4 in file header above.
    //    AN_ADC_SAMPLETIME_387_5: extended settling for ECG front-end impedance.
    //    Change to AN_ADC_SAMPLETIME_810_5 if signal distortion persists.
    hw_adc_ok = adc.begin(AN_RESOLUTION_16, SAMPLE_RATE_HZ,
                          N_SAMPLES_PER_BUF, 4,
                          true, AN_ADC_SAMPLETIME_387_5);

    if (!hw_adc_ok) {
        // AdvancedADC DMA init failed (resource conflict, pin conflict, etc.).
        // Fall back to software-timed analogRead(). This path has higher jitter
        // and only 12-bit effective resolution on the Giga R1 core.
        // The warning is printed once here at init, and again on every fallback
        // sample so it is never silent.
        pinMode(ADC_INPUT_PIN, INPUT);
        Serial.println("WARNING: AdvancedADC begin() failed. "
                       "Falling back to software-timed analogRead() at 360 Hz. "
                       "Expect higher jitter and lower effective resolution.");
    }

    buf_valid      = false;
    buf_index      = 0;
    last_sample_us = micros();
    is_initialized = true;
    return true;  // always report init success; hw_adc_ok tracks DMA state
}

bool adc_acquisition_read(AdcSample &sample) {
    if (!is_initialized) {
        return false;
    }

    uint32_t now_us = micros();

    // --- Hardware DMA path ---
    if (hw_adc_ok) {
        // If we have an active buffer with samples remaining, drain it first.
        if (buf_valid && buf_index < N_SAMPLES_PER_BUF) {
            sample.raw_value    = (uint16_t)current_buf[buf_index];
            sample.timestamp_ms = millis();
            sample.timestamp_us = now_us;
            sample.voltage_mv   = adc_raw_to_mv(sample.raw_value);
            buf_index++;

            // If we just consumed the last sample in this buffer, release it
            // back to the library's DMA pool so it can be reused. Forgetting
            // this call will stall the queue after n_buffers iterations.
            if (buf_index >= N_SAMPLES_PER_BUF) {
                current_buf.release();
                buf_valid = false;
                buf_index = 0;
            }

            last_sample_us = now_us;
            return true;
        }

        // No active buffer (or it was just released). Try to fetch a new one.
        if (adc.available()) {
            // adc.read() returns SampleBuffer (= DMABuffer<Sample>&).
            // We copy-assign into our DMABuffer<Sample> static to take
            // ownership. The buffer stays live until we call release().
            current_buf = adc.read();
            buf_valid   = true;
            buf_index   = 0;

            sample.raw_value    = (uint16_t)current_buf[buf_index];
            sample.timestamp_ms = millis();
            sample.timestamp_us = now_us;
            sample.voltage_mv   = adc_raw_to_mv(sample.raw_value);
            buf_index++;

            if (buf_index >= N_SAMPLES_PER_BUF) {
                current_buf.release();
                buf_valid = false;
                buf_index = 0;
            }

            last_sample_us = now_us;
            return true;
        }

        // No buffer available yet this call.
        return false;
    }

    // --- Software fallback path (~360 Hz, software timer) ---
    // Warning printed on every sample so fallback usage is never silent.
    if (now_us - last_sample_us >= SAMPLE_PERIOD_US) {
        Serial.println("WARNING: Using software-timed analogRead() fallback "
                       "(higher jitter, lower effective resolution).");
        sample.raw_value    = (uint16_t)analogRead(ADC_INPUT_PIN);
        sample.timestamp_ms = millis();
        sample.timestamp_us = now_us;
        sample.voltage_mv   = adc_raw_to_mv(sample.raw_value);
        last_sample_us      = now_us;
        return true;
    }

    return false;
}
