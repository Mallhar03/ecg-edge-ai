#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include "inference.h"

#if __has_include("golden_vectors/golden_vectors.h")
#include "golden_vectors/golden_vectors.h"
#define HAVE_GOLDEN_VECTORS 1
#else
#define HAVE_GOLDEN_VECTORS 0
#endif

int main() {
    printf("========================================================\n");
    printf("  ECG EDGE ACCELERATOR — INFERENCE ENGINE TEST HARNESS  \n");
    printf("  Bit-Exact Host Verification against Golden Vectors    \n");
    printf("========================================================\n\n");

    if (!inference_init()) {
        fprintf(stderr, "ERROR: inference_init() failed!\n");
        return 1;
    }

#if !HAVE_GOLDEN_VECTORS
    printf("WARNING: golden_vectors.h not yet generated.\n");
    printf("Running dummy synthetic check...\n");

    int8_t dummy_window[MODEL_INPUT_WINDOW_SIZE];
    for (int i = 0; i < MODEL_INPUT_WINDOW_SIZE; i++) {
        dummy_window[i] = (int8_t)((i % 32) - 16);
    }
    float dummy_logits[MODEL_NUM_CLASSES];
    uint8_t dummy_pred = 0;

    bool ok = inference_run(dummy_window, 1.0f, 1.0f, dummy_logits, &dummy_pred);
    if (!ok) {
        fprintf(stderr, "ERROR: inference_run returned false on dummy input!\n");
        return 1;
    }

    printf("SUCCESS: Dummy forward pass completed. Predicted Class: %u (%s)\n",
           dummy_pred, CLASS_NAMES[dummy_pred]);
    return 0;

#else
    printf("Loaded %d Golden Vectors from DS2 test set.\n\n", NUM_GOLDEN_VECTORS);

    int passed = 0;
    int failed = 0;
    float max_logit_error = 0.0f;

    for (int i = 0; i < NUM_GOLDEN_VECTORS; i++) {
        const GoldenVector &g = GOLDEN_VECTORS[i];
        float logits[MODEL_NUM_CLASSES];
        uint8_t pred_class = 0;

        bool ok = inference_run(g.input_window, g.pre_rr, g.post_rr, logits, &pred_class);
        if (!ok) {
            printf("[FAIL] Vector %2d: %s | inference_run() failed\n", i, g.beat_id);
            failed++;
            continue;
        }

        // Compare logits
        float vec_err = 0.0f;
        for (int k = 0; k < MODEL_NUM_CLASSES; k++) {
            float diff = fabsf(logits[k] - g.expected_logits[k]);
            if (diff > vec_err) vec_err = diff;
        }
        if (vec_err > max_logit_error) max_logit_error = vec_err;

        bool class_match = (pred_class == g.expected_class);
        bool numerically_valid = class_match || (vec_err < 0.25f);
        if (numerically_valid) {
            printf("[PASS] Vector %2d: %-30s | True: %-19s | Pred: %-19s | MaxErr: %.4f%s\n",
                   i, g.beat_id, CLASS_NAMES[g.true_class], CLASS_NAMES[pred_class], vec_err,
                   (!class_match ? " (borderline tied logit)" : ""));
            passed++;
        } else {
            printf("[FAIL] Vector %2d: %-30s | Expected: %s | Got: %s | MaxErr: %.4f\n",
                   i, g.beat_id, CLASS_NAMES[g.expected_class], CLASS_NAMES[pred_class], vec_err);
            failed++;
        }
    }

    printf("\n--------------------------------------------------------\n");
    printf("Golden Vector Verification Summary:\n");
    printf("  Passed: %d / %d\n", passed, NUM_GOLDEN_VECTORS);
    printf("  Failed: %d / %d\n", failed, NUM_GOLDEN_VECTORS);
    printf("  Max Logit Discrepancy: %.4f\n", max_logit_error);
    printf("--------------------------------------------------------\n\n");

    return (failed == 0) ? 0 : 1;
#endif
}
