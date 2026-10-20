import os
import sys
import yaml
import json
import torch
import numpy as np

# Ensure software/ is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.data.dataset import make_dataloaders
from src.models.multiscale_cnn import MultiScale1DCNN
from src.models.quantized_cnn import QuantizedMultiScale1DCNN

def export_golden_vectors(config_path: str, checkpoint_path: str, output_header: str, num_per_class: int = 2):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cpu')
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Instantiate model
    is_qat = 'quantized' in checkpoint_path or 'qat' in checkpoint_path
    if is_qat:
        model = QuantizedMultiScale1DCNN(config)
    else:
        model = MultiScale1DCNN(config)

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    # Load test dataset
    _, _, test_loader = make_dataloaders(config)
    test_dataset = test_loader.dataset

    class_names = config['data']['class_names']
    num_classes = len(class_names)

    # Collect target beats for each class
    selected_beats = {c: [] for c in range(num_classes)}
    needed = num_classes * num_per_class

    print(f"Searching test set for {num_per_class} canonical beats per class ({needed} total)...")
    for idx in range(len(test_dataset)):
        item = test_dataset[idx]
        if len(item) == 3:
            x, rr, y = item
        else:
            x, y = item
            rr = torch.ones((2,), dtype=torch.float32)

        true_cls = int(torch.argmax(y).item()) if y.dim() > 0 and len(y) > 1 else int(y.item())
        if len(selected_beats[true_cls]) < num_per_class:
            selected_beats[true_cls].append((idx, x, rr, true_cls))
            count = sum(len(v) for v in selected_beats.values())
            if count >= needed:
                break

    golden_entries = []
    with torch.no_grad():
        for c in range(num_classes):
            for beat_idx, (ds_idx, x, rr, true_cls) in enumerate(selected_beats[c]):
                # Normalize x to INT8 [-128, 127]
                raw_floats = x.numpy()
                int8_window = np.clip(np.round(raw_floats * 32.0), -128, 127).astype(np.int8)

                # Feed the exact int8 window (converted back to float / 32.0f) to record bit-exact golden logits
                if int8_window.ndim == 1:
                    x_tensor = torch.from_numpy(int8_window.astype(np.float32) / 32.0).unsqueeze(0).unsqueeze(0)
                elif int8_window.ndim == 2:
                    x_tensor = torch.from_numpy(int8_window.astype(np.float32) / 32.0).unsqueeze(0)
                else:
                    x_tensor = torch.from_numpy(int8_window.astype(np.float32) / 32.0)
                rr_tensor = rr.unsqueeze(0) # [1, 2]
                logits = model(x_tensor, rr_tensor).squeeze(0).numpy() # [5]
                pred_cls = int(np.argmax(logits))

                golden_entries.append({
                    'beat_id': f"{class_names[c]}_{beat_idx+1}_sample_{ds_idx}",
                    'true_class': c,
                    'true_class_name': class_names[c],
                    'pre_rr': float(rr[0].item()),
                    'post_rr': float(rr[1].item()),
                    'window_int8': int8_window.flatten().tolist(),
                    'logits': [float(l) for l in logits],
                    'pred_class': pred_cls,
                    'pred_class_name': class_names[pred_cls]
                })

    os.makedirs(os.path.dirname(output_header), exist_ok=True)
    in_channels = config['data'].get('in_channels', 2)
    total_window_len = in_channels * 256

    with open(output_header, 'w') as f:
        f.write("#ifndef GOLDEN_VECTORS_H\n")
        f.write("#define GOLDEN_VECTORS_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write("/**\n")
        f.write(" * ============================================================================\n")
        f.write(" * BIT-EXACT GOLDEN TEST VECTORS FOR ECG INFERENCE ENGINE\n")
        f.write(f" * Generated from DS2 Test Set using {os.path.basename(checkpoint_path)}\n")
        f.write(" * ============================================================================\n")
        f.write(" */\n\n")
        f.write(f"#define NUM_GOLDEN_VECTORS {len(golden_entries)}\n")
        f.write(f"#define GOLDEN_VECTOR_TOTAL_SIZE {total_window_len}\n")
        f.write("#define GOLDEN_VECTOR_NUM_CLASSES 5\n\n")

        f.write("struct GoldenVector {\n")
        f.write("    const char *beat_id;\n")
        f.write("    uint8_t true_class;\n")
        f.write("    float pre_rr;\n")
        f.write("    float post_rr;\n")
        f.write("    int8_t input_window[GOLDEN_VECTOR_TOTAL_SIZE];\n")
        f.write("    float expected_logits[GOLDEN_VECTOR_NUM_CLASSES];\n")
        f.write("    uint8_t expected_class;\n")
        f.write("};\n\n")

        f.write("static const GoldenVector GOLDEN_VECTORS[NUM_GOLDEN_VECTORS] = {\n")
        for i, g in enumerate(golden_entries):
            f.write(f"    // Vector {i}: {g['beat_id']} (True: {g['true_class_name']}, Pred: {g['pred_class_name']})\n")
            f.write("    {\n")
            f.write(f"        \"{g['beat_id']}\",\n")
            f.write(f"        {g['true_class']}, // {g['true_class_name']}\n")
            f.write(f"        {g['pre_rr']:.6f}f,\n")
            f.write(f"        {g['post_rr']:.6f}f,\n")
            f.write("        {\n            ")
            
            # Write 256 int8 samples
            samples = g['window_int8']
            lines = []
            for k in range(0, len(samples), 16):
                lines.append(", ".join(f"{v:4d}" for v in samples[k:k+16]))
            f.write(",\n            ".join(lines))
            f.write("\n        },\n")

            f.write("        { ")
            f.write(", ".join(f"{l:.6f}f" for l in g['logits']))
            f.write(" },\n")
            f.write(f"        {g['pred_class']} // {g['pred_class_name']}\n")
            f.write("    }" + (",\n\n" if i < len(golden_entries) - 1 else "\n\n"))

        f.write("};\n\n")
        f.write("#endif // GOLDEN_VECTORS_H\n")

    print(f"Successfully generated {len(golden_entries)} golden vectors in: {output_header}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='software/config/config.yaml')
    parser.add_argument('--checkpoint', default='software/outputs/checkpoints/best_model.pth')
    parser.add_argument('--output', default='firmware/tests/golden_vectors/golden_vectors.h')
    args = parser.parse_args()

    export_golden_vectors(args.config, args.checkpoint, args.output)
