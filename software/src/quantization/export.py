import torch
import torch.nn as nn
import numpy as np
import os
import json
from brevitas.nn import QuantConv1d, QuantLinear

def export_weights_to_mem(quantized_model: torch.nn.Module,
                           output_dir: str) -> dict:
    """
    Extract INT8 weights from quantized model and write .mem files
    for hardware handoff to Sanjana.

    For each QuantConv1d and QuantLinear layer:
        1. Extract integer weights using layer.weight.int_repr()
        2. Convert to numpy int8 array
        3. Flatten in row-major (C) order
        4. Write one value per line as 2-digit uppercase hex (two's complement)
        5. Filename: layer_{layer_name}_weights.mem

    Also writes weights_manifest.json containing for each layer:
        - shape (original tensor shape as list)
        - num_values (total number of int8 values)
        - scale_factor (float, from layer.weight.quant_scale().item())
        - mem_file (filename string)

    Args:
        quantized_model: Trained QuantizedMultiScale1DCNN instance
        output_dir: Directory to write .mem files and manifest

    Returns:
        dict: The manifest dict (same as written to JSON)

    Raises:
        ValueError: If quantized_model has no quantized layers found
        OSError: If output_dir cannot be created
    """
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            raise OSError(f"Could not create output directory {output_dir}: {e}")

    manifest = {}
    quant_layers_found = 0
    
    # Ensure model is on CPU for weight extraction
    quantized_model.cpu()
    
    for name, module in quantized_model.named_modules():
        if isinstance(module, (QuantConv1d, QuantLinear)):
            quant_layers_found += 1
            
            # 1. Extract integer weights
            # quant_weight() returns a QuantTensor
            qw = module.quant_weight()
            # Calculate integer representation: round(value / scale)
            int_weights_tensor = torch.round(qw.value / qw.scale).detach()
            scale = qw.scale.item()
            
            # 2. Convert to numpy int8 array
            int_weights = int_weights_tensor.numpy().astype(np.int8)
            
            # 3. Flatten in row-major (C) order
            flat_weights = int_weights.flatten(order='C')
            
            # 4. Convert to two's complement hex
            # We use a uint8 view to get the unsigned byte value for hex printing
            uint8_weights = flat_weights.view(np.uint8)
            
            mem_filename = f"layer_{name.replace('.', '_')}_weights.mem"
            mem_path = os.path.join(output_dir, mem_filename)
            
            # 5. Write to .mem file (uppercase hex, zero-padded, no blank lines)
            with open(mem_path, 'w') as f:
                for val in uint8_weights:
                    f.write(f"{val:02X}\n")
            
            manifest[name] = {
                "shape": list(int_weights.shape),
                "num_values": int(flat_weights.size),
                "scale_factor": float(scale),
                "mem_file": mem_filename
            }

    if quant_layers_found == 0:
        raise ValueError("No quantized layers (QuantConv1d, QuantLinear) found in model.")
        
    manifest_path = os.path.join(output_dir, "weights_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
        
    return manifest


def export_weights_to_c_header(quantized_model: torch.nn.Module,
                                output_dir: str) -> str:
    """
    Extract INT8 weights from quantized model and write a single C++ header file (weights.h)
    for embedded hardware targets (Arduino Giga R1 / C++ inference).

    For each QuantConv1d and QuantLinear layer:
        1. Extract integer weights using quant_weight()
        2. Convert to numpy int8 array & flatten in row-major (C) order
        3. Format as C array: const int8_t layer_{layer_name}_weights[{num_values}]
        4. Include shape and scale_factor comment above each array
        5. Enclose file in include guards (#ifndef WEIGHTS_H / #define WEIGHTS_H / #endif)

    Args:
        quantized_model: Trained QuantizedMultiScale1DCNN instance
        output_dir: Directory to write weights.h

    Returns:
        str: Absolute path to the generated weights.h file

    Raises:
        ValueError: If quantized_model has no quantized layers found
        OSError: If output_dir cannot be created
    """
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            raise OSError(f"Could not create output directory {output_dir}: {e}")

    quantized_model.cpu()
    header_path = os.path.join(output_dir, "weights.h")

    with open(header_path, 'w') as f:
        f.write("#ifndef WEIGHTS_H\n")
        f.write("#define WEIGHTS_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write("/**\n")
        f.write(" * ============================================================================\n")
        f.write(" * AUTOMATICALLY GENERATED MODEL ARRAYS FOR C++ / ARDUINO GIGA R1\n")
        f.write(" * Exported from Brevitas QuantizedMultiScale1DCNN model\n")
        f.write(" * Contains INT8 weights, layer scales, biases, BatchNorm, and SE attention parameters.\n")
        f.write(" * ============================================================================\n")
        f.write(" */\n\n")

        # 1. Export Quantized Conv and Linear layers
        for name, module in quantized_model.named_modules():
            clean_name = name.replace('.', '_')
            if isinstance(module, (QuantConv1d, QuantLinear)):
                qw = module.quant_weight()
                int_weights_tensor = torch.round(qw.value / qw.scale).detach()
                scale = qw.scale.item()

                int_weights = int_weights_tensor.numpy().astype(np.int8)
                flat_weights = int_weights.flatten(order='C')

                f.write(f"// --- Layer: {name} ({'Conv1D' if isinstance(module, QuantConv1d) else 'Linear'}) ---\n")
                f.write(f"// Shape: {list(int_weights.shape)}, Values: {flat_weights.size}\n")
                f.write(f"static const float layer_{clean_name}_scale = {scale:.16e}f;\n")
                f.write(f"static const int8_t layer_{clean_name}_weights[{flat_weights.size}] = {{\n    ")
                
                lines = []
                for i in range(0, len(flat_weights), 12):
                    chunk = ", ".join(f"{v:4d}" for v in flat_weights[i:i+12])
                    lines.append(chunk)
                f.write(",\n    ".join(lines))
                f.write("\n};\n\n")

                # Export bias if present
                if module.bias is not None:
                    bias_vals = module.bias.detach().numpy().flatten(order='C')
                    f.write(f"static const float layer_{clean_name}_bias[{bias_vals.size}] = {{\n    ")
                    b_lines = []
                    for i in range(0, len(bias_vals), 8):
                        chunk = ", ".join(f"{v:.8e}f" for v in bias_vals[i:i+8])
                        b_lines.append(chunk)
                    f.write(",\n    ".join(b_lines))
                    f.write("\n};\n\n")

            elif isinstance(module, nn.BatchNorm1d):
                f.write(f"// --- BatchNorm Layer: {name} ---\n")
                gamma = module.weight.detach().numpy().flatten()
                beta = module.bias.detach().numpy().flatten()
                mean = module.running_mean.detach().numpy().flatten()
                var = module.running_var.detach().numpy().flatten()

                f.write(f"static const float layer_{clean_name}_gamma[{gamma.size}] = {{\n    ")
                f.write(", ".join(f"{v:.8e}f" for v in gamma))
                f.write("\n};\n")

                f.write(f"static const float layer_{clean_name}_beta[{beta.size}] = {{\n    ")
                f.write(", ".join(f"{v:.8e}f" for v in beta))
                f.write("\n};\n")

                f.write(f"static const float layer_{clean_name}_mean[{mean.size}] = {{\n    ")
                f.write(", ".join(f"{v:.8e}f" for v in mean))
                f.write("\n};\n")

                f.write(f"static const float layer_{clean_name}_var[{var.size}] = {{\n    ")
                f.write(", ".join(f"{v:.8e}f" for v in var))
                f.write("\n};\n\n")

            elif isinstance(module, nn.Linear) and ('excite' in name or 'se' in name):
                f.write(f"// --- SE Linear Layer: {name} ---\n")
                se_w = module.weight.detach().numpy().flatten(order='C')
                f.write(f"static const float layer_{clean_name}_weights[{se_w.size}] = {{\n    ")
                se_lines = []
                for i in range(0, len(se_w), 8):
                    chunk = ", ".join(f"{v:.8e}f" for v in se_w[i:i+8])
                    se_lines.append(chunk)
                f.write(",\n    ".join(se_lines))
                f.write("\n};\n\n")

        f.write("#endif // WEIGHTS_H\n")

    return header_path

