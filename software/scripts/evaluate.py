#!/usr/bin/env python3
"""
Evaluate ECG-FPGA Accelerator Model on the inter-patient test set.

Usage:
    python software/scripts/evaluate.py --config software/config/config.yaml \
        --checkpoint software/outputs/checkpoints/best_model.pth

This script:
1. Loads test data using inter-patient splitting (test records from config)
2. Loads the trained model checkpoint
3. Runs inference on the test set
4. Computes per-class clinical metrics (sensitivity, specificity, F1, precision)
5. Optionally uses per-class optimized thresholds from training
6. Saves detailed metrics to JSON
"""

import argparse
import yaml
import sys
import os
import logging
import torch
import numpy as np
import json
from tqdm import tqdm

# Ensure software directory is in path for imports
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
_software_dir = os.path.join(_project_root, 'software')
if _software_dir not in sys.path:
    sys.path.insert(0, _software_dir)

from src.data.dataset import make_dataloaders
from src.models.multiscale_cnn import MultiScale1DCNN
from src.training.metrics import compute_clinical_metrics, CLASS_NAMES

def main():
    parser = argparse.ArgumentParser(description="Evaluate ECG-FPGA Accelerator Model")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pth")
    parser.add_argument("--thresholds", help="Path to optimal_thresholds.json (optional)")
    args = parser.parse_args()

    # Logging setup for CLI
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    logger = logging.getLogger("ecg_edge.scripts.evaluate")

    try:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # ── Data: Inter-patient test set ──
        _, _, test_loader = make_dataloaders(config)

        # ── Model ──
        model = MultiScale1DCNN(config).to(device)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # ── Load optimal thresholds if available ──
        class_names = config['data']['class_names']
        per_class_thresholds = None

        # Check for thresholds in checkpoint
        if 'optimal_thresholds' in checkpoint:
            per_class_thresholds = checkpoint['optimal_thresholds']
            logger.info(f"Using optimal thresholds from checkpoint: {per_class_thresholds}")
        # Check for thresholds file
        elif args.thresholds and os.path.exists(args.thresholds):
            with open(args.thresholds, 'r') as f:
                per_class_thresholds = json.load(f)
            logger.info(f"Using optimal thresholds from file: {per_class_thresholds}")
        # Check default location
        else:
            default_path = os.path.join(os.getcwd(), "software/outputs/checkpoints/optimal_thresholds.json")
            if os.path.exists(default_path):
                with open(default_path, 'r') as f:
                    per_class_thresholds = json.load(f)
                logger.info(f"Using optimal thresholds from {default_path}: {per_class_thresholds}")

        # ── Inference ──
        all_logits = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for signals, labels in tqdm(test_loader, desc="Evaluating"):
                signals = signals.to(device)
                labels = labels.to(device)
                logits = model(signals)
                probs_batch = torch.sigmoid(logits)
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_probs.append(probs_batch.cpu().numpy())

        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        probs = np.concatenate(all_probs, axis=0)

        # ── Compute metrics ──
        # With default threshold
        default_threshold = config['inference']['threshold']
        metrics_default = compute_clinical_metrics(
            all_labels, probs, threshold=default_threshold, class_names=class_names
        )

        # With per-class optimized thresholds (if available)
        metrics_optimized = None
        if per_class_thresholds:
            # Apply per-class thresholds
            y_pred_opt = np.zeros_like(probs)
            for i, name in enumerate(class_names):
                t = per_class_thresholds.get(name, default_threshold)
                y_pred_opt[:, i] = (probs[:, i] >= t).astype(float)

            # Compute metrics from optimized predictions
            opt_metrics = {}
            sensitivities = []
            specificities = []
            f1s = []
            for i, name in enumerate(class_names):
                tp = int(np.sum((all_labels[:, i] == 1) & (y_pred_opt[:, i] == 1)))
                tn = int(np.sum((all_labels[:, i] == 0) & (y_pred_opt[:, i] == 0)))
                fp = int(np.sum((all_labels[:, i] == 0) & (y_pred_opt[:, i] == 1)))
                fn = int(np.sum((all_labels[:, i] == 1) & (y_pred_opt[:, i] == 0)))
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
                opt_metrics[name] = {
                    'sensitivity': float(sens), 'specificity': float(spec),
                    'precision': float(prec), 'f1': float(f1),
                    'support': int(tp + fn), 'threshold': per_class_thresholds.get(name, default_threshold),
                }
                sensitivities.append(sens)
                specificities.append(spec)
                f1s.append(f1)
            opt_metrics['macro_avg'] = {
                'sensitivity': float(np.mean(sensitivities)),
                'specificity': float(np.mean(specificities)),
                'f1': float(np.mean(f1s)),
            }
            metrics_optimized = opt_metrics

        # ── Print results ──
        print(f"\n{'='*75}")
        print(f"  TEST SET EVALUATION (threshold={default_threshold})")
        print(f"{'='*75}")
        print(f"{'Class':<25} {'Sensitivity':<12} {'Specificity':<12} {'F1':<12} {'Support':<10}")
        print("-" * 75)
        for name in class_names:
            m = metrics_default[name]
            support = m.get('support', 'N/A')
            print(f"{name:<25} {m['sensitivity']:<12.4f} {m['specificity']:<12.4f} {m['f1']:<12.4f} {support:<10}")

        m_avg = metrics_default['macro_avg']
        print("-" * 75)
        print(f"{'Macro Average':<25} {m_avg['sensitivity']:<12.4f} {m_avg['specificity']:<12.4f} {m_avg['f1']:<12.4f}\n")

        if metrics_optimized:
            print(f"{'='*75}")
            print(f"  TEST SET EVALUATION (per-class optimized thresholds)")
            print(f"{'='*75}")
            print(f"{'Class':<25} {'Threshold':<10} {'Sensitivity':<12} {'Specificity':<12} {'F1':<12}")
            print("-" * 75)
            for name in class_names:
                m = metrics_optimized[name]
                t = m.get('threshold', default_threshold)
                print(f"{name:<25} {t:<10.3f} {m['sensitivity']:<12.4f} {m['specificity']:<12.4f} {m['f1']:<12.4f}")

            m_avg_opt = metrics_optimized['macro_avg']
            print("-" * 75)
            print(f"{'Macro Average':<25} {'---':<10} {m_avg_opt['sensitivity']:<12.4f} {m_avg_opt['specificity']:<12.4f} {m_avg_opt['f1']:<12.4f}\n")

        # ── Save to JSON ──
        output_dir = os.path.join(os.getcwd(), "software/outputs/plots")
        os.makedirs(output_dir, exist_ok=True)

        results = {
            'default_threshold': metrics_default,
            'evaluation_config': {
                'checkpoint': args.checkpoint,
                'default_threshold': default_threshold,
                'inter_patient_split': True,
                'test_records': config['data']['test_records'],
                'class_names': class_names,
            }
        }
        if metrics_optimized:
            results['optimized_thresholds'] = metrics_optimized

        output_path = os.path.join(output_dir, "test_metrics.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Metrics saved to {output_path}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
