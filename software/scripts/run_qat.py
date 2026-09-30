import argparse
import yaml
import torch
import logging
import os
import sys

# Ensure software/ is in path for imports if running as script
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.data.dataset import make_dataloaders
from src.models.quantized_cnn import QuantizedMultiScale1DCNN
from src.quantization.sp_qat import SPQATPipeline
from src.quantization.export import export_weights_to_mem, export_weights_to_c_header

def setup_logging():
    os.makedirs("software/outputs/logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("software/outputs/logs/qat.log")
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="Run Sensitivity-Preserving QAT Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--fp32-checkpoint", type=str, required=True, help="Path to FP32 best_model.pth")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("ecg_edge.scripts.run_qat")

    try:
        # 1. Initialization
        logger.info("Initializing Phase 2: QAT Pipeline")
        if not os.path.exists(args.fp32_checkpoint):
            raise FileNotFoundError(f"FP32 checkpoint not found: {args.fp32_checkpoint}")
        
        # 2. Load Config
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # 3. Prepare Data
        logger.info("Preparing datasets and dataloaders...")
        train_loader, val_loader, test_loader = make_dataloaders(config)
        
        # 4. Instantiate Pipeline & Run
        pipeline = SPQATPipeline(config, args.fp32_checkpoint)
        
        # This executes: load_fp32_weights -> calibrate -> train_qat -> evaluate_and_validate
        quant_model = pipeline.run(train_loader, val_loader, test_loader)
        
        # 5. Export to .mem and C++ header weights.h
        logger.info("Exporting verified quantized weights to .mem files and C++ weights.h header...")
        export_dir = config['export']['output_dir']
        export_weights_to_mem(quant_model, export_dir)
        header_path = export_weights_to_c_header(quant_model, export_dir)
        logger.info(f".mem files, weights_manifest.json, and {os.path.basename(header_path)} written to {export_dir}")
        
        logger.info("Phase 2 QAT Pipeline completed successfully.")
        
    except Exception as e:
        logger.error(f"QAT Pipeline failed: {str(e)}")
        # We re-raise to ensure CLI exit code is non-zero
        raise e

if __name__ == "__main__":
    main()
