import torch
import torch.optim as optim
import logging
import json
import os
from tqdm import tqdm
from src.training.loss import MorphologyWeightedBCELoss
from src.training.metrics import compute_clinical_metrics
from src.models.quantized_cnn import QuantizedMultiScale1DCNN

logger = logging.getLogger("ecg_edge.quantization.sp_qat")

class SPQATPipeline:
    """
    Sensitivity-Preserving Quantization-Aware Training pipeline.

    Steps:
        1. load_fp32_weights: Load pretrained FP32 weights into quantized model
        2. calibrate: Run calibration batches to initialize scale factors
        3. train_qat: Fine-tune with MorphologyWeightedBCELoss for config.qat.epochs
        4. evaluate_and_validate: Compare INT8 vs FP32 sensitivity, fail if drop > threshold
        5. save_checkpoint: Save final QAT model

    Args:
        config (dict): Full parsed config.yaml dict.
        fp32_checkpoint_path (str): Path to best_model.pth from Phase 1.
    """

    def __init__(self, config, fp32_checkpoint_path):
        self.config = config
        self.fp32_checkpoint_path = fp32_checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load baseline metrics for validation later
        metrics_path = "software/outputs/plots/test_metrics.json"
        if not os.path.exists(metrics_path):
             logger.warning(f"Baseline metrics not found at {metrics_path}. Validation drop check might use IMMUTABLE values.")
             self.fp32_baseline = None
        else:
            with open(metrics_path, 'r') as f:
                self.fp32_baseline = json.load(f)

    def load_fp32_weights(self, fp32_model, quant_model) -> None:
        """
        Transfer FP32 weights into quantized model using strict=False.
        Logs which keys were matched and which were skipped.
        """
        logger.info(f"Loading FP32 weights from {self.fp32_checkpoint_path}")
        checkpoint = torch.load(self.fp32_checkpoint_path, map_location='cpu')
        fp32_state_dict = checkpoint['model_state_dict']
        quant_state_dict = quant_model.state_dict()
        
        matched_keys = []
        skipped_keys = []
        
        # Geometrical architecture now strictly identical. Any mismatched layer is a critical error.
        quant_model.load_state_dict(fp32_state_dict, strict=True)
        
        for k in fp32_state_dict.keys():
            if k in quant_state_dict:
                matched_keys.append(k)
            else:
                skipped_keys.append(k)
                
        logger.info(f"Weight Transfer: Matched {len(matched_keys)} keys. Skipped {len(skipped_keys)} keys.")
        if len(skipped_keys) > 0:
            logger.error("Some weights were skipped despite strict=True being expected to fail. Review keys.")

    def calibrate(self, quant_model, train_loader) -> None:
        """
        Run config.qat.calibration_batches batches through model with
        torch.no_grad() to initialize Brevitas scale factors.
        """
        logger.info("Starting Brevitas calibration...")
        quant_model.eval()
        num_batches = self.config['qat']['calibration_batches']
        
        with torch.no_grad():
            for i, (images, _) in enumerate(train_loader):
                if i >= num_batches:
                    break
                quant_model(images.to(self.device))
                
        logger.info(f"Calibration complete: {min(i+1, num_batches)} batches processed.")

    def train_qat(self, quant_model, train_loader, val_loader) -> dict:
        """
        QAT fine-tuning loop for config.qat.epochs epochs.
        Saves best QAT checkpoint to: software/outputs/checkpoints/best_qat_model.pth
        """
        logger.info(f"Starting QAT fine-tuning for {self.config['qat']['epochs']} epochs...")
        quant_model.to(self.device)
        
        criterion = MorphologyWeightedBCELoss(
            class_weights=torch.tensor(self.config['model']['class_weights']).to(self.device)
        )
        optimizer = optim.Adam(quant_model.parameters(), lr=self.config['qat']['lr'])
        
        best_f1 = 0.0
        best_epoch = 0
        
        for epoch in range(self.config['qat']['epochs']):
            quant_model.train()
            train_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"QAT Epoch {epoch+1}")
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = quant_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                
            # Validation
            quant_model.eval()
            all_preds = []
            all_labels = []
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = quant_model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    all_preds.append(torch.sigmoid(outputs).cpu())
                    all_labels.append(labels.cpu())
            
            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()
            metrics = compute_clinical_metrics(all_labels, all_preds)
            f1_macro = metrics['macro_avg']['f1']
            
            logger.info(f"QAT Epoch {epoch+1}/{self.config['qat']['epochs']} | train_loss: {train_loss/len(train_loader):.4f} | val_loss: {val_loss/len(val_loader):.4f} | val_f1_macro: {f1_macro:.4f}")
            
            if f1_macro > best_f1:
                best_f1 = f1_macro
                best_epoch = epoch + 1
                os.makedirs("software/outputs/checkpoints", exist_ok=True)
                torch.save({
                    'model_state_dict': quant_model.state_dict(),
                    'val_f1': best_f1,
                    'epoch': best_epoch,
                    'config': self.config
                }, "software/outputs/checkpoints/best_qat_model.pth")
                logger.info(f"New best QAT model saved (F1: {best_f1:.4f})")
                
        return {'best_f1': best_f1, 'best_epoch': best_epoch}

    def evaluate_and_validate(self, quant_model, test_loader) -> dict:
        """
        Evaluate INT8 model on test set and compare sensitivity drop.
        """
        logger.info("Performing final validation of quantized model...")
        
        # Load best QAT model
        checkpoint = torch.load("software/outputs/checkpoints/best_qat_model.pth", map_location='cpu')
        quant_model.load_state_dict(checkpoint['model_state_dict'])
        quant_model.to(self.device)
        quant_model.eval()
        
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = quant_model(images)
                all_preds.append(torch.sigmoid(outputs).cpu())
                all_labels.append(labels.cpu())
        
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        int8_metrics = compute_clinical_metrics(all_labels, all_preds)
        
        # Baseline sensitivities (either from file or immutable defaults)
        baseline = {
            "ST_segment": 0.9993,
            "QT_interval": 0.7540,
            "P_wave": 0.9137,
            "Bundle_Branch_Block": 0.7040,
            "Normal": 0.9860
        }
        
        if self.fp32_baseline:
            for c in baseline.keys():
                if c in self.fp32_baseline:
                    baseline[c] = self.fp32_baseline[c]['sensitivity']

        threshold = self.config['qat']['max_sensitivity_drop_pct']
        
        for class_name, fp32_val in baseline.items():
            int8_val = int8_metrics[class_name]['sensitivity']
            drop = (fp32_val - int8_val) * 100.0
            
            if drop > threshold:
                error_msg = (f"QAT sensitivity drop too large for {class_name}: "
                           f"FP32={fp32_val:.4f}, INT8={int8_val:.4f}, "
                           f"drop={drop:.4f}% > {threshold}%")
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            logger.info(f"Validation Passed for {class_name}: INT8={int8_val:.4f}, Drop={drop:.4f}%")
                
        return int8_metrics

    def run(self, train_loader, val_loader, test_loader) -> QuantizedMultiScale1DCNN:
        """
        Execute full pipeline in order.
        """
        quant_model = QuantizedMultiScale1DCNN(self.config)
        
        # We don't necessarily need the FP32 model object, just the checkpoint path
        self.load_fp32_weights(None, quant_model)
        self.calibrate(quant_model, train_loader)
        self.train_qat(quant_model, train_loader, val_loader)
        self.evaluate_and_validate(quant_model, test_loader)
        
        return quant_model
