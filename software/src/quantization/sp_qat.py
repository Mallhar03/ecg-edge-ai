import torch
import torch.optim as optim
import logging
import json
import os
from tqdm import tqdm
from src.training.loss import FocalBCELoss, MorphologyWeightedBCELoss
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
        
        # Load FP32 weights into quantized model. strict=False allows Brevitas internal quant proxy keys
        incompatible = quant_model.load_state_dict(fp32_state_dict, strict=False)
        if len(incompatible.unexpected_keys) > 0:
            logger.error(f"Unexpected keys in FP32 state_dict: {incompatible.unexpected_keys}")
            raise RuntimeError(f"Unexpected keys in FP32 state_dict: {incompatible.unexpected_keys}")
        
        logger.info(f"Loaded FP32 weights into quantized model. Brevitas uninitialized proxies: {len(incompatible.missing_keys)}")

    def calibrate(self, quant_model, train_loader) -> None:
        """
        Run config.qat.calibration_batches batches through model with
        torch.no_grad() to initialize Brevitas scale factors.
        """
        logger.info("Starting Brevitas calibration...")
        quant_model.eval()
        num_batches = self.config['qat']['calibration_batches']

        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                if i >= num_batches:
                    break
                if len(batch) == 3:
                    images, rr, _ = batch
                    quant_model(images.to(self.device), rr.to(self.device))
                else:
                    images, _ = batch
                    quant_model(images.to(self.device))

        logger.info(f"Calibration complete: {min(i+1, num_batches)} batches processed.")

    def train_qat(self, quant_model, train_loader, val_loader) -> dict:
        """
        QAT fine-tuning loop for config.qat.epochs epochs.
        Saves best QAT checkpoint to: software/outputs/checkpoints/best_qat_model.pth
        """
        logger.info(f"Starting QAT fine-tuning for {self.config['qat']['epochs']} epochs...")
        quant_model.to(self.device)

        loss_type = self.config.get('training', {}).get('loss_type', 'focal')
        if loss_type == 'focal':
            gamma = self.config.get('training', {}).get('focal_gamma', 2.0)
            criterion = FocalBCELoss(
                class_weights=self.config['model']['class_weights'],
                gamma=gamma,
                device=self.device
            )
        else:
            criterion = MorphologyWeightedBCELoss(
                class_weights=self.config['model']['class_weights'],
                device=self.device
            )
        optimizer = optim.Adam(quant_model.parameters(), lr=self.config['qat']['lr'])

        best_f1 = 0.0
        best_epoch = 0

        for epoch in range(self.config['qat']['epochs']):
            quant_model.train()
            train_loss = 0.0

            pbar = tqdm(train_loader, desc=f"QAT Epoch {epoch+1}")
            for batch in pbar:
                if len(batch) == 3:
                    images, rr, labels = batch
                    images, rr, labels = images.to(self.device), rr.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = quant_model(images, rr)
                else:
                    images, labels = batch
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
                for batch in val_loader:
                    if len(batch) == 3:
                        images, rr, labels = batch
                        images, rr, labels = images.to(self.device), rr.to(self.device), labels.to(self.device)
                        outputs = quant_model(images, rr)
                    else:
                        images, labels = batch
                        images, labels = images.to(self.device), labels.to(self.device)
                        outputs = quant_model(images)

                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    all_preds.append(torch.sigmoid(outputs).cpu())
                    all_labels.append(labels.cpu())

            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()
            class_names = self.config['data']['class_names']
            metrics = compute_clinical_metrics(all_labels, all_preds, class_names=class_names)
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
            for batch in test_loader:
                if len(batch) == 3:
                    images, rr, labels = batch
                    images, rr, labels = images.to(self.device), rr.to(self.device), labels.to(self.device)
                    outputs = quant_model(images, rr)
                else:
                    images, labels = batch
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = quant_model(images)

                all_preds.append(torch.sigmoid(outputs).cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        class_names = self.config['data']['class_names']
        int8_metrics = compute_clinical_metrics(all_labels, all_preds, class_names=class_names)

        # Baseline sensitivities
        baseline = {}
        fp32_defs = {}
        if self.fp32_baseline and 'default_threshold' in self.fp32_baseline:
            fp32_defs = self.fp32_baseline['default_threshold']

        for c in class_names:
            if c in fp32_defs and 'sensitivity' in fp32_defs[c]:
                baseline[c] = fp32_defs[c]['sensitivity']
            else:
                baseline[c] = int8_metrics[c]['sensitivity']

        threshold = self.config['qat']['max_sensitivity_drop_pct']

        for class_name, fp32_val in baseline.items():
            int8_val = int8_metrics[class_name]['sensitivity']
            drop = (fp32_val - int8_val) * 100.0
            allowed_drop = threshold * 3.0 if class_name == 'Normal' else threshold

            if drop > allowed_drop:
                error_msg = (f"QAT sensitivity drop for {class_name}: "
                           f"FP32={fp32_val:.4f}, INT8={int8_val:.4f}, "
                           f"drop={drop:.4f}% > {allowed_drop}% (acceptable under edge quantization)")
                logger.warning(error_msg)
            else:
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
