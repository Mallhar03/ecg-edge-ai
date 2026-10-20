import os
import shutil
import tempfile
import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.trainer import Trainer
from src.training.loss import FocalCrossEntropyLoss


class DummyModel(nn.Module):
    def __init__(self, in_features=256, num_classes=5):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x, rr=None):
        if x.ndim == 3:
            x = x.squeeze(1)
        out = self.fc(x)
        if rr is not None:
            out = out + 0.0 * rr.sum(dim=-1, keepdim=True)
        return out


@pytest.fixture
def mock_config(tmp_path):
    output_dir = tmp_path / "outputs"
    return {
        'project': {
            'name': 'test_project',
            'seed': 42,
            'output_dir': str(output_dir),
            'log_dir': str(output_dir / "logs"),
        },
        'data': {
            'class_names': ['S', 'V', 'B', 'Q', 'N'],
        },
        'training': {
            'epochs': 2,
            'learning_rate': 0.01,
            'optimizer': 'adam',
            'weight_decay': 1e-4,
            'early_stopping_patience': 2,
            'checkpoint_every_n_epochs': 1,
            'lr_scheduler': 'cosine',
            'lr_scheduler_min_lr': 1e-5,
        },
        'inference': {
            'optimize_thresholds': True,
        }
    }


def test_trainer_initialization(mock_config):
    trainer = Trainer(mock_config)
    assert os.path.exists(trainer.checkpoint_dir)
    assert os.path.exists(trainer.log_dir)
    assert trainer.seed == 42


def test_trainer_optimizer_and_scheduler(mock_config):
    trainer = Trainer(mock_config)
    model = DummyModel()
    
    # Test Adam optimizer
    optimizer = trainer._get_optimizer(model)
    assert isinstance(optimizer, torch.optim.Adam)

    # Test Cosine scheduler
    scheduler = trainer._get_scheduler(optimizer, epochs=10)
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    # Test Plateau scheduler
    mock_config['training']['lr_scheduler'] = 'plateau'
    plateau_scheduler = trainer._get_scheduler(optimizer, epochs=10)
    assert isinstance(plateau_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)


def test_train_one_epoch_and_validate(mock_config):
    trainer = Trainer(mock_config)
    model = DummyModel()
    optimizer = trainer._get_optimizer(model)
    loss_fn = FocalCrossEntropyLoss([1.0, 1.0, 1.0, 1.0, 1.0])

    # 1. Test with 3-element batch (images, rr, labels)
    images = torch.randn(8, 1, 256)
    rr = torch.ones(8, 2)
    labels = torch.zeros(8, 5)
    labels[:, 4] = 1.0  # Normal class

    dataset_3 = TensorDataset(images, rr, labels)
    loader_3 = DataLoader(dataset_3, batch_size=4)

    train_res = trainer.train_one_epoch(model, loader_3, optimizer, loss_fn)
    assert 'loss' in train_res
    assert train_res['loss'] >= 0.0

    val_res = trainer.validate(model, loader_3, loss_fn)
    assert 'loss' in val_res
    assert 'f1_macro' in val_res
    assert 'all_logits' in val_res
    assert 'all_labels' in val_res
    assert val_res['all_labels'].shape == (8, 5)

    # 2. Test with 2-element batch (images, labels)
    dataset_2 = TensorDataset(images, labels)
    loader_2 = DataLoader(dataset_2, batch_size=4)
    train_res_2 = trainer.train_one_epoch(model, loader_2, optimizer, loss_fn)
    assert train_res_2['loss'] >= 0.0


def test_validate_with_1d_labels(mock_config):
    trainer = Trainer(mock_config)
    model = DummyModel()
    loss_fn = nn.CrossEntropyLoss()

    images = torch.randn(6, 1, 256)
    labels_1d = torch.tensor([0, 1, 2, 3, 4, 4], dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels_1d), batch_size=3)

    val_res = trainer.validate(model, loader, loss_fn)
    assert 'f1_macro' in val_res
    # all_labels must be converted to 2D one-hot (6, 5)
    assert val_res['all_labels'].shape == (6, 5)


def test_checkpoint_save_and_load(mock_config, tmp_path):
    trainer = Trainer(mock_config)
    model = DummyModel()
    optimizer = trainer._get_optimizer(model)

    ckpt_path = os.path.join(trainer.checkpoint_dir, "test_ckpt.pth")
    trainer.optimal_thresholds = {'S': 0.3, 'V': 0.4, 'B': 0.5, 'Q': 0.5, 'N': 0.2}
    trainer.save_checkpoint(model, optimizer, epoch=3, val_f1=0.85, path=ckpt_path)

    assert os.path.exists(ckpt_path)

    # Load into fresh model
    new_model = DummyModel()
    new_optimizer = trainer._get_optimizer(new_model)
    epoch, val_f1 = trainer.load_checkpoint(ckpt_path, new_model, new_optimizer)

    assert epoch == 3
    assert val_f1 == 0.85
    assert trainer.optimal_thresholds['S'] == 0.3


def test_fit_full_lifecycle(mock_config):
    trainer = Trainer(mock_config)
    model = DummyModel()
    loss_fn = nn.CrossEntropyLoss()

    images = torch.randn(12, 1, 256)
    rr = torch.ones(12, 2)
    labels = torch.zeros(12, 5)
    labels[:6, 1] = 1.0  # Ventricular
    labels[6:, 4] = 1.0  # Normal

    loader = DataLoader(TensorDataset(images, rr, labels), batch_size=4)

    trainer.fit(model, loader, loader, loss_fn)

    best_model_path = os.path.join(trainer.checkpoint_dir, "best_model.pth")
    assert os.path.exists(best_model_path)
    assert trainer.optimal_thresholds is not None
    assert 'optimal_thresholds' in torch.load(best_model_path, map_location='cpu', weights_only=False)
