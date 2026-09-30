import torch
import torch.nn as nn
from brevitas.nn import QuantConv1d, QuantLinear
from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat
import logging

logger = logging.getLogger("ecg_edge.models.quantized_cnn")


class QuantizedConvBranch(nn.Module):
    """
    Quantized version of ConvBranch: double QuantConv1d + BN + ReLU + dual pool.

    Mirrors MultiScale1DCNN.ConvBranch exactly for weight transfer compatibility.
    SE attention is NOT quantized (too small to matter and Brevitas doesn't
    support Sigmoid well).

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels per conv layer.
        kernel_size: Kernel size for both conv layers.
        pool_output_size: Output size for each adaptive pool.
        weight_bits: Weight quantization bit width.
        act_bits: Activation quantization bit width.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_output_size: int,
        weight_bits: int = 8,
        act_bits: int = 8,
    ):
        super().__init__()
        padding = kernel_size // 2

        # Double QuantConv1d matching FP32 architecture
        self.conv1 = QuantConv1d(
            in_channels, out_channels, kernel_size, padding=padding,
            weight_bit_width=weight_bits,
            weight_quant=Int8WeightPerTensorFloat,
            input_bit_width=act_bits,
            input_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = QuantConv1d(
            out_channels, out_channels, kernel_size, padding=padding,
            weight_bit_width=weight_bits,
            weight_quant=Int8WeightPerTensorFloat,
            input_bit_width=act_bits,
            input_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # SE attention (kept in FP32 — 6 params, not worth quantizing)
        mid = max(1, out_channels // 4)
        self.se_squeeze = nn.AdaptiveAvgPool1d(1)
        self.se_excite = nn.Sequential(
            nn.Linear(out_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, out_channels, bias=False),
            nn.Sigmoid(),
        )

        # Dual pooling
        self.max_pool = nn.AdaptiveMaxPool1d(pool_output_size)
        self.avg_pool = nn.AdaptiveAvgPool1d(pool_output_size)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        # SE attention (FP32)
        b, c, _ = x.shape
        scale = self.se_squeeze(x).view(b, c)
        scale = self.se_excite(scale).view(b, c, 1)
        x = x * scale

        max_out = self.max_pool(x)
        avg_out = self.avg_pool(x)
        return torch.cat([max_out, avg_out], dim=2)


class QuantizedMultiScale1DCNN(nn.Module):
    """
    INT8 quantized version of MultiScale1DCNN using Brevitas QAT.

    Architecture mirrors the enhanced MultiScale1DCNN exactly:
    - Three parallel branches with double QuantConv1d + SE + dual pool
    - Concat → Flatten → QuantLinear → ReLU → Dropout → QuantLinear
    - Output: raw logits shape (batch, num_classes) — NO sigmoid

    All quantization parameters from config:
        config['qat']['weight_bits'] = 8
        config['qat']['activation_bits'] = 8

    Args:
        config (dict): Full parsed config.yaml dict.
    """
    def __init__(self, config):
        super(QuantizedMultiScale1DCNN, self).__init__()

        in_channels = config['data'].get('in_channels', 1)
        num_classes = config['model']['num_classes']
        branch_out_channels = config['model']['branch_out_channels']  # 32
        kernel_sizes = config['model']['kernel_sizes']  # [3, 5, 7]
        pool_output_size = config['model']['pool_output_size']  # 64
        fc_hidden_size = config['model']['fc_hidden_size']  # 128
        dropout_rate = config['model']['dropout_rate']  # 0.3

        weight_bits = config['qat']['weight_bits']
        act_bits = config['qat']['activation_bits']

        # Parallel branches (matching FP32 architecture)
        self.branches = nn.ModuleList([
            QuantizedConvBranch(
                in_channels=in_channels,
                out_channels=branch_out_channels,
                kernel_size=k,
                pool_output_size=pool_output_size,
                weight_bits=weight_bits,
                act_bits=act_bits,
            )
            for k in kernel_sizes
        ])

        # Each branch outputs (batch, C, 2*pool_size) due to dual pooling
        flat_size = branch_out_channels * len(kernel_sizes) * (2 * pool_output_size)

        # Quantized classifier head
        self.fc1 = QuantLinear(
            flat_size, fc_hidden_size,
            weight_bit_width=weight_bits,
            weight_quant=Int8WeightPerTensorFloat,
            input_bit_width=act_bits,
            input_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)

        self.fc2 = QuantLinear(
            fc_hidden_size, num_classes,
            weight_bit_width=weight_bits,
            weight_quant=Int8WeightPerTensorFloat,
            input_bit_width=act_bits,
            input_quant=Int8ActPerTensorFloat,
            return_quant_tensor=False,  # Return plain tensor
        )

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"QuantizedMultiScale1DCNN initialized with {total_params} parameters.")

    def forward(self, x):
        # x shape: [batch, in_channels, seq_len]
        branch_outs = [branch(x) for branch in self.branches]

        # Concatenate along channel dimension
        x = torch.cat(branch_outs, dim=1)

        # Flatten
        x = torch.flatten(x, 1)

        # Classifier head
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
