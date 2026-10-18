"""Manuscript STS-Net core, extracted from the supplied public package.

Only the paper_exact/paper execution path is included. Numerical operators,
parameter registration order, and checkpoint key prefixes are preserved.
This extraction does not supply the missing historical ablation or L-sweep scripts.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F


class LegacyTemporalStem(nn.Module):
    """主文对齐的单卷积核 CNN stem，默认 kernel=25、pool=75/15。"""

    def __init__(
        self,
        n_channels: int,
        emb_size: int = 40,
        kernel: int = 25,
        dropout: float = 0.5,
        pool_kernel: int = 75,
        pool_stride: int = 15,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, emb_size, (1, kernel)),
            nn.Conv2d(emb_size, emb_size, (n_channels, 1)),
            nn.BatchNorm2d(emb_size),
            nn.ELU(),
            nn.AvgPool2d((1, pool_kernel), (1, pool_stride)),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x.squeeze(2).transpose(1, 2).contiguous()


class PaperSparseAttention(nn.Module):
    """主文对齐的稀疏注意力模块。

    ``random_sparse`` 对应旧脚本里的随机稀疏掩码；``use_global_key`` 对应全局
    key 分支。最终 anchor 使用随机稀疏、sparsity_factor=0.1、关闭 global key。
    """

    def __init__(
        self,
        emb_size: int = 40,
        heads: int = 10,
        top_k: int = 8,
        random_sparse: bool = False,
        sparsity_factor: float = 0.5,
        use_global_key: bool = True,
    ) -> None:
        super().__init__()
        if emb_size % heads != 0:
            raise ValueError("emb_size must be divisible by heads.")
        self.emb_size = emb_size
        self.heads = heads
        self.head_dim = emb_size // heads
        self.top_k = top_k
        self.random_sparse = random_sparse
        self.sparsity_factor = sparsity_factor
        self.use_global_key = use_global_key
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.projection = nn.Linear(emb_size, emb_size)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        queries = self._split(self.queries(x))
        keys = self._split(self.keys(x))
        values = self._split(self.values(x))
        energy = torch.matmul(queries, keys.transpose(-2, -1))
        if self.use_global_key:
            global_key = keys.mean(dim=2, keepdim=True)
            energy = energy + torch.matmul(queries, global_key.transpose(-2, -1))

        if self.random_sparse:
            mask = torch.rand(
                energy.shape[-2:],
                device=energy.device,
                dtype=energy.dtype,
            ) > self.sparsity_factor
            energy = energy.masked_fill(~mask.view(1, 1, *mask.shape), torch.finfo(energy.dtype).min)
        else:
            k = min(self.top_k, energy.shape[-1])
            threshold = torch.topk(energy, k, dim=-1).values[..., -1:]
            energy = energy.masked_fill(energy < threshold, torch.finfo(energy.dtype).min)

        att = F.softmax(energy, dim=-1)
        out = torch.matmul(att, values)
        out = out.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.emb_size)
        return self.projection(out)


class PaperTransformerBlock(nn.Module):
    """主文风格 Transformer block：稀疏注意力 + 前馈网络 + 残差归一化。"""

    def __init__(
        self,
        emb_size: int = 40,
        heads: int = 10,
        top_k: int = 8,
        dropout: float = 0.5,
        expansion: int = 4,
        random_sparse: bool = False,
        sparsity_factor: float = 0.5,
        use_global_key: bool = True,
    ) -> None:
        super().__init__()
        self.attention = PaperSparseAttention(
            emb_size=emb_size,
            heads=heads,
            top_k=top_k,
            random_sparse=random_sparse,
            sparsity_factor=sparsity_factor,
            use_global_key=use_global_key,
        )
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, expansion * emb_size),
            nn.ReLU(),
            nn.Linear(expansion * emb_size, emb_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.attention(x)
        x = self.dropout(self.norm1(attention + x))
        forward = self.feed_forward(x)
        return self.dropout(self.norm2(forward + x))


def conv3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """残差头中的 3x3 卷积封装。"""

    return nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)


class ChannelAttention(nn.Module):
    """通道注意力模块，用于增强残差块对不同特征通道的选择能力。"""

    def __init__(self, channels: int, reduction: int = 2) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        weights = self.avg_pool(x).view(b, c)
        weights = self.fc(weights).view(b, c, 1, 1)
        return weights


class ShrinkageResidualBlock(nn.Module):
    """带通道注意力的软阈值残差块。

    软阈值分支会根据特征幅值自适应估计阈值，抑制 EEG 中较弱的噪声响应。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        threshold_dropout: float = 0.03,
    ) -> None:
        super().__init__()
        self.conv1 = conv3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.channel_attention = ChannelAttention(out_channels)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.threshold = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(threshold_dropout),
            nn.Linear(out_channels, 1),
            nn.Sigmoid(),
        )
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(conv3(in_channels, out_channels, stride), nn.BatchNorm2d(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out * self.channel_attention(out)

        raw = out
        abs_out = out.abs()
        pooled = self.gap(abs_out).flatten(1)
        scale = pooled.mean(dim=1, keepdim=True) * self.threshold(pooled)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        out = torch.sign(raw) * torch.clamp(abs_out - scale, min=0.0)

        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu(out + residual)
        return out


class PaperResNetHead(nn.Module):
    """主文 STS 残差头，输入为 Transformer 后的序列特征。"""

    def __init__(self, layers: int = 3, threshold_dropout: float = 0.03) -> None:
        super().__init__()
        self.in_channels = 1
        self.threshold_dropout = threshold_dropout
        self.layer1 = self.make_layer(16, layers)
        self.layer2 = self.make_layer(32, layers, stride=2)
        self.avg_pool = nn.AvgPool2d(8)

    def make_layer(self, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers = [ShrinkageResidualBlock(self.in_channels, out_channels, stride, self.threshold_dropout)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ShrinkageResidualBlock(out_channels, out_channels, threshold_dropout=self.threshold_dropout))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2).unsqueeze(1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avg_pool(x)
        return x.flatten(1)


class PaperSTSNet(nn.Module):
    """主文对齐的 STS-Net：CNN stem、稀疏 Transformer、STS-ResNet。"""

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        emb_size: int = 40,
        heads: int = 10,
        top_k: int = 8,
        dropout: float = 0.5,
        conv_kernel: int = 25,
        pool_kernel: int = 75,
        pool_stride: int = 15,
        transformer_depth: int = 1,
        transformer_style: str = "paper",
        res_layers: int = 3,
        random_sparse: bool = False,
        sparsity_factor: float = 0.5,
        use_global_key: bool = True,
        threshold_dropout: float = 0.03,
    ) -> None:
        super().__init__()
        self.conv = LegacyTemporalStem(
            n_channels,
            emb_size=emb_size,
            kernel=conv_kernel,
            dropout=dropout,
            pool_kernel=pool_kernel,
            pool_stride=pool_stride,
        )
        block_cls: type[nn.Module]
        if transformer_style == "paper":
            block_cls = PaperTransformerBlock
        else:
            raise ValueError("Only transformer_style='paper' is included in the manuscript code package.")
        self.transformer = nn.Sequential(
            *[
                block_cls(
                    emb_size=emb_size,
                    heads=heads,
                    top_k=top_k,
                    dropout=dropout,
                    random_sparse=random_sparse,
                    sparsity_factor=sparsity_factor,
                    use_global_key=use_global_key,
                )
                for _ in range(transformer_depth)
            ]
        )
        self.resnet = PaperResNetHead(layers=res_layers, threshold_dropout=threshold_dropout)
        self.fc = nn.Sequential(
            nn.Linear(192, 192),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(192, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.transformer(x)
        x = self.resnet(x)
        return self.fc(x)


class STSNet(nn.Module):
    """统一模型入口。

    训练脚本通过 ``--stem`` 选择主文模型或对照模型。公开复现命令默认使用
    ``--stem paper_exact``，对应 ``PaperSTSNet``。
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        emb_size: int = 40,
        heads: int = 10,
        top_k: int = 8,
        dropout: float = 0.5,
        stem: str = "multiscale",
        kernels: Iterable[int] = (15, 25, 51),
        branch_channels: int = 16,
        conv_kernel: int = 25,
        pool_kernel: int = 75,
        pool_stride: int = 15,
        transformer_depth: int = 1,
        transformer_style: str = "paper",
        res_layers: int = 3,
        random_sparse: bool = False,
        sparsity_factor: float = 0.5,
        use_global_key: bool = True,
        classifier_width: int = 192,
        threshold_dropout: float = 0.03,
    ) -> None:
        super().__init__()
        if stem == "paper_exact":
            self.model = PaperSTSNet(
                n_channels=n_channels,
                n_classes=n_classes,
                emb_size=emb_size,
                heads=heads,
                top_k=top_k,
                dropout=dropout,
                conv_kernel=conv_kernel,
                pool_kernel=pool_kernel,
                pool_stride=pool_stride,
                transformer_depth=transformer_depth,
                transformer_style=transformer_style,
                res_layers=res_layers,
                random_sparse=random_sparse,
                sparsity_factor=sparsity_factor,
                use_global_key=use_global_key,
                threshold_dropout=threshold_dropout,
            )
            return
        raise ValueError("Only stem='paper_exact' is included in the manuscript code package.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

