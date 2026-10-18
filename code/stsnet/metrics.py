"""实验指标。

论文表格主要报告 accuracy 和 Cohen's kappa。这里手写 kappa 计算，避免为了一个
指标额外引入 scikit-learn 依赖，也便于在公开代码中直接查看计算过程。
"""

from __future__ import annotations

import torch


def accuracy_from_logits(logits: torch.Tensor, target: torch.Tensor) -> float:
    """从分类 logits 计算普通准确率。"""

    pred = logits.argmax(dim=1)
    return float((pred == target).sum().item() / max(1, target.numel()))


def cohen_kappa(pred: torch.Tensor, target: torch.Tensor, n_classes: int) -> float:
    """计算 Cohen's kappa，衡量预测与标签超出随机一致性的程度。"""

    pred = pred.detach().cpu().long()
    target = target.detach().cpu().long()
    total = int(target.numel())
    if total == 0:
        return 0.0
    confusion = torch.zeros((n_classes, n_classes), dtype=torch.float64)
    for t, p in zip(target, pred):
        if 0 <= int(t) < n_classes and 0 <= int(p) < n_classes:
            confusion[int(t), int(p)] += 1
    po = torch.trace(confusion) / total
    row = confusion.sum(dim=1)
    col = confusion.sum(dim=0)
    pe = (row * col).sum() / (total * total)
    denom = 1.0 - float(pe)
    if abs(denom) < 1e-12:
        return 0.0
    return float((float(po) - float(pe)) / denom)
