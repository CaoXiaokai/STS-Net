"""STS-Net 数据读取、归一化与伪样本增强工具。

公开版默认假设预处理后的 BCI Competition IV 2a/2b 数据放在项目根目录的
``data/BCI-38`` 下。MAT 文件中的 ``data`` 原始形状为
``time x channels x trials``，这里统一转换为 PyTorch 常用的
``trials x 1 x channels x time``，便于直接送入二维卷积网络。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import scipy.io
import torch
from torch.utils.data import DataLoader, TensorDataset


# 项目根目录下的 data 文件夹。这样公开代码后无需修改本机绝对路径。
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class SubjectData:
    """单个被试的训练/测试数据及复现实验需要记录的统计信息。"""

    train_x: torch.Tensor
    train_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    n_channels: int
    n_classes: int
    train_trials: int
    test_trials: int
    train_file: str
    test_file: str
    norm_mode: str
    raw_train_mean: float
    raw_train_std: float
    raw_test_mean: float
    raw_test_std: float
    norm_train_mean: float
    norm_train_std: float
    norm_test_mean: float
    norm_test_std: float


def dataset_prefix(dataset: str) -> Tuple[str, int]:
    """把命令行中的数据集别名转换为文件名前缀和类别数。"""

    normalized = dataset.lower()
    if normalized in {"2a", "iv2a", "bci2a"}:
        return "A", 4
    if normalized in {"2b", "iv2b", "bci2b"}:
        return "B", 2
    raise ValueError(f"Unsupported dataset: {dataset!r}. Use 2a or 2b.")


def _load_mat(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """读取一个 AxxT/E 或 BxxT/E 的 MAT 文件，并转成网络输入格式。"""

    if not path.exists():
        raise FileNotFoundError(path)
    mat = scipy.io.loadmat(path)
    data = mat["data"]
    label = mat["label"].reshape(-1)
    data = np.transpose(data, (2, 1, 0))
    data = np.expand_dims(data, axis=1)
    return data.astype(np.float32), label.astype(np.int64) - 1


def _normalize(
    train_x: np.ndarray,
    test_x: np.ndarray,
    mode: Literal["global", "channel", "none"],
) -> Tuple[np.ndarray, np.ndarray]:
    """只用训练集统计量归一化训练集和测试集，避免测试集信息泄漏。"""

    if mode == "none":
        return train_x, test_x
    if mode == "global":
        mean = train_x.mean()
        std = train_x.std()
    elif mode == "channel":
        mean = train_x.mean(axis=(0, 1, 3), keepdims=True)
        std = train_x.std(axis=(0, 1, 3), keepdims=True)
    else:
        raise ValueError(f"Unsupported norm_mode: {mode!r}. Use global, channel, or none.")
    std = np.where(std == 0, 1.0, std)
    return (train_x - mean) / std, (test_x - mean) / std


def load_subject(
    data_root: Path | str = DEFAULT_DATA_ROOT,
    band_dir: str = "BCI-38",
    dataset: str = "2a",
    subject: int = 1,
    file_prefix: str = "",
    norm_mode: Literal["global", "channel", "none"] = "global",
) -> SubjectData:
    """加载单个被试。

    ``data_root`` 指向包含 ``BCI-38`` 的目录；``band_dir`` 保留为参数，是为了
    方便以后切换其他预处理版本。标签在 MAT 中为 1 起始，这里统一改成 0 起始。
    """

    root = Path(data_root) / band_dir
    prefix, n_classes = dataset_prefix(dataset)
    stem = f"{file_prefix}{prefix}{subject:02d}"
    train_path = root / f"{stem}T.mat"
    test_path = root / f"{stem}E.mat"
    train_x, train_y = _load_mat(train_path)
    test_x, test_y = _load_mat(test_path)

    raw_train_mean = float(train_x.mean())
    raw_train_std = float(train_x.std())
    raw_test_mean = float(test_x.mean())
    raw_test_std = float(test_x.std())
    train_x, test_x = _normalize(train_x, test_x, norm_mode)

    return SubjectData(
        train_x=torch.from_numpy(train_x),
        train_y=torch.from_numpy(train_y),
        test_x=torch.from_numpy(test_x),
        test_y=torch.from_numpy(test_y),
        n_channels=int(train_x.shape[2]),
        n_classes=n_classes,
        train_trials=int(train_x.shape[0]),
        test_trials=int(test_x.shape[0]),
        train_file=str(train_path),
        test_file=str(test_path),
        norm_mode=norm_mode,
        raw_train_mean=raw_train_mean,
        raw_train_std=raw_train_std,
        raw_test_mean=raw_test_mean,
        raw_test_std=raw_test_std,
        norm_train_mean=float(train_x.mean()),
        norm_train_std=float(train_x.std()),
        norm_test_mean=float(test_x.mean()),
        norm_test_std=float(test_x.std()),
    )


def make_loaders(
    subject_data: SubjectData,
    batch_size: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """构造训练和测试 DataLoader；小样本 EEG 默认不丢弃最后一个 batch。"""

    train_ds = TensorDataset(subject_data.train_x, subject_data.train_y)
    test_ds = TensorDataset(subject_data.test_x, subject_data.test_y)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def interaug(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    n_classes: int,
    batch_size: int,
    segment_count: int = 8,
    generator: torch.Generator | None = None,
    mode: Literal["torch", "legacy_numpy"] = "torch",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """分段重组伪样本增强。

    每一类内部随机抽取若干 trial 的时间片段再拼接成新 trial。该策略保留了
    类内 EEG 片段分布，同时增加了小样本训练时的批内多样性。
    """

    if mode == "legacy_numpy":
        return interaug_legacy_numpy(train_x, train_y, n_classes, batch_size, segment_count)
    if mode != "torch":
        raise ValueError(f"Unsupported interaug mode: {mode}")
    per_class = max(1, batch_size // n_classes)
    _, _, n_channels, n_times = train_x.shape
    seg_len = n_times // segment_count
    aug_x = []
    aug_y = []

    for cls in range(n_classes):
        cls_idx = torch.where(train_y == cls)[0]
        if cls_idx.numel() == 0:
            continue
        cls_trials = train_x[cls_idx]
        out = torch.zeros(
            (per_class, 1, n_channels, n_times),
            dtype=train_x.dtype,
            device=train_x.device,
        )
        rand = torch.randint(
            0,
            cls_trials.shape[0],
            (per_class, segment_count),
            generator=generator,
            device=train_x.device,
        )
        for j in range(segment_count):
            start = j * seg_len
            end = n_times if j == segment_count - 1 else (j + 1) * seg_len
            out[:, :, :, start:end] = cls_trials[rand[:, j], :, :, start:end]
        aug_x.append(out)
        aug_y.append(torch.full((per_class,), cls, dtype=train_y.dtype, device=train_y.device))

    if not aug_x:
        return train_x[:0], train_y[:0]
    aug_x_t = torch.cat(aug_x, dim=0)
    aug_y_t = torch.cat(aug_y, dim=0)
    perm = torch.randperm(aug_x_t.shape[0], generator=generator, device=train_x.device)
    return aug_x_t[perm], aug_y_t[perm]


def interaug_legacy_numpy(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    n_classes: int,
    batch_size: int,
    segment_count: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """与旧脚本随机路径一致的 NumPy 版 S&R 伪样本增强。"""

    device = train_x.device
    x_np = train_x.detach().cpu().numpy()
    y_np = train_y.detach().cpu().numpy()
    per_class = max(1, batch_size // n_classes)
    _, _, n_channels, n_times = x_np.shape
    seg_len = n_times // segment_count
    aug_x = []
    aug_y = []

    for cls in range(n_classes):
        cls_trials = x_np[y_np == cls]
        if cls_trials.shape[0] == 0:
            continue
        out = np.zeros((per_class, 1, n_channels, n_times), dtype=x_np.dtype)
        for ri in range(per_class):
            for rj in range(segment_count):
                rand_idx = np.random.randint(0, cls_trials.shape[0], segment_count)
                start = rj * seg_len
                end = n_times if rj == segment_count - 1 else (rj + 1) * seg_len
                out[ri, :, :, start:end] = cls_trials[rand_idx[rj], :, :, start:end]
        aug_x.append(out)
        aug_y.append(np.full((per_class,), cls, dtype=y_np.dtype))

    if not aug_x:
        return train_x[:0], train_y[:0]
    aug_x_np = np.concatenate(aug_x, axis=0)
    aug_y_np = np.concatenate(aug_y, axis=0)
    perm = np.random.permutation(len(aug_x_np))
    aug_x_np = aug_x_np[perm]
    aug_y_np = aug_y_np[perm]
    aug_x_t = torch.from_numpy(aug_x_np).to(device=device, dtype=train_x.dtype)
    aug_y_t = torch.from_numpy(aug_y_np).to(device=device, dtype=train_y.dtype)
    return aug_x_t, aug_y_t
