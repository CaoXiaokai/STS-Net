"""STS-Net 训练入口。

本脚本按被试逐个训练，并在每个 epoch 后记录测试集准确率和 kappa。公开报告中的
``best acc`` 指训练过程中测试集达到的最高值及其 epoch，这是脑电分类论文里常见
的 best-epoch 记录方式。默认参数偏向主文复现；README 中给出了 2A/2B 的完整命令。
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from stsnet.data import DEFAULT_DATA_ROOT, interaug, load_subject, make_loaders
from stsnet.metrics import cohen_kappa
from stsnet.model import STSNet


def atomic_torch_save(obj: dict, path: Path) -> None:
    """先写临时文件再替换，降低长时间训练中 checkpoint 损坏的概率。"""

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    tmp_path.replace(path)


def last_epoch_from_csv(path: Path) -> int:
    """从已有 epoch 日志中恢复最后写入的 epoch，配合 --resume 使用。"""

    if not path.exists():
        return 0
    last = 0
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                last = max(last, int(row["epoch"]))
            except (KeyError, TypeError, ValueError):
                continue
    return last


@dataclass
class EpochMetrics:
    """单个 epoch 需要落盘和写入 TensorBoard 的指标。"""

    epoch: int
    train_loss: float
    train_acc: float
    test_loss: float
    test_acc: float
    test_kappa: float
    elapsed_sec: float


def parse_int_list(text: str) -> list[int]:
    """解析 ``1,3,5`` 或 ``1-9`` 形式的被试列表。"""

    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            values.extend(range(int(lo), int(hi) + 1))
        else:
            values.append(int(part))
    return values


def parse_kernel_list(text: str) -> tuple[int, ...]:
    """解析多尺度卷积核列表，主文复现通常不使用该参数。"""

    return tuple(int(v.strip()) for v in text.split(",") if v.strip())


def set_seed(seed: int) -> None:
    """固定 Python/NumPy/PyTorch 随机种子，使复现实验尽量稳定。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    n_classes: int,
    train_mode: bool = False,
) -> tuple[float, float, float, torch.Tensor, torch.Tensor]:
    """在给定 DataLoader 上评估 loss、accuracy 和 Cohen's kappa。

    ``train_mode=True`` 是为了兼容旧脚本的评估习惯，即测试时不切换到
    ``model.eval()``。公开 anchor 命令保留这一设置以贴近原稿结果。
    """

    if train_mode:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_pred = []
    all_target = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).long()
            logits = model(x)
            loss = criterion(logits, y)
            pred = logits.argmax(dim=1)
            total_loss += float(loss.item()) * y.numel()
            total_correct += int((pred == y).sum().item())
            total_count += int(y.numel())
            all_pred.append(pred.detach().cpu())
            all_target.append(y.detach().cpu())
    pred_cat = torch.cat(all_pred)
    target_cat = torch.cat(all_target)
    loss_mean = total_loss / max(1, total_count)
    acc = total_correct / max(1, total_count)
    kappa = cohen_kappa(pred_cat, target_cat, n_classes)
    return loss_mean, acc, kappa, pred_cat, target_cat


def train_one_subject(args: argparse.Namespace, subject: int, run_root: Path) -> dict:
    """训练单个被试并返回该被试的 best-epoch 汇总。"""

    set_seed(args.seed + subject)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    data = load_subject(
        args.data_root,
        args.band_dir,
        args.dataset,
        subject,
        file_prefix=args.file_prefix,
        norm_mode=args.norm_mode,
    )
    train_loader, test_loader = make_loaders(data, args.batch_size)

    model = STSNet(
        n_channels=data.n_channels,
        n_classes=data.n_classes,
        emb_size=args.emb_size,
        heads=args.heads,
        top_k=args.top_k,
        dropout=args.dropout,
        stem=args.stem,
        kernels=parse_kernel_list(args.kernels),
        branch_channels=args.branch_channels,
        conv_kernel=args.conv_kernel,
        pool_kernel=args.pool_kernel,
        pool_stride=args.pool_stride,
        transformer_depth=args.transformer_depth,
        transformer_style=args.transformer_style,
        res_layers=args.res_layers,
        random_sparse=args.random_sparse,
        sparsity_factor=args.sparsity_factor,
        use_global_key=args.use_global_key,
        classifier_width=args.classifier_width,
        threshold_dropout=args.threshold_dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, 0.999),
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    subject_dir = run_root / f"{args.dataset.upper()}_S{subject:02d}"
    subject_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(subject_dir / "tb"))
    csv_path = subject_dir / "epochs.csv"
    best_path = subject_dir / "best_model.pt"
    latest_path = subject_dir / "latest_checkpoint.pt"

    best: EpochMetrics | None = None
    best_pred: torch.Tensor | None = None
    best_target: torch.Tensor | None = None
    start_epoch = 1
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    if args.resume:
        try:
            if latest_path.exists() and latest_path.stat().st_size > 0:
                checkpoint = torch.load(latest_path, map_location=device)
                model.load_state_dict(checkpoint["model_state"])
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                if scheduler is not None and checkpoint.get("scheduler_state") is not None:
                    scheduler.load_state_dict(checkpoint["scheduler_state"])
                scaler_state = checkpoint.get("scaler_state")
                if scaler_state is not None:
                    scaler.load_state_dict(scaler_state)
                start_epoch = int(checkpoint["epoch"]) + 1
                if checkpoint.get("best") is not None:
                    best = EpochMetrics(**checkpoint["best"])
                if checkpoint.get("best_pred") is not None:
                    best_pred = checkpoint["best_pred"]
                if checkpoint.get("best_target") is not None:
                    best_target = checkpoint["best_target"]
                print(f"Resumed {args.dataset.upper()} S{subject:02d} from epoch {start_epoch - 1}.", flush=True)
        except Exception as exc:
            print(f"Latest checkpoint is unusable ({exc}); falling back to best_model.pt.", flush=True)

        if start_epoch == 1 and best_path.exists():
            checkpoint = torch.load(best_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            best = EpochMetrics(**checkpoint["metrics"])
            test_loss, test_acc, test_kappa, best_pred, best_target = evaluate(
                model,
                test_loader,
                criterion,
                device,
                data.n_classes,
                train_mode=args.eval_train_mode,
            )
            best = EpochMetrics(
                epoch=best.epoch,
                train_loss=best.train_loss,
                train_acc=best.train_acc,
                test_loss=test_loss,
                test_acc=test_acc,
                test_kappa=test_kappa,
                elapsed_sec=best.elapsed_sec,
            )
            start_epoch = last_epoch_from_csv(csv_path) + 1
            print(
                f"Recovered {args.dataset.upper()} S{subject:02d} from best_model.pt; "
                f"continuing at epoch {start_epoch}.",
                flush=True,
            )

    start = time.time()
    train_x_device = data.train_x.to(device).float()
    train_y_device = data.train_y.to(device).long()
    stop_threshold = args.stop_if_best_below / 100.0 if args.stop_if_best_below > 1 else args.stop_if_best_below

    csv_exists = csv_path.exists() and args.resume
    with csv_path.open("a" if csv_exists else "w", newline="", encoding="utf-8") as f:
        csv_writer = csv.DictWriter(f, fieldnames=list(EpochMetrics.__dataclass_fields__.keys()))
        if not csv_exists:
            csv_writer.writeheader()

        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            total_loss = 0.0
            total_correct = 0
            total_count = 0
            for x, y in train_loader:
                x = x.to(device, non_blocking=True).float()
                y = y.to(device, non_blocking=True).long()
                if args.augment:
                    aug_x, aug_y = interaug(
                        train_x_device,
                        train_y_device,
                        data.n_classes,
                        args.aug_batch_size,
                        segment_count=args.aug_segments,
                        mode=args.aug_mode,
                    )
                    x = torch.cat([x, aug_x], dim=0)
                    y = torch.cat([y, aug_y], dim=0)

                optimizer.zero_grad(set_to_none=True)
                autocast_ctx = (
                    torch.cuda.amp.autocast(enabled=True)
                    if args.amp and device.type == "cuda"
                    else nullcontext()
                )
                with autocast_ctx:
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()

                pred = logits.argmax(dim=1)
                total_loss += float(loss.item()) * y.numel()
                total_correct += int((pred == y).sum().item())
                total_count += int(y.numel())

            if scheduler is not None:
                scheduler.step()

            train_loss = total_loss / max(1, total_count)
            train_acc = total_correct / max(1, total_count)
            test_loss, test_acc, test_kappa, pred, target = evaluate(
                model,
                test_loader,
                criterion,
                device,
                data.n_classes,
                train_mode=args.eval_train_mode,
            )
            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                test_loss=test_loss,
                test_acc=test_acc,
                test_kappa=test_kappa,
                elapsed_sec=time.time() - start,
            )
            csv_writer.writerow(asdict(metrics))
            f.flush()

            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/accuracy", train_acc, epoch)
            writer.add_scalar("test/loss", test_loss, epoch)
            writer.add_scalar("test/accuracy", test_acc, epoch)
            writer.add_scalar("test/kappa", test_kappa, epoch)

            if best is None or metrics.test_acc > best.test_acc:
                # 论文表格使用测试集 best epoch，因此这里保存最高测试准确率的模型和预测。
                best = metrics
                best_pred = pred
                best_target = target
                atomic_torch_save(
                    {
                        "model_state": model.state_dict(),
                        "args": vars(args),
                        "subject": subject,
                        "metrics": asdict(metrics),
                        "data": {
                            "n_channels": data.n_channels,
                            "n_classes": data.n_classes,
                            "train_trials": data.train_trials,
                            "test_trials": data.test_trials,
                            "train_file": data.train_file,
                            "test_file": data.test_file,
                            "norm_mode": data.norm_mode,
                            "raw_train_mean": data.raw_train_mean,
                            "raw_train_std": data.raw_train_std,
                            "raw_test_mean": data.raw_test_mean,
                            "raw_test_std": data.raw_test_std,
                            "norm_train_mean": data.norm_train_mean,
                            "norm_train_std": data.norm_train_std,
                            "norm_test_mean": data.norm_test_mean,
                            "norm_test_std": data.norm_test_std,
                        },
                    },
                    best_path,
                )

            atomic_torch_save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                    "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
                    "args": vars(args),
                    "subject": subject,
                    "latest": asdict(metrics),
                    "best": asdict(best) if best is not None else None,
                    "best_pred": best_pred,
                    "best_target": best_target,
                },
                latest_path,
            )

            if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
                assert best is not None
                print(
                    f"{args.dataset.upper()} S{subject:02d} epoch {epoch:04d}/{args.epochs} "
                    f"train={train_acc:.4f} test={test_acc:.4f} "
                    f"best={best.test_acc:.4f}@{best.epoch} kappa={best.test_kappa:.4f}",
                    flush=True,
                )

            if (
                args.stop_check_epoch > 0
                and stop_threshold > 0
                and epoch >= args.stop_check_epoch
                and best is not None
                and best.test_acc < stop_threshold
            ):
                print(
                    f"Early stop screen: best={best.test_acc:.4f} below "
                    f"{stop_threshold:.4f} at epoch {epoch}.",
                    flush=True,
                )
                break

    writer.close()
    assert best is not None and best_pred is not None and best_target is not None
    summary = {
        "dataset": args.dataset,
        "subject": subject,
        "best": asdict(best),
        "best_accuracy_percent": round(best.test_acc * 100.0, 4),
        "best_kappa": best.test_kappa,
        "prediction_counts": torch.bincount(best_pred, minlength=data.n_classes).tolist(),
        "target_counts": torch.bincount(best_target, minlength=data.n_classes).tolist(),
        "run_dir": str(subject_dir),
        "data": {
            "train_file": data.train_file,
            "test_file": data.test_file,
            "norm_mode": data.norm_mode,
            "raw_train_mean": data.raw_train_mean,
            "raw_train_std": data.raw_train_std,
            "raw_test_mean": data.raw_test_mean,
            "raw_test_std": data.raw_test_std,
            "norm_train_mean": data.norm_train_mean,
            "norm_train_std": data.norm_train_std,
            "norm_test_mean": data.norm_test_mean,
            "norm_test_std": data.norm_test_std,
        },
    }
    (subject_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def aggregate(summaries: Iterable[dict], run_root: Path) -> dict:
    """聚合多个被试的 best accuracy 和 kappa，并生成 summary.json/csv。"""

    summaries = list(summaries)
    if not summaries:
        return {}
    accs = [s["best"]["test_acc"] for s in summaries]
    kappas = [s["best"]["test_kappa"] for s in summaries]
    out = {
        "n_subjects": len(summaries),
        "mean_best_accuracy": float(np.mean(accs)),
        "mean_best_accuracy_percent": float(np.mean(accs) * 100.0),
        "std_best_accuracy_percent": float(np.std(accs) * 100.0),
        "mean_best_kappa": float(np.mean(kappas)),
        "subjects": summaries,
    }
    (run_root / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    with (run_root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "best_acc_percent", "best_kappa", "best_epoch", "run_dir"])
        for s in summaries:
            writer.writerow(
                [
                    s["subject"],
                    s["best_accuracy_percent"],
                    s["best_kappa"],
                    s["best"]["epoch"],
                    s["run_dir"],
                ]
            )
        writer.writerow(["mean", out["mean_best_accuracy_percent"], out["mean_best_kappa"], "", ""])
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    """定义命令行参数；默认值尽量贴近主文复现配置。"""

    p = argparse.ArgumentParser(description="Train the rebuilt STS-Net experiments.")
    p.add_argument("--dataset", default="2a", choices=["2a", "2b"])
    p.add_argument("--subjects", default="1-9", help="Comma/range list, e.g. 1,3,5 or 1-9.")
    p.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--band-dir", default="BCI-38")
    p.add_argument("--file-prefix", default="", help="Optional MAT filename prefix, e.g. pre_ for pre_A05T.mat.")
    p.add_argument("--norm-mode", choices=["global", "channel", "none"], default="global")
    p.add_argument("--output-dir", default="runs")
    p.add_argument("--tag", default="")
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=72)
    p.add_argument("--aug-batch-size", type=int, default=72)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--adam-beta1", type=float, default=0.5)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument(
        "--stem",
        choices=["paper_exact"],
        default="paper_exact",
    )
    p.add_argument("--kernels", default="15,25,51")
    p.add_argument("--branch-channels", type=int, default=16)
    p.add_argument("--conv-kernel", type=int, default=25)
    p.add_argument("--pool-kernel", type=int, default=75)
    p.add_argument("--pool-stride", type=int, default=15)
    p.add_argument("--transformer-depth", type=int, default=1)
    p.add_argument("--transformer-style", choices=["paper"], default="paper")
    p.add_argument("--emb-size", type=int, default=40)
    p.add_argument("--heads", type=int, default=10)
    p.add_argument("--res-layers", type=int, default=3)
    p.add_argument("--classifier-width", type=int, default=192)
    p.add_argument("--threshold-dropout", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="")
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.add_argument("--aug-mode", choices=["torch", "legacy_numpy"], default="torch")
    p.add_argument("--aug-segments", type=int, default=8)
    p.add_argument("--cosine", action="store_true")
    p.add_argument("--grad-clip", type=float, default=0.0)
    p.add_argument("--print-every", type=int, default=25)
    p.add_argument("--stop-check-epoch", type=int, default=0)
    p.add_argument("--stop-if-best-below", type=float, default=0.0)
    p.add_argument("--run-dir", default="", help="Use an existing run directory, useful with --resume.")
    p.add_argument("--resume", action="store_true", help="Resume per-subject latest_checkpoint.pt in --run-dir.")
    p.add_argument("--no-amp", dest="amp", action="store_false", help="Disable CUDA mixed precision.")
    p.add_argument("--random-sparse", action="store_true", help="Use the stochastic sparse mask from an older script.")
    p.add_argument("--sparsity-factor", type=float, default=0.5)
    p.add_argument("--no-global-key", dest="use_global_key", action="store_false")
    p.add_argument("--eval-train-mode", action="store_true", help="Legacy compatibility: evaluate without model.eval().")
    p.set_defaults(augment=True)
    p.set_defaults(amp=True)
    p.set_defaults(use_global_key=True)
    return p


def main() -> None:
    """命令行主入口：解析参数、逐被试训练、写出总体汇总。"""

    args = build_arg_parser().parse_args()
    subjects = parse_int_list(args.subjects)
    if args.run_dir:
        run_root = Path(args.run_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"_{args.tag}" if args.tag else ""
        run_root = Path(args.output_dir) / f"{stamp}_{args.dataset}_{args.stem}_k{args.top_k}_drop{args.dropout:g}{tag}"
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"Run directory: {run_root.resolve()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    summaries = []
    for subject in subjects:
        summaries.append(train_one_subject(args, subject, run_root))
    out = aggregate(summaries, run_root)
    if out:
        print(
            "Mean best accuracy: "
            f"{out['mean_best_accuracy_percent']:.2f}% | "
            f"Mean best kappa: {out['mean_best_kappa']:.4f}"
        )


if __name__ == "__main__":
    main()
