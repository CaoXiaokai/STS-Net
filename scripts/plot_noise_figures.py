"""Reproduce the two EOG accuracy figures included in the manuscript.

This script reads existing results only. It does not train any model. The seven-
method comparison CSV contains the user's reported server results; it is not
evidence that the six baseline training implementations are bundled here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

SNRS = [10, 12, 15, 20, 25]
SUBJECTS = [f"A{i:02}" for i in range(1, 10)]
METHODS = ["FBCSP", "ConvNet", "EEGNet", "C2CM", "FBCNet", "DRDA", "STS-Net"]
SUBJECT_COLORS = ["#2166AC", "#E1812C", "#2A8C55", "#C74343", "#8061A8",
                  "#986A45", "#C25D99", "#808326", "#1599A5"]
SUBJECT_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
SUBJECT_STYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-"]
METHOD_STYLES = [("#0072B2", "o", "--"), ("#E69F00", "s", "-."),
                 ("#009E73", "^", "-"), ("#AA6DA3", "D", ":"),
                 ("#36A8CA", "v", "--"), ("#666666", "P", "-."),
                 ("#D55E00", "*", "-")]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_subject_accuracy(path: Path) -> np.ndarray:
    rows = [row for row in read_csv(path) if row["kind"] == "eog"]
    keyed = {(float(row["snr_db"]), row["subject"]): row for row in rows}
    expected = {(float(snr), subject) for snr in SNRS for subject in SUBJECTS}
    if len(rows) != 45 or set(keyed) != expected:
        raise ValueError("Expected exactly one EOG result for A01-A09 at each of 10, 12, 15, 20, 25 dB.")
    values = np.array([[float(keyed[(snr, subject)]["accuracy"])
                        for snr in SNRS] for subject in SUBJECTS])
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError("Subject accuracy must be a finite fraction between 0 and 1.")
    return values


def read_comparison_accuracy(path: Path) -> np.ndarray:
    rows = read_csv(path)
    keyed = {float(row["EOG_SNR_dB"]): row for row in rows}
    if len(rows) != 5 or set(keyed) != set(SNRS):
        raise ValueError("Comparison CSV must contain each requested SNR exactly once.")
    values = np.array([[float(keyed[snr][method]) for snr in SNRS]
                       for method in METHODS])
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 100)).all():
        raise ValueError("Comparison accuracies must be finite percentages between 0 and 100.")
    return values


def configure_font() -> str:
    # The published figures use Times New Roman. Retain a portable fallback so
    # another computer can regenerate the plots without Windows font paths.
    font = "Times New Roman"
    try:
        font_manager.findfont(font, fallback_to_default=False)
    except ValueError:
        font = "DejaVu Serif"
        warnings.warn("Times New Roman is unavailable; using DejaVu Serif. Install Times New Roman for the original typography.")
    plt.rcParams.update({"font.family": font, "font.size": 12,
        "mathtext.fontset": "stix", "axes.labelsize": 13,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "axes.linewidth": .8, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white"})
    return font


def plot_comparison(values: np.ndarray, out: Path) -> dict:
    title = "Accuracy of Seven Methodsat Different EOG Artifact SNR Levels"
    # Keep the manuscript's title verbatim, including "Methodsat".
    with plt.rc_context({"axes.titlesize": 14, "legend.fontsize": 11.5}):
        fig, ax = plt.subplots(figsize=(9.2, 6.0))
        fig.subplots_adjust(left=.105, right=.975, bottom=.14, top=.87)
        for name, ys, (color, marker, linestyle) in zip(METHODS, values, METHOD_STYLES):
            is_sts = name == "STS-Net"
            ax.plot(SNRS, ys, label=name, color=color, marker=marker,
                linestyle=linestyle, linewidth=2.5 if is_sts else 1.8,
                markersize=10.2 if is_sts else 6.4,
                markerfacecolor=color if is_sts else "white",
                markeredgewidth=1., zorder=5 if is_sts else 3)
        ax.set_title(title, pad=17, linespacing=1.25)
        ax.set_xlabel("EOG Artifact SNR (dB)", labelpad=9)
        ax.set_ylabel("Accuracy(%)", labelpad=9)
        ax.set(xticks=SNRS, xlim=(25.5, 9.5), ylim=(45, 85), yticks=range(45, 86, 5))
        ax.grid(axis="y", color="#DDDDDD", linewidth=.65)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="out", length=4, width=.8)
        legend = ax.legend(loc="lower left", ncol=2, frameon=True, framealpha=1,
            facecolor="white", edgecolor="#CCCCCC", handlelength=2.7,
            columnspacing=1.3, borderpad=.65, labelspacing=.6)
        legend.get_frame().set_linewidth(.6)
        if values.min() <= 45 or values.max() >= 85:
            warnings.warn("Comparison data extend outside the original manuscript y-axis range (45, 85).")
        validate_lines(ax, values, METHODS)
        base = out / "eog_snr_accuracy_comparison"
        fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=.16)
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.16,
            metadata={"Title": title, "Subject": "BCI Competition IV 2a; user-supplied averages of nine subjects"})
        plt.close(fig)
    return {"title": title, "line_count": 7, "series": METHODS, "accuracy_percent": values.tolist()}


def validate_lines(ax, expected: np.ndarray, labels: list[str]) -> None:
    if len(ax.lines) != len(labels) or ax.get_xlim()[0] <= ax.get_xlim()[1]:
        raise AssertionError("Line count or descending SNR axis differs from the manuscript.")
    for line, values, label in zip(ax.lines, expected, labels):
        np.testing.assert_array_equal(line.get_xdata(), SNRS)
        np.testing.assert_allclose(line.get_ydata(), values, rtol=0, atol=1e-13)
        if line.get_label() != label:
            raise AssertionError("Legend label does not match the data series.")


def plot_subjects(values: np.ndarray, out: Path) -> dict:
    title = "Accuracy at Different EOG Artifact SNR Levels"
    average = values.mean(axis=0)
    with plt.rc_context({"axes.titlesize": 15}):
        fig, ax = plt.subplots(figsize=(7.4, 5.35))
        fig.subplots_adjust(left=.115, right=.975, bottom=.25, top=.9)
        for i, subject in enumerate(SUBJECTS):
            ax.plot(SNRS, values[i] * 100, label=subject, color=SUBJECT_COLORS[i],
                marker=SUBJECT_MARKERS[i], linestyle=SUBJECT_STYLES[i], linewidth=1.35,
                markersize=5.5, markeredgewidth=.85, markerfacecolor="white", zorder=3)
        ax.plot(SNRS, average * 100, label="Average", color="#111111",
            marker="D", linewidth=2.6, markersize=6.2, markeredgewidth=.9,
            markerfacecolor="#111111", zorder=6)
        ax.set(xlim=(25.55, 9.45), ylim=(30, 100), xticks=SNRS,
            yticks=np.arange(30, 101, 10), xlabel="Signal-to-noise ratio (dB)",
            ylabel="Accuracy (%)", title=title)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#DFE3E8", linewidth=.65, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(direction="out", length=4, width=.8)
        handles, labels = ax.get_legend_handles_labels()
        order = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]
        legend = fig.legend([handles[i] for i in order], [labels[i] for i in order],
            loc="lower center", bbox_to_anchor=(.53, .018), ncol=5, frameon=False,
            handlelength=2.7, columnspacing=1.65, handletextpad=.55,
            labelspacing=.7, fontsize=11.5)
        for text in legend.get_texts():
            if text.get_text() == "Average":
                text.set_weight("bold")
        validate_lines(ax, np.vstack([values, average]) * 100, SUBJECTS + ["Average"])
        for extension in ("png", "pdf"):
            fig.savefig(out / f"eog_accuracy_by_subject.{extension}", dpi=600)
        plt.close(fig)
    return {"title": title, "line_count": 10, "series": SUBJECTS + ["Average"],
        "average_accuracy_percent_unrounded": (average * 100).tolist(),
        "accuracy_percent": (values * 100).tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", required=True, type=Path,
        help="Subject-level noise results; accuracy values are fractions in [0, 1].")
    parser.add_argument("--comparison-csv", required=True, type=Path,
        help="Seven-method server comparison; accuracy values are percentages.")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    hashes_before = {str(path.resolve()): sha256(path) for path in (args.results_csv, args.comparison_csv)}
    values = read_subject_accuracy(args.results_csv)
    comparison = read_comparison_accuracy(args.comparison_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    font = configure_font()
    comparison_check = plot_comparison(comparison, args.output_dir)
    subject_check = plot_subjects(values, args.output_dir)
    if hashes_before != {str(path.resolve()): sha256(path) for path in (args.results_csv, args.comparison_csv)}:
        raise RuntimeError("Input files changed during plotting.")
    report = {"status": "passed", "input_sha256": hashes_before, "font": font,
        "snr_left_to_right_db": list(reversed(SNRS)), "input_files_unchanged": True,
        "comparison_source": "User-supplied server results; plotting only, no baseline training code.",
        "comparison": comparison_check, "subjects": subject_check}
    (args.output_dir / "plot-verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "output_dir": str(args.output_dir.resolve()),
                      "figure_count": 2, "files_per_figure": ["pdf", "png"]}))


if __name__ == "__main__":
    main()
