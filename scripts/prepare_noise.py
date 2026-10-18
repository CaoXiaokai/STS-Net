"""Prepare the paper's 63 fixed noisy T/E pairs without training a model."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import numpy as np
import scipy
from scipy import io, signal
from noise_common import (
    PACKAGE_ROOT, FORMAT_ID, SPLIT_SEEDS, NOISE_SEEDS, task_specs, task_id,
    task_folder, read_json, write_json, sha256, validate_config, source_manifest,
    make_bank, split_bank, build_noise,
)


def load_banks(emg_path, eog_path, eog_mat_path):
    specification = source_manifest()["files"]
    sources = ((Path(emg_path), "EMG_all_epochs_512hz.npy"),
               (Path(eog_path), "EOG_all_epochs.npy"),
               (Path(eog_mat_path), "EOG_all_epochs.mat"))
    for path, name in sources:
        if sha256(path) != specification[name]["sha256"]:
            raise ValueError(f"Noise source checksum mismatch: {name}")
    emg = np.load(emg_path, allow_pickle=False)
    eog = np.load(eog_path, allow_pickle=False)
    eog_mat = io.loadmat(eog_mat_path)
    assert emg.shape == (5598, 1024) and emg.dtype == np.float64 and np.isfinite(emg).all()
    assert eog.shape == (3400, 512) and eog.dtype == np.float64 and np.isfinite(eog).all()
    assert np.array_equal(eog, eog_mat["EOG_all_epochs"])
    assert float(np.asarray(eog_mat["fs"]).squeeze()) == 256.0
    banks = {}
    for kind, raw, fs in (("emg", emg, 512), ("eog", eog, 256)):
        bank, exact, information = make_bank(raw, fs)
        train_pool, test_pool = split_bank(exact, SPLIT_SEEDS[kind])
        information["canonical_record_count"] = int(len(exact["unique_record_ids"]))
        banks[kind] = (bank, train_pool, test_pool, information)
    return banks


def verify_clean_inputs(data_root):
    band = Path(data_root) / "BCI-38"
    expected = source_manifest()["clean_2a_mat_sha256"]
    assert len(expected) == 18
    for name, digest in expected.items():
        if sha256(band / name) != digest:
            raise ValueError(f"Clean EEG source checksum mismatch: {name}")
    return expected


def prepare_task(output_root, data_root, kind, snr, subject, banks):
    """Keep numerical operations/dtypes/order identical to the original panel."""
    folder = task_folder(output_root, kind, snr, subject)
    if folder.exists():
        raise FileExistsError(f"Refusing to overwrite a prepared task: {folder}")
    folder.mkdir(parents=True)
    bank, train_pool, test_pool, information = banks.get(kind, (None, None, None, None))
    sos = signal.cheby1(8, 0.5, [4.0, 38.0], btype="bandpass", fs=250.0, output="sos")
    train_seed0, test_seed0 = NOISE_SEEDS[kind]
    train_seed = train_seed0 + (subject - 1) * 1000
    test_seed = test_seed0 + (subject - 1) * 1000
    recipe = {"noise_kind": kind, "snr_db": float(snr), "subject": subject,
              "train_seed": train_seed, "test_seed": test_seed,
              "filter_sos": sos, "source_split_seed": SPLIT_SEEDS.get(kind)}
    statistics = {}
    for split, suffix, seed, pool in (("train", "T", train_seed, train_pool),
                                      ("test", "E", test_seed, test_pool)):
        original = io.loadmat(Path(data_root) / "BCI-38" / f"A{subject:02d}{suffix}.mat")
        raw, labels = original["data"], original["label"]
        assert raw.shape == (1000, 22, 288) and labels.shape == (288, 1)
        assert np.array_equal(np.bincount(labels.reshape(-1).astype(int))[1:], [72]*4)
        clean = raw.transpose(2, 1, 0).astype(np.float32)
        assert np.isfinite(clean).all()
        unit, selected = build_noise(kind, bank, pool, seed, sos)
        assert unit.shape == (288, 22, 1000) and unit.dtype == np.float32
        signal_power = np.mean(clean.astype(np.float64) ** 2, axis=(1, 2))
        unit_power = np.mean(unit.astype(np.float64) ** 2, axis=(1, 2))
        scale = np.sqrt(signal_power / (unit_power * (10.0 ** (float(snr) / 10.0))))
        added = (unit * scale[:, None, None]).astype(np.float32)
        noisy = clean + added
        out_path = folder / "data/BCI-38" / f"A{subject:02d}{suffix}.mat"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("xb") as handle:
            io.savemat(handle, {"data": noisy.transpose(2, 1, 0), "label": labels}, do_compression=True)
        restored = io.loadmat(out_path)
        stored = restored["data"].transpose(2, 1, 0)
        assert restored["data"].dtype == np.float32 and np.array_equal(stored, noisy)
        assert np.array_equal(restored["label"], labels)
        actual_added = stored.astype(np.float64) - clean.astype(np.float64)
        actual_noise_power = np.mean(actual_added ** 2, axis=(1, 2))
        actual_snr = 10.0 * np.log10(signal_power / actual_noise_power)
        assert np.isfinite(actual_snr).all() and np.max(np.abs(actual_snr - float(snr))) < 1e-5
        recipe[f"{split}_noise_unit_sha256"] = hashlib.sha256(unit.tobytes(order="C")).hexdigest()
        recipe[f"{split}_unit_power"] = unit_power
        recipe[f"{split}_trial_signal_power"] = signal_power
        recipe[f"{split}_trial_scale"] = scale
        recipe[f"{split}_actual_snr_db"] = actual_snr
        if selected is not None:
            recipe[f"{split}_source_record_ids"] = selected
            assert np.isin(selected, pool).all()
        statistics[split] = {"actual_snr_min_db": float(actual_snr.min()),
                             "actual_snr_max_db": float(actual_snr.max()),
                             "actual_snr_mean_db": float(actual_snr.mean())}
    # The original Gaussian recipe stored None as an object scalar. Store an
    # explicit -1 sentinel in the portable archive so it never needs pickle.
    if recipe["source_split_seed"] is None:
        recipe["source_split_seed"] = -1
    with (folder / "noise_recipe.npz").open("xb") as handle:
        np.savez_compressed(handle, **recipe)
    job = {"format": FORMAT_ID, "task_id": task_id(kind, snr, subject),
           "noise_kind": kind, "snr_db": float(snr), "subject": subject,
           "base_seed": 7, "subject_seed": 7 + subject,
           "source_kind": "new_training", "resume": False,
           "statistics": statistics, "source_record_split": information,
           "noise_seeds": {"train": train_seed, "test": test_seed}}
    write_json(folder / "job.json", job)
    artifacts = [folder / "noise_recipe.npz", folder / "job.json"]
    artifacts += [folder / "data/BCI-38" / f"A{subject:02d}{suffix}.mat" for suffix in ("T", "E")]
    return {"task_id": job["task_id"], "noise_kind": kind, "snr_db": float(snr), "subject": subject,
            "folder": folder.relative_to(output_root).as_posix(),
            "artifact_sha256": {p.relative_to(output_root).as_posix(): sha256(p) for p in artifacts}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path, help="Directory containing BCI-38/A01T.mat etc.")
    parser.add_argument("--emg", required=True, type=Path, help="EEGdenoiseNet EMG_all_epochs_512hz.npy")
    parser.add_argument("--eog", required=True, type=Path, help="EEGdenoiseNet EOG_all_epochs.npy")
    parser.add_argument("--eog-mat", required=True, type=Path, help="Matching EOG_all_epochs.mat including fs")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "configs/clean_2a.json")
    parser.add_argument("--dry-run", action="store_true", help="Check sources and configuration, list tasks; write nothing")
    args = parser.parse_args()
    config = validate_config(read_json(args.config))
    clean_hashes = verify_clean_inputs(args.data_root)
    banks = load_banks(args.emg, args.eog, args.eog_mat)
    if args.dry_run:
        for kind, snr, subject in task_specs():
            print(task_id(kind, snr, subject))
        print("Source checks passed. 7 conditions x 9 subjects = 63 fresh models; no training executed.")
        return
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory must be absent or empty; preparation never overwrites existing data.")
    output.mkdir(parents=True, exist_ok=True)
    tasks = []
    for kind, snr, subject in task_specs():
        if (output / "STOP_AFTER_CHECKPOINT").exists():
            raise RuntimeError("Stop marker present; no further task will be prepared.")
        tasks.append(prepare_task(output, args.data_root, kind, snr, subject, banks))
        print(f"Prepared {len(tasks)}/63: {tasks[-1]['task_id']}", flush=True)
    if verify_clean_inputs(args.data_root) != clean_hashes:
        raise RuntimeError("Clean input changed during preparation")
    write_json(output / "manifest.json", {
        "format": FORMAT_ID, "dataset": "2a", "training_config": config,
        "source_files": source_manifest()["files"], "clean_2a_mat_sha256": clean_hashes,
        "protocol": source_manifest()["protocol"], "tasks": tasks,
        "models": 63, "reused_results": 0,
        "software": {"numpy": np.__version__, "scipy": scipy.__version__},
        "shared_condition": "EOG 20 dB is one trained condition used in both comparisons."})
    print("Preparation finished. Start training explicitly with run_noise.py.")


if __name__ == "__main__":
    main()
