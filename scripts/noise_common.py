"""Noise construction used by the paper's seven-condition 2A panel.

The four waveform functions below are copied without mathematical changes from
its verified preparation implementation. No source data or old run is imported.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import signal

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = (("emg",20.0),("eog",20.0),("gaussian",20.0),
              ("eog",10.0),("eog",12.0),("eog",15.0),("eog",25.0))
SUBJECTS = tuple(range(1,10))
SPLIT_SEEDS = {"emg":2026100510,"eog":2026100530}
NOISE_SEEDS = {"emg":(2026100511,2026100512),"eog":(2026100531,2026100532),
               "gaussian":(2026100541,2026100542)}
FORMAT_ID = "stsnet_paper_noise_v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    temp.replace(path)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()


def task_id(kind,snr,subject):
    return f"{kind}-{int(snr)}dB-A{subject:02d}"


def task_folder(root,kind,snr,subject):
    return Path(root)/"conditions"/f"{kind}_{int(snr)}dB"/f"A{subject:02d}"


def task_specs():
    return [(kind,snr,subject) for kind,snr in CONDITIONS for subject in SUBJECTS]


def validate_config(config):
    expected=read_json(PACKAGE_ROOT/"configs/noise_sources.json")["training_parameters"]
    for key,value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Paper noise configuration differs at {key}: {config.get(key)!r} != {value!r}")
    return config


def source_manifest():
    return read_json(PACKAGE_ROOT/"configs/noise_sources.json")


def ensure_inside(root,relative):
    root=Path(root).resolve()
    path=(root/relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path leaves the requested output directory: {relative}")
    return path


def waveform_hash(row):
    canonical = np.array(row, dtype="<f8", order="C", copy=True)
    canonical[canonical == 0] = 0.0
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def make_bank(source, fs):
    """Union exact raw OR standardized/resampled duplicates, retain lowest source ID."""
    assert source.ndim == 2 and source.shape[1] == 2*fs and source.dtype == np.float64
    assert np.isfinite(source).all()
    sd = source.std(axis=1, keepdims=True)
    assert np.all(sd > 0)
    standardized = (source-source.mean(axis=1, keepdims=True))/sd
    bank = signal.resample_poly(standardized, 125, fs//2, axis=1)
    assert bank.shape == (len(source), 500) and np.isfinite(bank).all()
    raw_hashes = np.array([waveform_hash(row) for row in source], dtype="U64")
    processed_hashes = np.array([waveform_hash(row) for row in bank], dtype="U64")
    parent = np.arange(len(source), dtype=np.int32)

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    seen_raw, seen_processed = {}, {}
    for index, (raw_key, processed_key) in enumerate(zip(raw_hashes, processed_hashes)):
        for seen, key in ((seen_raw, raw_key), (seen_processed, processed_key)):
            if key in seen:
                left, right = find(index), find(seen[key])
                parent[max(left, right)] = min(left, right)
            else:
                seen[key] = index
    mapping = np.array([find(index) for index in range(len(source))], dtype=np.int32)
    unique = np.unique(mapping)
    assert all(rep == np.flatnonzero(mapping == rep).min() for rep in unique)
    assert len(set(raw_hashes[unique])) == len(unique)
    assert len(set(processed_hashes[unique])) == len(unique)
    arrays = {"raw_record_sha256": raw_hashes, "processed_record_sha256": processed_hashes,
              "source_record_to_canonical_id": mapping, "unique_record_ids": unique}
    information = {"source_records": len(source), "raw_unique_waveforms": len(set(raw_hashes)),
                   "processed_unique_waveforms": len(set(processed_hashes)), "retained_canonical_records": len(unique),
                   "removed_duplicate_records": len(source)-len(unique),
                   "duplicate_group_count": int(sum(np.sum(mapping == rep) > 1 for rep in unique)),
                   "rule": "Union records with identical raw float64 bytes OR identical explicitly standardized and 250 Hz resampled float64 bytes; normalize signed zero before hashing; retain smallest original record ID",
                   "hash_format": "SHA256 of C-contiguous little-endian float64 row bytes, +0 and -0 both written as +0",
                   "limitations": "Exact duplicates only, not approximate duplicates or source-person/session identities"}
    return bank, arrays, information


def split_bank(arrays, seed):
    unique = arrays["unique_record_ids"]
    shuffled = np.random.default_rng(seed).permutation(unique).astype(np.int32)
    split = int(np.floor(0.8*len(unique)))
    train, test = shuffled[:split], shuffled[split:]
    assert min(len(train), len(test)) > 0 and not np.intersect1d(train, test).size
    assert np.array_equal(np.sort(np.concatenate((train, test))), unique)
    mapping = arrays["source_record_to_canonical_id"]
    # Check every original group member, not only retained representatives.
    train_members, test_members = np.isin(mapping, train), np.isin(mapping, test)
    assert np.all(train_members ^ test_members)
    for field in ("raw_record_sha256", "processed_record_sha256"):
        assert not set(arrays[field][train_members]) & set(arrays[field][test_members]), field
    return train, test


def build_noise(kind, bank, pool, seed, sos):
    generator = np.random.default_rng(seed)
    selected = None
    if kind == "gaussian":
        traces = generator.standard_normal((288, 22, 1450), dtype=np.float64)
    else:
        offsets = generator.integers(0, len(pool), size=(288,22,3), dtype=np.int32)
        selected = pool[offsets]
        ramp = np.linspace(0.0, 1.0, 25, dtype=np.float64)
        traces = bank[selected[..., 0]].copy()
        for index in (1, 2):
            segment = bank[selected[..., index]]
            cross = traces[..., -25:]*(1-ramp) + segment[..., :25]*ramp
            traces = np.concatenate((traces[..., :-25], cross, segment[..., 25:]), axis=-1)
    assert traces.shape == (288,22,1450)
    noise = signal.sosfiltfilt(sos, traces, axis=-1)[..., 225:1225]
    noise -= noise.mean(axis=-1, keepdims=True)
    rms = np.sqrt(np.mean(noise**2, axis=(1,2), keepdims=True))
    assert np.isfinite(noise).all() and np.all(rms > 0)
    return (noise/rms).astype(np.float32), selected
