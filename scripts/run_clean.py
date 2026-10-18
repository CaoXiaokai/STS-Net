"""Run only the supplied STS-Net clean 2A/2B configuration, one subject at a time."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['2a', '2b'], required=True)
    p.add_argument('--data-root', type=Path, required=True, help='Directory containing BCI-38/*.mat')
    p.add_argument('--output-dir', type=Path, required=True, help='A new directory; existing directories are rejected')
    p.add_argument('--device', default='', help='Original automatic CUDA/CPU selection when omitted')
    p.add_argument('--dry-run', action='store_true', help='Print the fixed configuration without training or creating output')
    opts = p.parse_args()
    config_path = ROOT / 'configs' / f'clean_{opts.dataset}.json'
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    cfg.update(data_root=str(opts.data_root.resolve()), run_dir=str(opts.output_dir.resolve()),
               output_dir=str(opts.output_dir.resolve().parent), device=opts.device)
    assert cfg['subjects'] == '1-9' and cfg['epochs'] == 2000 and not cfg['resume']
    assert cfg['stem'] == 'paper_exact' and cfg['transformer_style'] == 'paper'
    assert cfg['random_sparse'] and cfg['eval_train_mode'] and cfg['seed'] == 7
    if opts.dry_run:
        print(json.dumps({'subjects': list(range(1, 10)), 'models': 9, 'config': cfg}, indent=2))
        return
    for directory in (ROOT, opts.output_dir, opts.output_dir.parent):
        if (directory / 'STOP_AFTER_CHECKPOINT').exists():
            raise RuntimeError(f'Stop marker exists: {directory}')
    if opts.output_dir.exists():
        raise FileExistsError('Use a new output directory. This entry point never resumes or overwrites a run.')
    prefix = 'A' if opts.dataset == '2a' else 'B'
    inputs = [opts.data_root / cfg['band_dir'] / f'{prefix}{s:02d}{split}.mat'
              for s in range(1, 10) for split in ('T', 'E')]
    for f in inputs:
        if not f.is_file():
            raise FileNotFoundError(f)
    input_hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in inputs}
    sys.path.insert(0, str(ROOT / 'code'))
    import train
    args = argparse.Namespace(**cfg)
    output = opts.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    (output / 'input_hashes.json').write_text(json.dumps(input_hashes, indent=2), encoding='utf-8')
    summaries = []
    for subject in range(1, 10):
        summaries.append(train.train_one_subject(args, subject, output))
    summary = train.aggregate(summaries, output)
    print(json.dumps({'average_accuracy_percent': summary['mean_best_accuracy_percent'],
                      'average_kappa': summary['mean_best_kappa'], 'subjects': 9}, indent=2))


if __name__ == '__main__':
    main()
