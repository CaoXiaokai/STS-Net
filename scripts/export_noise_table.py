"""Export the manuscript's 20 dB three-noise table from subject-level results."""
import argparse
import csv
import math
from pathlib import Path
from statistics import mean


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-csv', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    opts = p.parse_args()
    with opts.results_csv.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    summaries = []
    for kind, label in [('emg', 'EMG'), ('eog', 'EOG'), ('gaussian', 'Gaussian noise')]:
        values = [r for r in rows if r['kind'] == kind and float(r['snr_db']) == 20]
        if len(values) != 9 or {r['subject'] for r in values} != {f'A{i:02d}' for i in range(1, 10)}:
            raise ValueError(f'{kind}: expected one result per subject A01-A09 at 20 dB')
        acc = [float(r['accuracy']) for r in values]
        kappa = [float(r['kappa']) for r in values]
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in acc):
            raise ValueError('Accuracy must be a finite fraction in [0, 1]')
        if not all(math.isfinite(x) and -1 <= x <= 1 for x in kappa):
            raise ValueError('Kappa must be finite and between -1 and 1')
        summaries.append((label, mean(acc) * 100, mean(kappa)))
    opts.output_dir.mkdir(parents=True, exist_ok=True)
    with (opts.output_dir / 'noise20_table.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['noise_type', 'average_accuracy_percent', 'average_kappa'])
        w.writerows(summaries)
    body = '\n'.join(f'{label} & {acc:.2f} & {kappa:.4f} ' + r'\\' for label, acc, kappa in summaries)
    tex = (r'''\begin{table}[!htbp]
\centering
\caption{Classification performance of STS-Net under three noise types at an SNR of 20 dB.}
\label{tab:noise20}
\begin{tabular}{lcc}
\toprule
Noise type & accuracy (\%) & $\kappa$ \\
\midrule
''' + body + '\n' + r'''\bottomrule
\end{tabular}
\end{table}
''')
    (opts.output_dir / 'noise20_table.tex').write_text(tex, encoding='utf-8')
    for label, acc, kappa in summaries:
        print(f'{label}: accuracy={acc:.2f}%, kappa={kappa:.4f}')


if __name__ == '__main__':
    main()
