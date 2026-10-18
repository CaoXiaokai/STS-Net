# STS-Net: Code for the Experiments Included in the Manuscript

[中文](README_CN.md) | [English](README.md)

This directory contains the available code corresponding to the experiments included in the final ESWA manuscript. The original model, training, and data processing logic is preserved. Model branches and standalone experiment entry points not included in the manuscript have been removed, and portable entry points have been added for running experiments, preparing data, and generating figures. No models were retrained, and neither the manuscript nor historical results were modified during this code organization.

## Included Experiments

- Tables 1 and 2: subject-specific STS-Net training and accuracy and Kappa summaries for BCI Competition IV datasets 2A and 2B.
- Table 4: training and testing with EMG, EOG, and Gaussian noise at 20 dB for all nine subjects in dataset 2A, together with table export.
- Figure 5: accuracy curves for the nine subjects and their Average under EOG artifacts at 10, 12, 15, 20, and 25 dB.
- Figure 4: the EOG accuracy comparison plot for seven methods. The six baseline methods' values come from server results supplied by the user; their training implementations are not included in this directory.

The three noise types at 20 dB and the five EOG SNR levels constitute seven distinct conditions. EOG at 20 dB is trained only once per subject, giving 63 subject-condition models in total. Noise is added to both the training and test data in every noise condition.

The original package did not provide original experiment scripts that could be confirmed to correspond to the ablations in Figure 2, the t-SNE visualization in Figure 3, the L parameter experiment in Table 3, or the six baseline methods. See [docs/EXPERIMENT_SCOPE.md](docs/EXPERIMENT_SCOPE.md) for the missing materials and differences between the manuscript and the available implementation. This package does not substitute other experiments for these missing implementations.

## Directory Structure

```text
code/train.py                 Original training, evaluation, and aggregation functions
code/stsnet/                  Main model, data loading, S&R augmentation, and metrics
configs/clean_2a.json          Fixed configuration for all nine subjects in dataset 2A
configs/clean_2b.json          Original package configuration for dataset 2B
configs/noise_panel.json       Noise conditions included in the manuscript
configs/noise_sources.json     Noise sources, input hashes, and protocol
scripts/run_clean.py           Entry point for clean-data experiments
scripts/prepare_noise.py       Prepare the fixed noisy datasets used in the manuscript
scripts/run_noise.py           Entry point for the 63 noise experiment tasks
scripts/noise_common.py        Original noise generation functions
scripts/plot_noise_figures.py  Generate only Figures 4 and 5
scripts/export_noise_table.py  Export Table 4 as CSV and LaTeX
reference_results/            Existing noise results and server comparison data for plotting checks
docs/                         Experiment mapping, data formats, and implementation notes
provenance/                   Source hashes and code organization checks
```

## Installation

Python 3.11 or 3.12 is recommended. Install an appropriate PyTorch build for your machine first, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

Existing datasets, model weights, virtual environments, and training logs are not bundled with the code. Data locations can be supplied through command-line arguments without editing the source. See [docs/DATA.md](docs/DATA.md) for the required preprocessed MAT files and noise source files.

## Clean-Data Experiments: 2A and 2B

Run the following commands from this directory. The `data` directory must contain a `BCI-38` subdirectory; an absolute path to your data directory can also be used.

Use the `scripts` entry points below to load the fixed configurations. `code/train.py` retains the original training functions and low-level parameter interface; running it with its default arguments does not reproduce the manuscript configuration.

Inspect the configuration without training:

```bash
python scripts/run_clean.py --dataset 2a --data-root data --output-dir runs/clean_2a --dry-run
```

Run training:

```bash
python scripts/run_clean.py --dataset 2a --data-root data --output-dir runs/clean_2a
python scripts/run_clean.py --dataset 2b --data-root data --output-dir runs/clean_2b
```

Each subject is trained independently for 2,000 epochs. The base seed is 7, and the original training function uses `7 + subject`. Dataset 2A retains a batch size of 72 and an S&R augmentation batch size of 72. Dataset 2B retains the batch size of 100 and S&R augmentation batch size of 50 recorded in the original package. The output directory must be new: existing models are not overwritten, and no existing checkpoint is loaded for initialization. Metrics are averaged equally across the nine subjects without rounding the individual values before aggregation.

## Noise Experiments: Data Preparation and Training

First prepare the fixed noisy datasets for all seven conditions. The `artifacts` directory must contain the three original noise files specified below:

```bash
python scripts/prepare_noise.py --data-root data --emg artifacts/EMG_all_epochs_512hz.npy --eog artifacts/EOG_all_epochs.npy --eog-mat artifacts/EOG_all_epochs.mat --output-dir prepared_noise --dry-run
```

After the checks pass, remove `--dry-run` to generate the data. Data preparation and model training are separate steps; the preparation command does not train a model. Then run the training commands in [docs/NOISE_RUN.md](docs/NOISE_RUN.md).

The original processing pipeline is preserved: duplicate source records are removed before separating training and test noise sources; noise is resampled or generated at 250 Hz and filtered to 4–38 Hz; its amplitude is scaled to the target SNR for each trial. After fixed noise is added to the training (T) and test (E) data, channel normalization uses statistics from the training data only. The same EOG waveforms are used across SNR levels, with only their amplitude changed. Training calls the original `train_one_subject` function without replacing the training loop.

## Generate the Manuscript Figures and Export Table 4

The following commands use the bundled existing results and do not start training:

```bash
python scripts/plot_noise_figures.py --results-csv reference_results/subject_results.csv --comparison-csv reference_results/eog_snr_accuracy_data.csv --output-dir plots
python scripts/export_noise_table.py --results-csv reference_results/subject_results.csv --output-dir plots
```

The plotting script produces only the seven-method accuracy comparison and the accuracy curves for the nine subjects and their Average, each in PDF and PNG formats. Both figures use a descending horizontal axis from 25 to 10 dB. Table 4 reports average accuracy and average Kappa for the three noise types, without standard deviation or final-epoch accuracy columns. To use newly trained results, change `--results-csv` to the subject-level results exported by that run.

Reference CSV files are stored separately from new run outputs. Plotting does not insert reference scores into newly trained results. Figures use Times New Roman; if it is unavailable, the script issues a warning and falls back to DejaVu Serif.

## Verification

The training functions are unchanged. The data loading, normalization, S&R augmentation, and metric files retain their original contents. In small-batch forward checks for datasets 2A and 2B on CPU and GPU in training and evaluation modes, the original and streamlined models had identical initialization, outputs, batch normalization states, and random number generator states. Detailed records are provided in `provenance`. These checks validate the code organization; they are not additional manuscript experiments.

The package excludes the alpha/beta threshold experiments, separate CA on/off experiments and their repeat run, input-gradient heatmaps, overfitting diagnostics, the earlier Top-L rewrite, the additional peak-E pipeline, validation-set searches, inference-only noise experiments, the control without noise band-pass filtering, unused A01 exploratory experiments, and Kappa curves.

## Code Authors

- [CaoXiaokai](https://github.com/CaoXiaokai)
- [yanzeyuan](https://github.com/yanzeyuan)
