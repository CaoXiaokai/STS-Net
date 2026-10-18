# 噪声实验运行

在工程根目录运行。全部命令只使用本目录代码和显式传入的数据路径，不读取旧实验目录或旧模型。

## 1. 准备固定噪声数据

```bash
python scripts/prepare_noise.py --data-root data --emg artifacts/EMG_all_epochs_512hz.npy --eog artifacts/EOG_all_epochs.npy --eog-mat artifacts/EOG_all_epochs.mat --output-dir prepared_noise --dry-run
```

去掉`--dry-run`执行数据准备。需要一个不存在或为空的输出目录；生成所有7个条件、9人的T/E数据。此步不训练。

```bash
python scripts/prepare_noise.py --data-root data --emg artifacts/EMG_all_epochs_512hz.npy --eog artifacts/EOG_all_epochs.npy --eog-mat artifacts/EOG_all_epochs.mat --output-dir prepared_noise
```

相同文件名但哈希不同会报错。MAT文件保存头可能带新时间，但重建数据数组及标签必须一致；噪声配方记录源ID、种子、缩放与实际SNR。

## 2. 检查任务与训练

```bash
python scripts/run_noise.py --run-dir prepared_noise --dry-run
python scripts/run_noise.py --run-dir prepared_noise
```

共63个模型，串行执行。保留正式实验的CUDA AMP、8个CPU线程、基础seed7、每位被试2000轮、原生评分方式。EOG20 dB是共享条件，不重复训练。

已完成任务只有在本次目录中的记录、配置、源码、输入和产物哈希检查通过后才跳过。不读取历史批次的模型或成绩。中断任务不自动resume，也不按成绩重新挑选训练。

创建`prepared_noise/STOP_AFTER_CHECKPOINT`可要求在保存后停止。保留停止标记时不会继续训练。若进程被强制终止，`.training.lock`可能遗留；应先确认原进程已退出，再处理锁文件。不要同时启动两份队列。

## 3. 输出

每个任务的目录含数据、噪声配方、配置、初始化记录、2000轮日志、best/latest模型及验收记录。全部完成后生成：

- `subject_results.csv`：63条逐人指标。
- `condition_results.csv`：7个条件九人等权平均。
- `results.json`：同批完整汇总。

只重建汇总、不训练：

```bash
python scripts/run_noise.py --run-dir prepared_noise --aggregate-only
```

使用本次结果绘图及导表：

```bash
python scripts/plot_noise_figures.py --results-csv prepared_noise/subject_results.csv --comparison-csv reference_results/eog_snr_accuracy_data.csv --output-dir plots_new
python scripts/export_noise_table.py --results-csv prepared_noise/subject_results.csv --output-dir plots_new
```

七方法图的数据由`--comparison-csv`独立提供；该文件里的STS-Net列也属于原服务器对比表。若要比较新一轮结果，应明确更新对应CSV，脚本不会静默替换其中某一条曲线。
