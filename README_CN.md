# STS-Net：论文实验代码整理

[中文](README_CN.md) | [English](README.md)

本目录按最终 ESWA 稿件中的实验范围整理。原始模型、训练和数据处理逻辑保持原实现；删除未入文的模型分支与独立实验入口，并增加独立目录下的运行、数据准备和绘图入口。没有重新训练模型，也没有修改论文或历史结果。

## 包含内容

- 表1、表2：STS-Net 在 BCI Competition IV 2A / 2B 的逐被试训练与准确率、Kappa 汇总。
- 表4：EMG、EOG、Gaussian 三种噪声在 20 dB 下、2A 九位被试的训练与测试，以及表格导出。
- 图5：EOG 在 10、12、15、20、25 dB 下、九位被试和 Average 的准确率曲线。
- 图4：七种方法的 EOG 准确率对比绘图。六个 baseline 的数值来自用户提供的服务器结果；本目录不包含这些方法的训练实现。

三噪声20 dB与五档EOG实验合计7个不同条件，EOG 20 dB只训练一次，共63个被试—条件模型。所有噪声条件在训练和测试数据中均加入噪声。

原包没有提供能够确认对应论文图2消融、图3 t-SNE、表3 L参数以及六个baseline的原始实验脚本。具体待补内容与现有实现差异见 [docs/EXPERIMENT_SCOPE.md](docs/EXPERIMENT_SCOPE.md)。本包不以其他试验替代这些缺失实现。

## 目录

```text
code/train.py                 原训练、评分、汇总函数
code/stsnet/                  主方法模型、数据加载、S&R和指标
configs/clean_2a.json          九位被试2A固定配置
configs/clean_2b.json          原包2B固定配置
configs/noise_panel.json       论文噪声条件清单
configs/noise_sources.json     噪声来源、输入哈希及协议
scripts/run_clean.py           干净数据实验入口
scripts/prepare_noise.py       一次性生成论文噪声数据
scripts/run_noise.py           63个噪声任务入口
scripts/noise_common.py        原噪声生成函数
scripts/plot_noise_figures.py  仅绘制论文图4、图5
scripts/export_noise_table.py  导出论文表4（CSV和LaTeX）
reference_results/             已有噪声结果和服务器对比数据，仅供绘图核对
docs/                         实验对应、数据格式及实现说明
provenance/                   来源哈希与整理核验
```

## 安装

建议 Python 3.11 或 3.12。首先安装适合本机的 PyTorch，再安装其余依赖：

```bash
python -m pip install -r requirements.txt
```

已有数据、模型权重、虚拟环境和训练日志不随代码重复打包。读取数据的位置可通过命令行传入，无需修改源码。预处理MAT与噪声源文件的具体要求见 [docs/DATA.md](docs/DATA.md)。

## 干净数据：2A / 2B

以下命令均在本目录执行。`data` 应包含 `BCI-38` 子目录；也可以替换为数据所在的绝对路径。

请使用下方`scripts`入口加载固定配置。`code/train.py`保留原训练函数和底层参数接口，直接使用其默认参数不等于论文配置。

先查看配置，不训练：

```bash
python scripts/run_clean.py --dataset 2a --data-root data --output-dir runs/clean_2a --dry-run
```

执行训练：

```bash
python scripts/run_clean.py --dataset 2a --data-root data --output-dir runs/clean_2a
python scripts/run_clean.py --dataset 2b --data-root data --output-dir runs/clean_2b
```

每人独立训练2000轮，基础seed为7，原入口实际使用`7 + subject`。2A保留batch72、S&R72；2B保留原包实际记录的batch100、S&R50。输出目录必须是新目录，不覆盖旧模型，不读取旧checkpoint初始化。汇总使用九位被试未提前舍入的指标等权平均。

## 噪声实验：准备与训练

先准备全部7条件的固定噪声数据。`artifacts`中应有下面三个原始噪声文件：

```bash
python scripts/prepare_noise.py --data-root data --emg artifacts/EMG_all_epochs_512hz.npy --eog artifacts/EOG_all_epochs.npy --eog-mat artifacts/EOG_all_epochs.mat --output-dir prepared_noise --dry-run
```

检查通过后，去掉`--dry-run`生成数据。数据准备与模型训练是两个独立步骤；准备命令不会训练。随后按 [docs/NOISE_RUN.md](docs/NOISE_RUN.md) 中的训练命令执行。

原流程保留：源记录去重后分开训练/测试来源；噪声重采样或生成于250 Hz并经4–38 Hz滤波；逐试次按目标SNR缩放；固定噪声加到T/E后，仅按训练集统计量做通道标准化。EOG不同强度共享相同波形，只改变幅度。训练调用原`train_one_subject`，不替换训练循环。

## 绘制论文图片和导出表4

仅使用随包的已有结果绘图，不会启动训练：

```bash
python scripts/plot_noise_figures.py --results-csv reference_results/subject_results.csv --comparison-csv reference_results/eog_snr_accuracy_data.csv --output-dir plots
python scripts/export_noise_table.py --results-csv reference_results/subject_results.csv --output-dir plots
```

绘图输出仅含七方法准确率图、九位被试与Average准确率图，各为PDF和PNG；横轴均从25到10 dB。表4输出三类噪声的平均准确率和平均Kappa，不增加标准差列或末轮准确率列。使用新训练结果时，将`--results-csv`改为本次训练导出的逐被试结果。

参考CSV与新运行输出分开放置；绘图不会把参考成绩写进新训练结果。图片字体使用Times New Roman，缺失时会明确提示并回退到DejaVu Serif。

## 整理核验

训练函数未改；数据读取、归一化、S&R及指标文件保留原内容。原模型和精简模型在2A/2B、CPU/GPU、train/eval的小批前向检查中，初始化、输出、BN状态和随机状态一致。具体记录见`provenance`。这些是代码整理检查，不是新的论文实验。

未收录α/β阈值试验、独立CA开关与复跑、输入梯度热力图、过拟合诊断、旧Top-L改写、额外peak-E流程、验证集搜索、仅推理加噪、取消噪声带通的对照、未采用的A01探索及κ曲线。

## 代码作者

- [CaoXiaokai](https://github.com/CaoXiaokai)
- [yanzeyuan](https://github.com/yanzeyuan)
