# 数据输入

## EEG数据

使用原包提供的预处理数据，目录结构为：

```text
data/BCI-38/A01T.mat ... A09T.mat
data/BCI-38/A01E.mat ... A09E.mat
data/BCI-38/B01T.mat ... B09T.mat
data/BCI-38/B01E.mat ... B09E.mat
```

`T`为训练session，`E`为测试session。MAT包含`data`和`label`。`data`排列为时间×通道×试次，`label`从1开始。2A为22通道4类，2B为3通道2类；每个试次1000个采样点。

`--data-root`指向包含`BCI-38`的父目录。程序保留原数据读取、训练统计量通道标准化和S&R增强，不需要把数据复制进工程。

本包不附预处理MAT，不附GDF预处理器；使用原有BCI-38数据即可。论文噪声入口按已有正式实验冻结哈希核对输入，以避免同名但内容不同的数据混入。

## EMG / EOG来源

噪声数据来自原实验使用的EEGdenoiseNet记录。所需文件为：

- `EMG_all_epochs_512hz.npy`：5598×1024，512 Hz。
- `EOG_all_epochs.npy`：3400×512，256 Hz。
- `EOG_all_epochs.mat`：与NPY对应的原MAT版本，用于核对采样信息及内容。

来源地址与SHA256保存在`configs/noise_sources.json`。这里仅记录原实验所用来源，不自动联网下载或替换文件。

噪声构造的源记录划分、随机种子、去重、重采样、滤波、时间截取和SNR缩放均保留原实现。Gaussian噪声本地生成，不需要外部波形文件。
