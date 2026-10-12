> 后续更新：针对 pathline_MVUNet_fixed.log 的报错和验证指标，见 [第二次修复说明](mvunet_fixed_followup.md)。本文保留第一次修复的诊断记录。

# MVUNet：根据 pathline_MVUNet_new.log 的诊断与修改

## 日志结论

此次训练在远端 `/home/featurize/work/PyflowVis` 运行，训练集 48000、验证集 6000、测试集 6000。训练 loss 持续下降，但验证 loss 在第 6 轮后整体恶化。

| Epoch | Train BCE | Validation BCE |
| --- | ---: | ---: |
| 1 | 0.675968 | 0.662368 |
| 6（验证最佳） | 0.448929 | 0.633435 |
| 20 | 0.287078 | 0.692619 |
| 50 | 0.216800 | 0.778547 |
| 100 | 0.189650 | 0.797456 |

日志中没有 NaN 或 OOM。它显示的是训练/验证差距扩大，不能仅归结为学习率不下降。远端的实际配置与此前检查的本地 YAML 不同：此次使用 cosine、5 轮 warmup，而且已经获得 minV/maxV；此前针对本地配置提出的 TwoStep 和缺少归一化参数，不是此次训练失败的证据。

## 主要问题及已修改内容

### 1. 空间轴错位

`SteadyVelocityGridSegmentationMVUnet` 原来将 `[Y,X,4]` 转为 `[4,X,Y]`，标签仍为 `[Y,X]`。现默认使用 `permute(2,0,1).contiguous()`，得到对齐的 `[4,Y,X]`。

添加了 `dataset.legacy_xy_layout: True` 开关，供评估旧模型时复现历史输入。新训练配置显式设为 False。非方形二进制样本的加载、特征位置和标签位置已通过回归测试。

这是本地源码中确定存在的问题；远端运行文件未随日志提供，仍需保证远端同步了这一修改。

### 2. 归一化后的噪声幅度使用了原始速度单位

日志实际使用：

```text
train: [MinMaxNormalization, WhiteNoise]
minV = -12.964859008789062
maxV = 12.285758972167969
noiseMaginitude = 0.02
```

按仓库 WhiteNoise 实现，噪声范围为 `[minV*0.02,maxV*0.02]`，即 `[-0.259297,0.245715]`，加在已经归一化的输入上。它不是通常期待的归一化空间 ±0.02。验证集不加噪，加上 dropout=0.15 和模型中的 BatchNorm，这些因素可能扩大训练/验证差异；仅凭日志不能分别证明它们的贡献。

新配置先关闭 WhiteNoise 和 dropout，建立无噪声基线。WhiteNoise 新增了显式参数 `noise_in_normalized_space: True`，将未来启用的噪声限制为 ±noiseMaginitude；旧配置默认保留原有语义。回归测试同时检查新旧模式和输入不会被原地修改。

### 3. batch_size 拼写和梯度累积

日志同时出现 `batchsize:64` 和 `batch_size:256`。程序读取后者，因此实际 batch 是 256，梯度累积是 2，完整累积组的有效 batch 是 512。

新配置明确设置 `batch_size:64`、`gradient_accumulate:1`，启动日志打印这两项和有效 batch。解析配置时若出现 `batchsize` 会直接报告拼写错误，避免再次静默回落到默认值。

原训练循环在一个 epoch 有奇数个批次且梯度累积为 2 时，不更新最后一批，而把梯度带到下一轮。日志每轮 187 个批次，满足这个触发条件。已修改为在最后一批完成更新，并按最后一组的实际批次数归一化；已与显式合并批次的 SGD 更新进行数值对照测试。

### 4. 验证与测试指标

新配置的 train/val/test 均使用同一组元数据或显式指定的 minV/maxV 归一化；MVUNet 不再借用注入路径线参数的初始化函数。

每轮验证新增全像素 IoU、Dice、召回率，以及真实/预测前景占比；验证 BCE 按样本数加权，避免最后一个小批次权重过大，并恢复进入验证前的模型模式。

原 `TestSegmentation` 调用 `segmentationCriteria` 对 `[B,Y,X]` 数据取 `[...,0]`，只统计一列像素。新配置使用 `TestGridSegmentation`，沿用现有全元素混淆矩阵实现，汇总所有像素。日志的 `test: []` 还导致原始输入直接进入使用归一化输入训练的模型，因此末尾的 IoU=0.274847、F1=0.430585 不能作为可靠的完整测试成绩。

日志最后的 `save_segmentation_as_png is not defined` 发生在训练完成后的可视化阶段；已补齐导入。TorchScript 的 script→trace 回退后成功保存，并不是此次无法取得更好验证结果的直接证据。

### 5. 训练基线

新 `config/segmentation/mvu_net.yaml` 使用：

- 模型宽度 32，与日志一致；dropout=0。
- AdamW，lr=1e-4，weight_decay=1e-4，与日志一致。
- cosine，warmup=5，min_lr=1e-6，与日志一致。
- batch_size=64，gradient_accumulate=1。
- 每轮验证，连续 20 次验证无改善时提前停止。
- train/val/test 归一化一致，不添加输入噪声。
- 单独保存到 `outputModels/MVUNet_aligned/`。

修复了每轮末尾调度时的偏移：调用 `scheduler.step(epoch+1)`，不再让第 1、2 轮都使用 warmup 的初始学习率。每轮开始显式进入 train 模式并清空梯度。训练异常现在输出完整堆栈并以非零状态退出。

## 运行

在包含实际数据和 GPU 的训练环境同步修改后，从项目根目录重新训练：

```sh
nohup python -u train.py \
  --config config/segmentation/mvu_net.yaml \
  --data_dir /home/featurize/data/vatistas_mvunet_trainable/ \
  > pathline_MVUNet_fixed.log 2>&1 &
```

修改输入布局后从头训练，避免沿用不匹配的输入约定。首先检查启动日志 `batch_size=64, gradient_accumulate=1, effective_batch_size=64`，以及每轮 `Validation pixels` 指标。不能只看 BCE 或像素 accuracy。

测试新检查点：

```sh
python test_mvunet.py \
  --config config/segmentation/mvu_net.yaml \
  --data_dir /home/featurize/data/vatistas_mvunet_trainable/ \
  --model_path outputModels/MVUNet_aligned/<run_name>/best_checkpoint.pth.tar
```

若评估旧的转置输入检查点，使用它原来的模型配置，并额外传入 `--legacy_xy_layout`。现有 `mvunet_comparison.py` 仍服务于历史检查点的预处理/输出约定；不要直接把新检查点接入该旧适配器。本次新检查点通过 `test_mvunet.py` 或通用 `test.py` 按对齐布局评估。

远端日志包含本地 train.py 中没有的 TorchScript 导出逻辑。同步修改时应保留所需的远端导出功能；导出若临时调用 eval，应保存并恢复原模式，或确保下一轮调用 train。这里无法验证未提供的远端导出实现。

## 验证范围

在本机 conda `vortex` 环境中完成：

1. `python validate_mvunet_training.py`：6 个回归测试通过，涵盖矩形数据对齐与旧模式、三种 split 的预处理、归一化噪声、全像素指标与末尾小批次、梯度累积尾批次、配置模型的有限且非零梯度。
2. 使用本地真实二进制文件建立互不重叠的 8/2/2 样本临时数据集，以较窄模型运行完整 train.py 两轮：训练、验证、保存与重新载入最佳检查点、全像素测试、PNG 输出均成功。
3. 独立 test_mvunet.py 成功载入上述检查点，评估全部 8192 个测试像素，其混淆矩阵和通用 TestGridSegmentation 一致。
4. batchsize 拼写拦截、语法编译和 git diff --check 检查通过。

小数据流程只证明代码连通和回归修复，不能证明远端 48000 个训练样本上的完整收敛或泛化。远端数据分布以及各问题的单独影响仍需要新训练日志确认。如果对齐、无噪声的基线仍出现持续扩大的训练/验证差距，下一步比较两份 split 的标签占比和每通道分布，并在相同 eval 模式下评估干净训练子集，以区分数据分布差异和 BatchNorm 统计差异。
