# MVUNet 第二次修复：依据 pathline_MVUNet_fixed.log

后续 v2 日志的诊断和当前运行方法见 [第三次调整](mvunet_generalization_v3.md)。下文保留第二次修复的背景。

## 日志说明了什么

这次训练的输入布局、batch_size、归一化和关闭噪声等设置均已生效。

| Epoch | Train BCE | Validation BCE | Validation IoU | Dice |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.722969 | 0.686111 | 0.397884 | 0.569266 |
| 5 | 0.269965 | **0.310033** | 0.711761 | 0.831613 |
| 10 | 0.087469 | 0.357849 | **0.744872** | **0.853784** |
| 25 | 0.018379 | 0.649916 | 0.737503 | 0.848923 |

训练能够拟合，且验证分割明显改善。剩下的是较大的训练/验证差距以及后期验证 BCE 上升。IoU 较稳定、BCE 却恶化，与模型对错误像素越来越自信的情况相符，但仅凭汇总日志不能把它与数据分布差异、BatchNorm 统计差异完全区分。

前景比例为 37.1692%，不能套用先前那份本地候选数据约 4% 的统计。

原策略按验证 BCE 保存模型，在第 5 轮后不再更新最佳检查点，第 25 轮触发正常早停。最终重载的是第 5 轮检查点，而非 IoU 最高的第 10 轮。该日志对应的保存策略没有留下第 10 轮或第 25 轮检查点，不能只凭日志恢复它们的权重。

## 报错根因及修复

异常发生在早停、重载检查点、载入测试数据之后：

```text
NameError: name 'TestGridSegmentation' is not defined
```

服务器的 `train_mvunet.py` 和配置已更新，`test.py` 却仍是没有这个类的旧版本。训练本身没有因该异常中断；中断的是训练后的测试阶段，已有最佳检查点仍然可用。

现在 MVUNet 直接调用 `test_mvunet.evaluate_trained_model()`，不经过 `test.py` 的字符串任务解析。训练入口在开始加载数据前导入专用评估函数；如果同步时遗漏依赖，会立即报告，而不是训练结束后才失败。新 `train_mvunet.py` 是 `train.py` 的薄入口，避免复制两份训练实现后各自漂移。

测试结果保存为运行目录下 `test/metrics.json`，并输出全像素 BCE、IoU、Dice、召回率和可视化 PNG。异常日志会区分 setup、training 和 final evaluation 阶段。

## 新训练策略

1. `checkpoint_monitor: val_iou`：以验证 IoU 选择最终模型和早停；IoU 完全相同时用更低的 BCE 打破平局。
2. 同时保存：
   - `best_checkpoint.pth.tar`：最佳验证 IoU。
   - `best_loss_checkpoint.pth.tar`：最低验证 BCE。
   - `last_checkpoint.pth.tar`：最后一轮，避免早停后没有近期权重。
3. 学习率改为验证驱动的 plateau：保留 lr=1e-4 和 5 轮 warmup，验证 BCE 停滞后按 factor=0.5 降低，patience=3，最低 lr=1e-6。当前日志中第 25 轮仍使用约 8.66e-5；新策略可更早降低学习率，而不必等待 100 轮 cosine 的后半段。
4. `SmoothedBCELoss(label_smoothing=0.02)` 只在训练时把标签 0/1 变为 0.01/0.99，抑制过度自信；进入 eval 模式后使用原始标签的普通 BCE。标签文件、IoU/Dice 和输出概率的接口不变。该调整是针对日志中过拟合迹象的训练实验，不代表已经验证全量泛化改善。
5. 每轮增加固定 512 个干净训练样本的 `Train(eval)` 指标，使用与验证集相同的 eval 模式和预处理，不更新 BatchNorm 统计。比较其原始 BCE 与 `Validation` BCE，才能更准确地判断训练/验证差距。
6. 保留 features=32、dropout=0、无输入噪声、AdamW weight_decay=1e-4、batch_size=64；不同时改变网络结构或重新划分数据。

注意：训练行中的 SmoothedBCELoss 与过去的原始训练 BCE 不直接可比。比较原始 BCE 时看 `Train(eval)` 与 `Validation` 行。标签平滑后的训练目标也不应被要求降至零。

每个新运行目录保存 `resolved_config.yaml`，包含实际解析后的模型与预处理参数，后续评估应优先使用该配置。

## 同步与运行

当前服务器没有本地这次新增的所有定义，因此不能只复制 YAML 或 train_mvunet.py。相关文件完整清单：

- `train_mvunet.py`
- `train.py`
- `test_mvunet.py`
- `config/segmentation/mvu_net.yaml`
- `DeepUtils/loss/build.py`
- `DeepUtils/scheduler/plateau_lr.py`
- `DeepUtils/MiscFunctions.py`
- `DeepUtils/dataset/SteadyVastisDataset.py`
- `DeepUtils/dataset/transforms/basic_transform.py`

这些文件连同回归脚本、测试辅助文件和本文已打包在 `tmp/mvunet_update_v2.zip`，包内有 SHA256SUMS。服务器若有额外修改，应合并相应改动，尤其保留服务器自己需要的其他模型和导出功能。

同步后先在训练环境验证：

```sh
python validate_mvunet_training.py
```

从头运行新的训练策略：

```sh
nohup python -u train_mvunet.py \
  --config config/segmentation/mvu_net.yaml \
  --data_dir /home/featurize/data/vatistas_mvunet_trainable/ \
  > pathline_MVUNet_v2.log 2>&1 &
```

输出位于 `outputModels/MVUNet_calibrated/<run_name>/`。

本次失败运行的第 5 轮检查点仍可单独测试，无须为了补测试重新训练。其网络形状和输入布局与当前配置兼容：

```sh
python test_mvunet.py \
  --config config/segmentation/mvu_net.yaml \
  --data_dir /home/featurize/data/vatistas_mvunet_trainable/ \
  --model_path outputModels/MVUNet_aligned/bs_64_ep_100_lr_0.0001_20260908_115257_seed_4259/best_checkpoint.pth.tar \
  --save_samples 10 \
  --report testOutput/mvunet_fixed_existing_checkpoint.json
```

上面的命令针对此次已经对齐输入布局的检查点，不加 `--legacy_xy_layout`。

## 已完成的验证及限制

- 9 个回归测试通过，包括：旧 test 模块缺少 TestGridSegmentation 时专用评估仍能工作；IoU 提升而 BCE 变差时保存正确检查点；plateau 收到验证指标并降低学习率；标签平滑仅在训练生效且不修改原标签；以及前一轮的数据对齐、噪声和梯度累积检查。
- 本机实际训练环境完成三轮小数据完整流程：训练、普通 BCE 验证、干净训练子集 eval、三类检查点保存、最佳模型重载、专用全像素测试及 PNG/JSON 输出。
- 语法与 diff 检查通过。

没有远端全量数据或此次运行的新检查点，因此无法在本机测定新策略在 48000 个样本上的最终收敛和泛化。若新的 `Train(eval)` 很好而 Validation 仍明显较差，应检查两份数据的分布和标签；若 Train(eval) 也明显差于训练时表现，则应进一步检查 BatchNorm 统计。日志将提供这项区分所需的证据。
