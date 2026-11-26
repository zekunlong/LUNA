# LUNA
实现位于 `trainers/luna.py`，配置和脚本分别位于
`configs/trainers/LUNA` 与 `scripts/luna`。

## 环境

Python 3.9、PyTorch 2.0.1、torchvision 0.15.2 

```bash
conda activate luna
python -m pip install -r requirements.txt
```

`train.py` 和主要可视化入口会优先加载包内的 `Dassl.pytorch`。如需在
交互式 Python 中直接导入 Dassl，可执行：

```bash
python -m pip install -e Dassl.pytorch --no-deps
```

## 训练


```bash
CUDA_VISIBLE_DEVICES=0 sh scripts/luna/base2new_train.sh dtd 1
```

训练过程会在输出目录保存 `loss_trends.csv`，其中记录每个 epoch 的三个
实际蒸馏损失 `loss_kd`、`loss_bc` 和 `loss_cc`；脚本成功结束后会自动生成
PNG 曲线图。记录值不包含单独展示共享系数或配置系数。

跨数据集实验：

```bash
export LUNA_CROSS_DATASET_TEACHER_CHECKPOINT=/path/to/model.pth.tar-20
CUDA_VISIBLE_DEVICES=0 sh scripts/luna/xd_train.sh dtd 1
```

## 可视化

可视化工具默认从本包的 `output` 目录读取训练后权重，也支持各工具的
命令行参数覆盖。示例：

```bash
python tools/visualize_logits_tsne.py dtd --seed 1 --root "$LUNA_DATA_ROOT"
python tools/visualize_luna_attention.py dtd --seed 1 --root "$LUNA_DATA_ROOT"
```

查看完整参数：

```bash
python tools/visualize_logits_tsne.py --help
python tools/visualize_luna_attention.py --help
```
