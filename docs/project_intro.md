# Project Overview

This project is a PyTorch-based network attack classification system built around
multi-source feature fusion.

## What the code supports now

- CIC-IDS-2017 full pipeline through [`main.py`](D:/毕设相关/测试数据集/huanjing/main.py)
- KDD Cup 99 quick pipeline through [`quick_test.py`](D:/毕设相关/测试数据集/huanjing/quick_test.py)
- Two-branch fusion models in [`src/models/fusion_net.py`](D:/毕设相关/测试数据集/huanjing/src/models/fusion_net.py)
- Encoder variants: `mlp`, `cnn`, `lstm`, `transformer`
- Fusion variants: `attention`, `multi_head`, `gated`, `cross`, `bilinear`, `concat`

## Current local data state

The repository currently contains processed KDD Cup 99 artifacts under
[`data/processed`](D:/毕设相关/测试数据集/huanjing/data/processed):

- classes: `dos`, `normal`, `probe`, `r2l`, `u2r`
- source1 dim: `22`
- source2 dim: `19`
- train samples: `13,369`
- val samples: `1,910`
- test samples: `3,820`

These numbers are the ones present in the checked-in `multi_source_data.pkl`.

## Important limitations

- The model is still a two-branch model. Extra sources are concatenated into branch 2.
- Threat-intel downloads exist in the repo, but threat-intel feature augmentation is not
  connected to preprocessing yet.
- Multi-institution preprocess only works for sources with the same raw CSV schema.
- Train and evaluate must use the same config, or checkpoint loading can fail.

## Recommended server path

For Linux/GPU training, use [`src/config/config_server.yaml`](D:/毕设相关/测试数据集/huanjing/src/config/config_server.yaml)
and [`server_train_guide.sh`](D:/毕设相关/测试数据集/huanjing/server_train_guide.sh).

Supported command flow:

```bash
python main.py --data_dir /path/to/CIC-IDS-2017 --mode preprocess --config src/config/config_server.yaml
python main.py --mode train --config src/config/config_server.yaml
python main.py --mode evaluate --experiment exp_YYYYMMDD_HHMMSS --config src/config/config_server.yaml
python main.py --mode report --experiment exp_YYYYMMDD_HHMMSS --config src/config/config_server.yaml
```
