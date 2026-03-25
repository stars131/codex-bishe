# Multi-Institution Dataset Notes

The repository can merge multiple raw CSV sources during `main.py --mode preprocess`,
but only when those sources share the same CIC-style tabular schema.

What is supported now:

- `CIC-IDS2017`
- Other CIC-family exports that keep the same columns and label field

What is not supported as a drop-in merge:

- `UNSW-NB15`
- `KDD Cup 99`
- `CTU-13`
- `UGR'16`
- threat-intel directories such as `abuse_ch`, `MITRE ATT&CK`, or Suricata rule feeds

Why:

- The current preprocessing path in [`main.py`](D:/毕设相关/测试数据集/huanjing/main.py) concatenates
  raw CSV frames directly before feature engineering.
- The current CIC splitter in [`src/data/dataloader.py`](D:/毕设相关/测试数据集/huanjing/src/data/dataloader.py)
  only understands predefined CIC feature groups.
- The current model in [`src/models/fusion_net.py`](D:/毕设相关/测试数据集/huanjing/src/models/fusion_net.py)
  is still a two-branch model. Extra sources are merged into branch 2 rather than
  being encoded independently.

Recommended practice:

1. Use one dataset family per run unless you have already built a schema-mapping step.
2. Keep `institutional_sources` disabled by default in server configs.
3. If you want cross-dataset fusion, add an explicit conversion pipeline first.
