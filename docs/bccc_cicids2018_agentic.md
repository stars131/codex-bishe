# BCCC-CSE-CIC-IDS-2018 Agentic Workflow

This repository now supports a direct workflow for the local dataset:

- dataset path: `data/数据集BCCC-CSE-CIC-IDS-2018`
- flow source: numeric traffic features from the original BCCC CSV files
- log source: derived context/log-like features from timestamp, ports, protocol, handshake state, IP properties
- threat-intel source: local IOC library queried through a mock HTTP API

## One-click demo

```bash
python run_bccc_agentic_demo.py
```

That command will:

1. read a small sample from the nested BCCC zip archives
2. build `flow_*` and `log_*` features
3. save a sampled multimodal CSV
4. build a local threat-intel library JSON
5. start a mock threat-intel API server
6. call the API for every sampled record
7. save a threat-intel sidecar CSV
8. preprocess
9. train
10. evaluate

## Output files

- sampled raw file: `data/processed/bccc_cicids2018_raw_sample.csv`
- multimodal training file: `data/processed/bccc_cicids2018_agentic_sample.csv`
- threat-intel library: `data/threat_intel/bccc_cicids2018_mock_library.json`
- threat-intel sidecar: `data/threat_intel/bccc_cicids2018_agentic_threat_intel.csv`
- runtime config: `outputs/generated_configs/bccc_cicids2018_agentic_runtime.yaml`

## Useful arguments

```bash
python run_bccc_agentic_demo.py \
  --sample-per-member 200 \
  --max-members 6 \
  --epochs 3 \
  --batch-size 64
```

You can also control the sampled archive groups:

```bash
python run_bccc_agentic_demo.py \
  --member-keywords benign,bf_ftp,bf_ssh,bot,sql_injection,infiltration
```

## Why there is a log branch even though BCCC is flow-centric

The raw BCCC-CSE-CIC-IDS-2018 files are flow-level CSVs, not host log files.  
To keep the thesis architecture consistent, this project derives a second branch of log-like contextual features from:

- timestamp
- protocol
- ports
- handshake state
- handshake completion markers
- IP structure and network context

That gives you a reproducible `flow + context/log-like` two-branch fusion setup on top of the BCCC dataset.

## Formal experiment mode

For thesis-ready experiments, use:

```bash
python run_bccc_formal_experiments.py --suite full
```

This runner will automatically produce:

- main experiment results
- ablation results
- fusion comparison results
- per-experiment HTML reports
- CSV summary tables
- LaTeX tables
- comparison figures
- a markdown summary report

Server wrapper:

```bash
bash server_formal_bccc_experiments.sh
```

## Threat-intel behavior

The default mock threat-intel library uses a heuristic IOC strategy instead of using the current sample labels directly.

That is intentional:

- it avoids leaking train/test labels into threat-intel features
- it still gives you a realistic external-API style integration path
- later you can replace the local mock API with your own real API lookup results

If you only want to start the saved mock API server:

```bash
python serve_mock_threat_intel_api.py --library data/threat_intel/bccc_cicids2018_mock_library.json
```
