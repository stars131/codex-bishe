# CIC-IDS-2018 Agentic Fusion Data Format

This project now supports the following pipeline:

1. Flow features + log features: feature-level fusion
2. Threat-intel features: decision-level fusion
3. Agentic mode: inference-time rule controller on top of fused decisions

The code entry points are:

- `main.py`
- `src/data/multimodal_builder.py`
- `src/models/fusion_net.py`
- `src/config/config_cicids2018_agentic.yaml`

## Recommended data preparation

Prepare your data in one of these two formats.

### Format A: single processed table

Use when you already have one processed CSV/Parquet file containing both flow and log columns.

Required columns:

- label column, for example `label`
- optional sample id column, for example `sample_id`
- numeric flow feature columns
- numeric log feature columns

Example:

```text
sample_id,label,flow_duration,flow_packets,flow_bytes,log_event_count,log_error_count
s1,BENIGN,12,3,800,0,0
s2,DoS,85,27,9100,4,1
```

Corresponding config pattern:

```yaml
data:
  multimodal:
    enabled: true
    input_format: single_table
    path: data/processed/um_nids_cicids2018.csv
    label_column: label
    id_column: sample_id
    flow:
      prefixes: ["flow_", "net_", "traffic_"]
    log:
      prefixes: ["log_", "host_", "event_", "context_"]
```

Notes:

- Flow and log columns must be numeric.
- String labels are supported and will be encoded automatically.
- If your prefixes are not clean enough, use `columns:` explicitly instead of `prefixes:`.

### Format B: pre-split files

Use when you already have separate train/val/test files.

Each split can be provided in either of these ways:

- one unified table with `path`
- separate files with `flow_path`, `log_path`, `label_path`, and optional `id_path`

Example:

```yaml
data:
  multimodal:
    enabled: true
    input_format: pre_split
    splits:
      train:
        flow_path: data/processed/train_flow.csv
        log_path: data/processed/train_log.csv
        label_path: data/processed/train_label.csv
        id_path: data/processed/train_ids.csv
      val:
        flow_path: data/processed/val_flow.csv
        log_path: data/processed/val_log.csv
        label_path: data/processed/val_label.csv
        id_path: data/processed/val_ids.csv
      test:
        flow_path: data/processed/test_flow.csv
        log_path: data/processed/test_log.csv
        label_path: data/processed/test_label.csv
        id_path: data/processed/test_ids.csv
```

## Threat-intel sidecar format

Threat-intel is consumed as an external numeric feature file. You will generate this yourself after querying APIs or processing your own intel.

Two join modes are supported.

### Mode 1: `row_order`

Use when the threat-intel file is already aligned row-by-row with the multimodal table.

Example:

```text
intel_score,intel_hits,cve_risk_score
0.91,3,0.84
0.12,0,0.05
```

### Mode 2: `key`

Use when the threat-intel file must be joined by an id such as `sample_id`.

Example:

```text
sample_id,intel_score,intel_hits,cve_risk_score
s1,0.91,3,0.84
s2,0.12,0,0.05
```

Corresponding config pattern:

```yaml
data:
  threat_intel:
    enabled: true
    source_name: threat_intel
    source_path: data/threat_intel/cicids2018_intel.csv
    join_strategy: key
    join_key: sample_id
    intel_key: sample_id
    feature_columns: [intel_score, intel_hits, cve_risk_score]
```

Notes:

- Threat-intel features must be numeric.
- If `feature_columns` is omitted, all columns except the join key will be treated as intel features.
- For `key` mode, unmatched rows are filled with `0.0`.

## Model behavior

Current three-source behavior is fixed as:

1. source 1: flow
2. source 2: log
3. source 3: threat_intel

The model works as follows:

1. Flow and log are encoded and fused by `FusionNet`
2. Threat-intel is encoded by a separate branch
3. Final logits are fused by a learned decision gate
4. If `agentic_mode.enabled=true`, inference-time rules can override or boost final decisions

## Commands

## One-click UM-NIDS run

If you already have a processed unified file from `UM-NIDS-Tool`, use:

```bash
python run_um_nids_agentic.py --input data/processed/um_nids_cicids2018.csv
```

If you also have a threat-intel sidecar file:

```bash
python run_um_nids_agentic.py \
  --input data/processed/um_nids_cicids2018.csv \
  --threat-intel data/threat_intel/cicids2018_intel.csv
```

What this runner does:

1. detects label/id columns
2. infers flow columns and log columns
3. writes a runtime YAML config under `outputs/generated_configs/`
4. runs preprocess
5. runs train
6. runs evaluate
7. optionally runs report

Server wrapper:

```bash
INPUT_FILE=data/processed/um_nids_cicids2018.csv bash server_train_um_nids_agentic.sh
```

### Preprocess processed files into project artifacts

```bash
python main.py --config src/config/config_cicids2018_agentic.yaml --mode preprocess
```

### Train

```bash
python main.py --config src/config/config_cicids2018_agentic.yaml --mode train
```

### Evaluate

```bash
python main.py --config src/config/config_cicids2018_agentic.yaml --mode evaluate
```

### Full run

```bash
python main.py --config src/config/config_cicids2018_agentic.yaml --mode full
```

## Outputs

After preprocessing, the builder writes:

- `data/processed/multi_source_data.pkl`
- `data/processed/single_source_data.pkl`

After evaluation, the result payload can include:

- predictions
- attention weights
- agentic actions

## Practical recommendation for your thesis setup

Use this split:

- flow: statistical and traffic behavior features from CIC-IDS-2018
- log: host or event features derived from processed files
- threat_intel: external lookup features such as intel score, IOC hit count, CVE severity score, blacklist hit count, TTP relevance score

That matches your stated design:

- flow/log fusion
- threat-intel decision fusion
- agentic inference layer
