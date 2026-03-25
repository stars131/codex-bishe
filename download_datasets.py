#!/usr/bin/env python3
"""
数据集一键下载脚本

下载四种网络安全数据集，用于多源数据融合（流量 + 日志 + 威胁情报）：
  1. CIC-IDS-2017     — 网络流量特征数据集（主数据集）
  2. CSE-CIC-IDS2018  — 大规模网络流量 + 系统日志
  3. UNSW-NB15        — 综合网络安全数据集
  4. KDD Cup 99       — 经典入侵检测基准数据集

另外自动拉取公开威胁情报源，用于三源融合。

使用方法:
    python download_datasets.py                # 交互式选择
    python download_datasets.py --all          # 下载全部数据集
    python download_datasets.py --dataset cicids2017
    python download_datasets.py --dataset cse2018
    python download_datasets.py --dataset unsw
    python download_datasets.py --dataset kddcup
    python download_datasets.py --dataset threat_intel
    python download_datasets.py --list         # 列出所有数据集信息
"""

import os
import sys
import io
import gzip
import json
import shutil
import hashlib
import argparse
import zipfile
import tarfile
import urllib.request
import urllib.error
import ssl
import time
from pathlib import Path
from datetime import datetime

# Windows 控制台 UTF-8 输出
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ============================================================
# 路径常量
# ============================================================
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
THREAT_DIR = DATA_DIR / "threat_intel"

# 各数据集子目录
DATASET_DIRS = {
    "cicids2017": RAW_DIR / "CIC-IDS-2017",
    "cse2018": RAW_DIR / "CSE-CIC-IDS2018",
    "unsw": RAW_DIR / "UNSW-NB15",
    "kddcup": RAW_DIR / "KDD-Cup-99",
    "threat_intel": THREAT_DIR,
}

# ============================================================
# 工具函数
# ============================================================

def _make_ssl_context():
    """创建宽松 SSL 上下文（部分学术站点证书有问题）"""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _sizeof_fmt(num_bytes):
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def download_file(url, dest_path, description="", retries=3, timeout=120):
    """
    下载单个文件，显示进度条。

    Args:
        url: 下载链接
        dest_path: 保存路径
        description: 可选描述
        retries: 重试次数
        timeout: 超时秒数
    Returns:
        成功返回 True，失败返回 False
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"  [跳过] 已存在: {dest_path.name}")
        return True

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Python-DatasetDownloader/1.0"
    }
    ctx = _make_ssl_context()
    label = description or dest_path.name

    for attempt in range(1, retries + 1):
        try:
            print(f"  [{attempt}/{retries}] 下载 {label}")
            print(f"    URL: {url}")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 1024 * 256  # 256 KB
                start_time = time.time()

                with open(dest_path, "wb") as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total > 0:
                            pct = downloaded / total * 100
                            elapsed = time.time() - start_time
                            speed = downloaded / max(elapsed, 0.01)
                            print(
                                f"\r    进度: {pct:5.1f}%  "
                                f"{_sizeof_fmt(downloaded)}/{_sizeof_fmt(total)}  "
                                f"{_sizeof_fmt(speed)}/s   ",
                                end="", flush=True,
                            )
                        else:
                            print(
                                f"\r    已下载: {_sizeof_fmt(downloaded)}   ",
                                end="", flush=True,
                            )

                print()  # 换行
                print(f"  [完成] 已保存: {dest_path}")
                return True

        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
            print(f"\n  [失败] {e}")
            if dest_path.exists():
                dest_path.unlink()
            if attempt < retries:
                wait = 3 * attempt
                print(f"  等待 {wait}s 后重试...")
                time.sleep(wait)

    return False


def extract_gz(gz_path, out_path):
    """解压 .gz 文件"""
    print(f"  解压 {gz_path.name} -> {out_path.name}")
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()


def extract_zip(zip_path, out_dir):
    """解压 .zip 文件"""
    print(f"  解压 {zip_path.name} -> {out_dir}/")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    zip_path.unlink()


def extract_tar(tar_path, out_dir):
    """解压 .tar.gz / .tgz 文件"""
    print(f"  解压 {tar_path.name} -> {out_dir}/")
    with tarfile.open(tar_path, "r:*") as t:
        t.extractall(out_dir)
    tar_path.unlink()


# ============================================================
# 1. KDD Cup 99
# ============================================================

def download_kddcup99():
    """下载 KDD Cup 99 全量数据集"""
    print("\n" + "=" * 60)
    print("  KDD Cup 99 -- 经典入侵检测数据集")
    print("  来源: UCI Machine Learning Repository")
    print("  特点: 41维特征, 5大类攻击 (DoS/Probe/R2L/U2R/Normal)")
    print("  融合角色: 流量特征 + 主机日志特征")
    print("=" * 60)

    out_dir = DATASET_DIRS["kddcup"]
    out_dir.mkdir(parents=True, exist_ok=True)

    full_csv = out_dir / "kddcup.data.csv"
    if full_csv.exists() and full_csv.stat().st_size > 0:
        print(f"  [跳过] 已存在: {full_csv.name} ({_sizeof_fmt(full_csv.stat().st_size)})")
    else:
        # 优先用 sklearn (最稳定)，再尝试 UCI 直链
        print("  尝试通过 sklearn 下载全量数据 (最稳定)...")
        if not _kddcup_sklearn_fallback(full_csv, percent10=False):
            gz_path = out_dir / "kddcup.data.gz"
            urls = [
                "https://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz",
                "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz",
            ]
            for url in urls:
                if download_file(url, gz_path, "kddcup.data.gz (~18MB)", retries=2):
                    extract_gz(gz_path, full_csv)
                    break

    # 10% 子集 (用于快速测试)
    subset_csv = out_dir / "kddcup.data_10_percent.csv"
    if subset_csv.exists() and subset_csv.stat().st_size > 0:
        print(f"  [跳过] 已存在: {subset_csv.name}")
    else:
        print("  尝试通过 sklearn 下载 10% 子集...")
        if not _kddcup_sklearn_fallback(subset_csv, percent10=True):
            gz_path = out_dir / "kddcup.data_10_percent.gz"
            urls = [
                "https://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz",
                "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz",
            ]
            for url in urls:
                if download_file(url, gz_path, "kddcup 10% (~2MB)", retries=2):
                    extract_gz(gz_path, subset_csv)
                    break

    # --- 测试数据 ---
    test_csv = out_dir / "corrected.csv"
    if not test_csv.exists():
        gz_path = out_dir / "corrected.gz"
        urls = [
            "https://kdd.ics.uci.edu/databases/kddcup99/corrected.gz",
            "http://kdd.ics.uci.edu/databases/kddcup99/corrected.gz",
        ]
        for url in urls:
            if download_file(url, gz_path, "corrected 测试集"):
                extract_gz(gz_path, test_csv)
                break
    else:
        print(f"  [跳过] 已存在: {test_csv.name}")

    _check_dir_result(out_dir)


def _kddcup_sklearn_fallback(csv_path, percent10=False):
    """sklearn 后备下载，返回 True/False 表示是否成功"""
    label = "10% 子集" if percent10 else "全量数据"
    print(f"  尝试通过 sklearn 后备下载 ({label})...")
    try:
        import numpy as np
        import pandas as pd
        from sklearn.datasets import fetch_kddcup99

        bunch = fetch_kddcup99(percent10=percent10, as_frame=False, download_if_missing=True)
        X, y = bunch.data, bunch.target

        decode = lambda v: v.decode("utf-8", errors="ignore") if isinstance(v, (bytes, bytearray)) else v
        vec_decode = np.vectorize(decode, otypes=[object])
        if getattr(X, "dtype", None) == object:
            X = vec_decode(X)
        if getattr(y, "dtype", None) == object:
            y = vec_decode(y)

        cols = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
            'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
            'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
            'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
            'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
            'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
            'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
        ]
        df = pd.DataFrame(X, columns=cols)
        df["label"] = y
        df.to_csv(csv_path, index=False, header=False)
        print(f"  [完成] sklearn 下载成功: {csv_path}")
        return True
    except Exception as e:
        print(f"  [失败] sklearn 后备下载也失败: {e}")
        return False


# ============================================================
# 2. CIC-IDS-2017
# ============================================================

def download_cicids2017():
    """下载 CIC-IDS-2017 数据集"""
    print("\n" + "=" * 60)
    print("  CIC-IDS-2017 — 网络入侵检测数据集")
    print("  来源: Canadian Institute for Cybersecurity, UNB")
    print("  特点: 80维流量特征，15种攻击类型，2,830,743条记录")
    print("  融合角色: 核心流量特征数据源")
    print("=" * 60)

    out_dir = DATASET_DIRS["cicids2017"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # CIC-IDS-2017 MachineLearningCSV 文件列表
    csv_files = [
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Wednesday-workingHours.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    ]

    # 检查是否已有文件
    existing = [f for f in csv_files if (out_dir / f).exists()]
    if len(existing) == len(csv_files):
        print("  [跳过] 所有 CSV 文件均已存在")
        _check_dir_result(out_dir)
        return

    # 尝试方法 1: 直接从 UNB 下载
    base_url = "https://iscxdownloads.cs.unb.ca/iscxdownloads/CIC-IDS-2017/TrafficLabelling/"
    print("\n  方法1: 从 UNB 官方站点下载...")
    success_count = 0
    for fname in csv_files:
        if (out_dir / fname).exists():
            print(f"  [跳过] 已存在: {fname}")
            success_count += 1
            continue
        url = base_url + urllib.request.quote(fname)
        if download_file(url, out_dir / fname, fname, retries=2, timeout=300):
            success_count += 1

    if success_count == len(csv_files):
        print("  UNB 官方站点下载成功!")
        _check_dir_result(out_dir)
        return

    # 尝试方法 2: 备用镜像
    mirror_url = "https://cic-ids-2017.s3.ca-central-1.amazonaws.com/TrafficLabelling/"
    print(f"\n  方法2: 尝试备用镜像...")
    for fname in csv_files:
        if (out_dir / fname).exists():
            continue
        url = mirror_url + urllib.request.quote(fname)
        download_file(url, out_dir / fname, fname, retries=2, timeout=300)

    remaining = [f for f in csv_files if not (out_dir / f).exists()]
    if remaining:
        print("\n  [提示] 以下文件自动下载失败，请手动下载:")
        print("  下载页面: https://www.unb.ca/cic/datasets/ids-2017.html")
        print(f"  放置目录: {out_dir}")
        for f in remaining:
            print(f"    - {f}")

    _check_dir_result(out_dir)


# ============================================================
# 3. CSE-CIC-IDS2018
# ============================================================

def download_cse2018():
    """下载 CSE-CIC-IDS2018 数据集"""
    print("\n" + "=" * 60)
    print("  CSE-CIC-IDS2018 — 大规模网络安全数据集")
    print("  来源: CIC + CSE, University of New Brunswick")
    print("  特点: 80维特征，7种攻击场景，含系统日志")
    print("  融合角色: 流量特征 + 系统日志特征")
    print("=" * 60)

    out_dir = DATASET_DIRS["cse2018"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # 处理后的 CSV 文件列表
    csv_files = [
        "02-14-2018.csv",
        "02-15-2018.csv",
        "02-16-2018.csv",
        "02-20-2018.csv",
        "02-21-2018.csv",
        "02-22-2018.csv",
        "02-23-2018.csv",
        "03-01-2018.csv",
        "03-02-2018.csv",
    ]

    existing = [f for f in csv_files if (out_dir / f).exists()]
    if existing:
        print(f"  已有 {len(existing)}/{len(csv_files)} 个文件")

    # 方法1: AWS S3 (无需认证)
    print("\n  方法1: 通过 AWS CLI 从 S3 下载...")
    if _try_aws_s3_download(out_dir):
        _check_dir_result(out_dir)
        return

    # 方法2: 直接链接
    s3_base = "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/Processed%20Traffic%20Data%20for%20ML%20Algorithms/"
    print("\n  方法2: 通过 HTTPS 从 S3 下载...")
    for fname in csv_files:
        if (out_dir / fname).exists():
            continue
        url = s3_base + urllib.request.quote(fname)
        download_file(url, out_dir / fname, fname, retries=2, timeout=600)

    remaining = [f for f in csv_files if not (out_dir / f).exists()]
    if remaining:
        print("\n  [提示] 以下文件自动下载失败，可手动下载:")
        print("  方法A - AWS CLI (推荐):")
        print(f"    aws s3 sync --no-sign-request \\")
        print(f'      "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" \\')
        print(f"      \"{out_dir}/\"")
        print("  方法B - 网页下载:")
        print("    https://www.unb.ca/cic/datasets/ids-2018.html")
        print(f"  放置目录: {out_dir}")

    _check_dir_result(out_dir)


def _try_aws_s3_download(out_dir):
    """尝试用 aws cli 下载"""
    try:
        import subprocess
        result = subprocess.run(["aws", "--version"], capture_output=True, timeout=10)
        if result.returncode != 0:
            print("  AWS CLI 不可用，跳过此方法")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  AWS CLI 未安装，跳过此方法")
        return False

    import subprocess
    cmd = [
        "aws", "s3", "sync", "--no-sign-request",
        "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/",
        str(out_dir) + "/",
    ]
    print(f"  执行: {' '.join(cmd[:5])} ...")
    try:
        result = subprocess.run(cmd, timeout=3600)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("  下载超时")
        return False


# ============================================================
# 4. UNSW-NB15
# ============================================================

def download_unsw_nb15():
    """下载 UNSW-NB15 数据集"""
    print("\n" + "=" * 60)
    print("  UNSW-NB15 — 综合网络安全数据集")
    print("  来源: UNSW Canberra Cyber, Australia")
    print("  特点: 49维特征，9种攻击，2,540,044条记录")
    print("  融合角色: 流量特征 + 内容特征 + 基础日志")
    print("=" * 60)

    out_dir = DATASET_DIRS["unsw"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # UNSW-NB15 主要文件
    files_info = {
        "UNSW-NB15_1.csv": "https://research.unsw.edu.au/sites/default/files/documents/UNSW-NB15_1.csv",
        "UNSW-NB15_2.csv": "https://research.unsw.edu.au/sites/default/files/documents/UNSW-NB15_2.csv",
        "UNSW-NB15_3.csv": "https://research.unsw.edu.au/sites/default/files/documents/UNSW-NB15_3.csv",
        "UNSW-NB15_4.csv": "https://research.unsw.edu.au/sites/default/files/documents/UNSW-NB15_4.csv",
        "NUSW-NB15_GT.csv": "https://research.unsw.edu.au/sites/default/files/documents/NUSW-NB15_GT.csv",
        "NUSW-NB15_features.csv": "https://research.unsw.edu.au/sites/default/files/documents/NUSW-NB15_features.csv",
        "UNSW-NB15_LIST_EVENTS.csv": "https://research.unsw.edu.au/sites/default/files/documents/UNSW-NB15_LIST_EVENTS.csv",
    }
    # 训练/测试 split
    train_test_info = {
        "UNSW_NB15_training-set.csv": "https://research.unsw.edu.au/sites/default/files/documents/UNSW_NB15_training-set.csv",
        "UNSW_NB15_testing-set.csv": "https://research.unsw.edu.au/sites/default/files/documents/UNSW_NB15_testing-set.csv",
    }
    files_info.update(train_test_info)

    for fname, url in files_info.items():
        download_file(url, out_dir / fname, fname, retries=3, timeout=300)

    # 检查结果, 如果有失败的给出 Kaggle 备选
    all_files = list(files_info.keys())
    remaining = [f for f in all_files if not (out_dir / f).exists()]
    if remaining:
        print("\n  [提示] 部分文件下载失败，备选下载方式:")
        print("  Kaggle: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15")
        print("  官网:   https://research.unsw.edu.au/projects/unsw-nb15-dataset")
        print(f"  放置目录: {out_dir}")

    _check_dir_result(out_dir)


# ============================================================
# 5. 威胁情报数据 (Threat Intelligence)
# ============================================================

def download_threat_intel():
    """下载公开威胁情报数据，用于第三数据源融合"""
    print("\n" + "=" * 60)
    print("  威胁情报数据 — 公开CTI源")
    print("  融合角色: 第三数据源 (威胁情报特征)")
    print("  包含: Abuse.ch 恶意IP/域名, MITRE ATT&CK 映射")
    print("=" * 60)

    out_dir = DATASET_DIRS["threat_intel"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Abuse.ch Feodo Tracker (C2 IP) -----
    abuse_dir = out_dir / "abuse_ch"
    abuse_dir.mkdir(parents=True, exist_ok=True)

    abuse_feeds = {
        "feodo_ipblocklist.json": "https://feodotracker.abuse.ch/downloads/ipblocklist.json",
        "feodo_ipblocklist.csv": "https://feodotracker.abuse.ch/downloads/ipblocklist.csv",
        "sslbl_ip_blacklist.csv": "https://sslbl.abuse.ch/blacklist/sslipblacklist.csv",
        "urlhaus_recent.csv": "https://urlhaus.abuse.ch/downloads/csv_recent/",
    }

    print("\n  [1/3] 下载 Abuse.ch 威胁情报...")
    for fname, url in abuse_feeds.items():
        download_file(url, abuse_dir / fname, fname, retries=2, timeout=60)

    # ----- Emerging Threats (Suricata/Snort 规则提取) -----
    et_dir = out_dir / "emerging_threats"
    et_dir.mkdir(parents=True, exist_ok=True)

    print("\n  [2/3] 下载 Emerging Threats 开放规则...")
    et_url = "https://rules.emergingthreats.net/open/suricata/emerging.rules.tar.gz"
    et_tar = et_dir / "emerging.rules.tar.gz"
    if download_file(et_url, et_tar, "Emerging Threats Rules", retries=2, timeout=120):
        if et_tar.exists():
            try:
                extract_tar(et_tar, et_dir)
            except Exception as e:
                print(f"  解压失败: {e}，保留压缩文件")

    # ----- MITRE ATT&CK 知识库 -----
    mitre_dir = out_dir / "mitre_attack"
    mitre_dir.mkdir(parents=True, exist_ok=True)

    print("\n  [3/3] 下载 MITRE ATT&CK 知识库...")
    mitre_feeds = {
        "enterprise-attack.json": "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json",
        "ics-attack.json": "https://raw.githubusercontent.com/mitre/cti/master/ics-attack/ics-attack.json",
    }
    for fname, url in mitre_feeds.items():
        download_file(url, mitre_dir / fname, fname, retries=2, timeout=120)

    # ----- 生成威胁情报特征映射文件 -----
    _generate_threat_intel_feature_mapping(out_dir)

    _check_dir_result(out_dir)


def _generate_threat_intel_feature_mapping(out_dir):
    """生成威胁情报特征映射配置，供模型训练时使用"""
    mapping = {
        "description": "威胁情报特征映射配置",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feature_groups": {
            "ip_reputation": {
                "description": "IP信誉特征 (基于Abuse.ch等黑名单)",
                "features": [
                    "is_known_c2",
                    "is_known_malware_ip",
                    "is_ssl_blacklisted",
                    "ip_threat_score",
                    "ip_first_seen_days",
                    "ip_report_count",
                ],
            },
            "domain_reputation": {
                "description": "域名信誉特征 (基于URLhaus等)",
                "features": [
                    "is_malicious_url",
                    "url_threat_type",
                    "domain_age_days",
                    "domain_report_count",
                ],
            },
            "attack_pattern": {
                "description": "攻击模式特征 (基于MITRE ATT&CK)",
                "features": [
                    "attack_technique_id",
                    "attack_tactic_id",
                    "attack_severity",
                    "attack_complexity",
                ],
            },
            "signature_match": {
                "description": "签名匹配特征 (基于Emerging Threats规则)",
                "features": [
                    "et_rule_matched",
                    "et_rule_category",
                    "et_rule_severity",
                    "signature_count",
                ],
            },
        },
        "enrichment_strategy": {
            "method": "将网络流量记录与威胁情报进行IP/端口/协议匹配",
            "steps": [
                "1. 提取流量记录的源IP/目的IP",
                "2. 与Abuse.ch黑名单进行匹配，生成IP信誉特征",
                "3. 与URLhaus进行URL/域名匹配，生成域名信誉特征",
                "4. 基于协议/端口/行为与MITRE ATT&CK映射，生成攻击模式特征",
                "5. 与ET规则进行签名匹配，生成签名特征",
                "6. 所有特征合并为威胁情报特征向量",
            ],
        },
    }

    mapping_path = out_dir / "threat_intel_features.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"\n  已生成威胁情报特征映射: {mapping_path.name}")


# ============================================================
# 辅助函数
# ============================================================

def _check_dir_result(directory):
    """统计目录下的文件情况"""
    directory = Path(directory)
    if not directory.exists():
        print(f"\n  目录不存在: {directory}")
        return

    files = list(directory.rglob("*"))
    file_count = sum(1 for f in files if f.is_file())
    total_size = sum(f.stat().st_size for f in files if f.is_file())

    print(f"\n  --- 目录统计: {directory.relative_to(PROJECT_ROOT)} ---")
    print(f"  文件数: {file_count}")
    print(f"  总大小: {_sizeof_fmt(total_size)}")

    # 列出 CSV 文件
    csv_files = sorted(f for f in files if f.is_file() and f.suffix.lower() == ".csv")
    if csv_files:
        print(f"  CSV 文件:")
        for f in csv_files[:15]:
            print(f"    {f.name:50s}  {_sizeof_fmt(f.stat().st_size)}")
        if len(csv_files) > 15:
            print(f"    ... 还有 {len(csv_files) - 15} 个文件")


def print_dataset_info():
    """打印所有数据集的详细信息"""
    info = """
╔══════════════════════════════════════════════════════════════╗
║           多源数据融合 — 数据集总览                          ║
║       融合类型: 流量(Traffic) + 日志(Log) + 威胁情报(CTI)    ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. CIC-IDS-2017        [cicids2017]                         ║
║     大小: ~6.5 GB (8个CSV文件)                               ║
║     记录: 2,830,743 条                                       ║
║     特征: 80 维流量特征 + 标签                               ║
║     攻击: DoS, DDoS, PortScan, Brute Force, Web Attack 等   ║
║     角色: ★ 核心流量特征数据源                               ║
║                                                              ║
║  2. CSE-CIC-IDS2018     [cse2018]                            ║
║     大小: ~8 GB (Processed CSV)                              ║
║     记录: 16,000,000+ 条                                     ║
║     特征: 80 维流量特征 + 系统日志                           ║
║     攻击: Brute Force, DoS, DDoS, Botnet, Infiltration 等   ║
║     角色: ★ 流量 + 日志特征数据源                            ║
║                                                              ║
║  3. UNSW-NB15           [unsw]                               ║
║     大小: ~1.8 GB (4个CSV文件 + 辅助文件)                    ║
║     记录: 2,540,044 条                                       ║
║     特征: 49 维 (基本+内容+时间+流量+附加)                   ║
║     攻击: Fuzzers, Analysis, Backdoors, DoS, Exploits 等    ║
║     角色: ★ 综合特征 (流量+内容+基础日志)                    ║
║                                                              ║
║  4. KDD Cup 99          [kddcup]                             ║
║     大小: ~18 MB (gz) / ~750 MB (解压)                       ║
║     记录: 4,898,431 条                                       ║
║     特征: 41 维 (基本+内容+流量+主机)                        ║
║     攻击: DoS, Probe, R2L, U2R                               ║
║     角色: ★ 经典基准 (流量+主机日志)                         ║
║                                                              ║
║  5. 威胁情报数据        [threat_intel]                       ║
║     Abuse.ch:  C2 IP黑名单, SSL黑名单, URLhaus              ║
║     Emerging Threats: 开放 Suricata/Snort 规则               ║
║     MITRE ATT&CK: 攻击技术知识库                             ║
║     角色: ★ 第三数据源 — 威胁情报特征                        ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  三源融合策略:                                               ║
║    源1 (流量): 网络流特征 (包长/速率/IAT/标志位等)           ║
║    源2 (日志): 主机/内容/系统日志特征                        ║
║    源3 (CTI):  IP信誉/域名信誉/攻击模式/签名匹配            ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(info)

    # 显示本地已下载状态
    print("  本地状态:")
    for key, d in DATASET_DIRS.items():
        if d.exists():
            files = list(d.rglob("*"))
            fc = sum(1 for f in files if f.is_file())
            sz = sum(f.stat().st_size for f in files if f.is_file())
            print(f"    [{key:14s}]  {fc:3d} 文件  {_sizeof_fmt(sz):>10s}  ✓ {d.relative_to(PROJECT_ROOT)}")
        else:
            print(f"    [{key:14s}]  未下载                  — {d.relative_to(PROJECT_ROOT)}")
    print()


# ============================================================
# 主函数
# ============================================================

DOWNLOADERS = {
    "cicids2017": ("CIC-IDS-2017", download_cicids2017),
    "cse2018": ("CSE-CIC-IDS2018", download_cse2018),
    "unsw": ("UNSW-NB15", download_unsw_nb15),
    "kddcup": ("KDD Cup 99", download_kddcup99),
    "threat_intel": ("威胁情报数据", download_threat_intel),
}


def interactive_menu():
    """交互式菜单"""
    print("\n  请选择要下载的数据集 (输入编号，多个用逗号分隔，0=全部):\n")
    items = list(DOWNLOADERS.items())
    for i, (key, (name, _)) in enumerate(items, 1):
        d = DATASET_DIRS[key]
        status = "✓ 已下载" if d.exists() and any(d.iterdir()) else "  未下载"
        print(f"    {i}. {name:20s}  [{key:14s}]  {status}")
    print(f"    0. 全部下载")
    print()

    try:
        choice = input("  请输入 (默认 0): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  已取消")
        return []

    if not choice or choice == "0":
        return [key for key, _ in items]

    selected = []
    for c in choice.replace(" ", "").split(","):
        try:
            idx = int(c)
            if 1 <= idx <= len(items):
                selected.append(items[idx - 1][0])
        except ValueError:
            if c in DOWNLOADERS:
                selected.append(c)

    return selected


def main():
    parser = argparse.ArgumentParser(
        description="多源数据融合 — 数据集一键下载",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python download_datasets.py --all              # 下载全部
  python download_datasets.py --dataset kddcup   # 仅下载 KDD Cup 99
  python download_datasets.py --dataset cicids2017 --dataset unsw
  python download_datasets.py --list             # 查看数据集信息
        """,
    )
    parser.add_argument("--all", action="store_true", help="下载全部数据集")
    parser.add_argument(
        "--dataset", "-d",
        action="append",
        choices=list(DOWNLOADERS.keys()),
        help="指定要下载的数据集 (可多次使用)",
    )
    parser.add_argument("--list", "-l", action="store_true", help="列出所有数据集信息")

    args = parser.parse_args()

    print("=" * 60)
    print("  多源数据融合 — 数据集自动下载工具")
    print("  融合类型: 流量 + 日志 + 威胁情报")
    print("=" * 60)

    if args.list:
        print_dataset_info()
        return

    # 确定要下载的数据集
    if args.all:
        targets = list(DOWNLOADERS.keys())
    elif args.dataset:
        targets = args.dataset
    else:
        print_dataset_info()
        targets = interactive_menu()

    if not targets:
        print("  未选择任何数据集，退出。")
        return

    print(f"\n  将下载: {', '.join(DOWNLOADERS[t][0] for t in targets)}")
    print()

    # 创建基础目录
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # 执行下载
    results = {}
    for key in targets:
        name, func = DOWNLOADERS[key]
        try:
            func()
            d = DATASET_DIRS[key]
            has_files = d.exists() and any(d.rglob("*"))
            results[name] = "✓ 成功" if has_files else "△ 部分"
        except KeyboardInterrupt:
            print(f"\n  用户中断 {name} 的下载")
            results[name] = "✗ 中断"
            break
        except Exception as e:
            print(f"\n  [错误] {name}: {e}")
            results[name] = f"✗ 失败: {e}"

    # 汇总
    print("\n" + "=" * 60)
    print("  下载结果汇总")
    print("=" * 60)
    for name, status in results.items():
        print(f"  {name:25s}  {status}")

    # 总大小
    total_size = 0
    for d in DATASET_DIRS.values():
        if d.exists():
            total_size += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    print(f"\n  数据总大小: {_sizeof_fmt(total_size)}")

    print(f"\n  数据目录: {DATA_DIR}")
    print("  接下来可以运行预处理:")
    print("    python main.py --data_dir data/raw/CIC-IDS-2017 --mode preprocess --config src/config/config_server.yaml")
    print()


if __name__ == "__main__":
    main()
