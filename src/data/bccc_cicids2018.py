"""
Adapters for sampling and reshaping BCCC-CSE-CIC-IDS-2018 into the project format.
"""
from __future__ import annotations

import hashlib
import ipaddress
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from zipfile import ZipFile

import numpy as np
import pandas as pd


DEFAULT_MEMBER_KEYWORDS = (
    "benign",
    "bf_ftp",
    "bf_ssh",
    "bot",
    "sql_injection",
    "infiltration",
)

NON_FLOW_COLUMNS = {
    "flow_id",
    "timestamp",
    "src_ip",
    "src_port",
    "dst_ip",
    "dst_port",
    "protocol",
    "delta_start",
    "handshake_duration",
    "handshake_state",
    "label",
}


def _safe_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return numeric


def _parse_ip(value: str):
    try:
        return ipaddress.ip_address(str(value))
    except ValueError:
        return None


def _stable_hash_fraction(value: str) -> float:
    digest = hashlib.md5(str(value).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


@dataclass(frozen=True)
class NestedArchiveMember:
    outer_zip_path: str
    inner_zip_name: str
    csv_name: str

    @property
    def member_key(self) -> str:
        return f"{Path(self.outer_zip_path).name}:{self.inner_zip_name}:{self.csv_name}"


class BCCCCICIDS2018Adapter:
    """Sample nested BCCC-CSE-CIC-IDS-2018 archives and build flow/log tables."""

    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"BCCC dataset directory not found: {self.dataset_dir}")

    def discover_members(self) -> List[NestedArchiveMember]:
        members: List[NestedArchiveMember] = []
        for outer_zip in sorted(self.dataset_dir.glob("*.zip")):
            with ZipFile(outer_zip) as outer_archive:
                for inner_name in outer_archive.namelist():
                    if not inner_name.lower().endswith(".zip"):
                        continue
                    with outer_archive.open(inner_name) as inner_file:
                        inner_bytes = BytesIO(inner_file.read())
                    with ZipFile(inner_bytes) as inner_archive:
                        csv_names = [name for name in inner_archive.namelist() if name.lower().endswith(".csv")]
                        if not csv_names:
                            continue
                        members.append(
                            NestedArchiveMember(
                                outer_zip_path=str(outer_zip),
                                inner_zip_name=inner_name,
                                csv_name=csv_names[0],
                            )
                        )
        if not members:
            raise ValueError(f"No nested CSV archives found under {self.dataset_dir}")
        return members

    def select_members(
        self,
        keywords: Optional[Sequence[str]] = None,
        max_members: Optional[int] = None,
    ) -> List[NestedArchiveMember]:
        keywords = tuple((keywords or DEFAULT_MEMBER_KEYWORDS))
        discovered = self.discover_members()
        selected: List[NestedArchiveMember] = []
        used_keys = set()

        for keyword in keywords:
            lowered = keyword.lower()
            for member in discovered:
                token = f"{member.inner_zip_name} {member.csv_name}".lower()
                if lowered in token and member.member_key not in used_keys:
                    selected.append(member)
                    used_keys.add(member.member_key)
                    break

        if not selected:
            selected = discovered[: max_members or len(discovered)]

        if max_members is not None:
            selected = selected[:max_members]
        return selected

    def read_member_sample(
        self,
        member: NestedArchiveMember,
        nrows: Optional[int],
    ) -> pd.DataFrame:
        with ZipFile(member.outer_zip_path) as outer_archive:
            with outer_archive.open(member.inner_zip_name) as inner_file:
                inner_bytes = BytesIO(inner_file.read())
        with ZipFile(inner_bytes) as inner_archive:
            with inner_archive.open(member.csv_name) as csv_file:
                read_nrows = None if nrows is None or nrows <= 0 else nrows
                frame = pd.read_csv(csv_file, nrows=read_nrows, low_memory=False)

        frame["__outer_zip__"] = Path(member.outer_zip_path).name
        frame["__inner_zip__"] = Path(member.inner_zip_name).name
        frame["__csv_name__"] = Path(member.csv_name).name
        return frame

    def build_raw_sample(
        self,
        sample_per_member: int = 200,
        keywords: Optional[Sequence[str]] = None,
        max_members: Optional[int] = None,
    ) -> pd.DataFrame:
        frames = [
            self.read_member_sample(member, sample_per_member)
            for member in self.select_members(keywords=keywords, max_members=max_members)
        ]
        raw = pd.concat(frames, ignore_index=True)
        if "label" not in raw.columns:
            raise ValueError("BCCC sample is missing the label column")

        raw = raw.reset_index(drop=True)
        base_ids = raw.get("flow_id", pd.Series([f"row_{idx}" for idx in range(len(raw))]))
        raw["sample_id"] = [
            f"{value}__{idx}"
            for idx, value in enumerate(base_ids.astype(str).tolist())
        ]
        return raw

    def build_flow_table(self, raw: pd.DataFrame) -> pd.DataFrame:
        numeric_candidates = []
        for column in raw.columns:
            if column in NON_FLOW_COLUMNS or column.startswith("__") or column == "sample_id":
                continue
            numeric_series = pd.to_numeric(raw[column], errors="coerce")
            if numeric_series.notna().any():
                numeric_candidates.append(column)

        flow = pd.DataFrame(index=raw.index)
        for column in numeric_candidates:
            flow[f"flow_{column}"] = _safe_numeric(raw[column]).astype(np.float32)

        if flow.empty:
            raise ValueError("No numeric flow features could be extracted from BCCC sample")
        return flow

    def build_log_table(self, raw: pd.DataFrame) -> pd.DataFrame:
        timestamps = pd.to_datetime(raw.get("timestamp"), errors="coerce")
        src_ips = raw.get("src_ip", pd.Series(["0.0.0.0"] * len(raw), index=raw.index)).astype(str)
        dst_ips = raw.get("dst_ip", pd.Series(["0.0.0.0"] * len(raw), index=raw.index)).astype(str)
        src_ports = _safe_numeric(raw.get("src_port", pd.Series([0] * len(raw), index=raw.index)))
        dst_ports = _safe_numeric(raw.get("dst_port", pd.Series([0] * len(raw), index=raw.index)))
        handshake_state = _safe_numeric(raw.get("handshake_state", pd.Series([0] * len(raw), index=raw.index)))
        protocols = raw.get("protocol", pd.Series(["unknown"] * len(raw), index=raw.index)).astype(str).str.upper()

        src_ip_objs = src_ips.map(_parse_ip)
        dst_ip_objs = dst_ips.map(_parse_ip)

        log = pd.DataFrame(index=raw.index)
        log["log_timestamp_hour"] = timestamps.dt.hour.fillna(0).astype(np.float32)
        log["log_timestamp_minute"] = timestamps.dt.minute.fillna(0).astype(np.float32)
        log["log_timestamp_weekday"] = timestamps.dt.weekday.fillna(0).astype(np.float32)
        log["log_src_port"] = src_ports.astype(np.float32)
        log["log_dst_port"] = dst_ports.astype(np.float32)
        log["log_src_port_well_known"] = (src_ports <= 1024).astype(np.float32)
        log["log_dst_port_well_known"] = (dst_ports <= 1024).astype(np.float32)
        log["log_src_port_registered"] = ((src_ports > 1024) & (src_ports <= 49151)).astype(np.float32)
        log["log_dst_port_registered"] = ((dst_ports > 1024) & (dst_ports <= 49151)).astype(np.float32)
        log["log_handshake_state"] = handshake_state.astype(np.float32)
        log["log_handshake_missing"] = raw.get(
            "handshake_duration",
            pd.Series([""] * len(raw), index=raw.index),
        ).astype(str).str.contains("not a complete handshake", case=False, na=False).astype(np.float32)
        log["log_delta_start_numeric"] = _safe_numeric(raw.get("delta_start", pd.Series([0] * len(raw), index=raw.index))).astype(np.float32)
        log["log_handshake_duration_numeric"] = _safe_numeric(
            raw.get("handshake_duration", pd.Series([0] * len(raw), index=raw.index))
        ).astype(np.float32)
        log["log_protocol_tcp"] = (protocols == "TCP").astype(np.float32)
        log["log_protocol_udp"] = (protocols == "UDP").astype(np.float32)
        log["log_protocol_icmp"] = (protocols == "ICMP").astype(np.float32)
        log["log_protocol_other"] = (~protocols.isin(["TCP", "UDP", "ICMP"])).astype(np.float32)
        log["log_src_ip_private"] = src_ip_objs.map(lambda item: float(item.is_private) if item else 0.0).astype(np.float32)
        log["log_dst_ip_private"] = dst_ip_objs.map(lambda item: float(item.is_private) if item else 0.0).astype(np.float32)
        log["log_same_subnet_24"] = [
            float(
                src is not None
                and dst is not None
                and getattr(src, "version", None) == 4
                and getattr(dst, "version", None) == 4
                and ".".join(str(src).split(".")[:3]) == ".".join(str(dst).split(".")[:3])
            )
            for src, dst in zip(src_ip_objs, dst_ip_objs)
        ]
        log["log_src_ip_hash"] = src_ips.map(_stable_hash_fraction).astype(np.float32)
        log["log_dst_ip_hash"] = dst_ips.map(_stable_hash_fraction).astype(np.float32)
        log["log_member_hash"] = raw["__inner_zip__"].astype(str).map(_stable_hash_fraction).astype(np.float32)
        return log

    def build_multimodal_table(
        self,
        sample_per_member: int = 200,
        keywords: Optional[Sequence[str]] = None,
        max_members: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        raw = self.build_raw_sample(
            sample_per_member=sample_per_member,
            keywords=keywords,
            max_members=max_members,
        )
        flow = self.build_flow_table(raw)
        log = self.build_log_table(raw)
        multimodal = pd.concat(
            [
                raw[["sample_id", "label"]],
                flow,
                log,
            ],
            axis=1,
        )
        return raw, multimodal

    @staticmethod
    def save_outputs(
        raw: pd.DataFrame,
        multimodal: pd.DataFrame,
        raw_path: str,
        multimodal_path: str,
    ) -> None:
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        os.makedirs(os.path.dirname(multimodal_path), exist_ok=True)
        raw.to_csv(raw_path, index=False)
        multimodal.to_csv(multimodal_path, index=False)
