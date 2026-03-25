"""
Local mock threat-intel library and HTTP API.
"""
from __future__ import annotations

import ipaddress
import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import urlopen

import numpy as np
import pandas as pd


def _is_malicious(label: str) -> bool:
    return str(label).strip().lower() not in {"benign", "normal"}


def _risk_from_counts(malicious_count: int, benign_count: int) -> float:
    total = malicious_count + benign_count
    if total == 0:
        return 0.0
    ratio = malicious_count / total
    return float(min(0.99, 0.05 + 0.94 * ratio))


class ThreatIntelLibraryBuilder:
    """Build a simple local IOC library from sampled BCCC records."""

    HEURISTIC_PORT_SCORES = {
        "21": 0.85,
        "22": 0.82,
        "23": 0.88,
        "80": 0.58,
        "81": 0.52,
        "443": 0.30,
        "445": 0.80,
        "1433": 0.78,
        "3306": 0.76,
        "3389": 0.82,
        "5900": 0.72,
        "8080": 0.62,
    }

    HEURISTIC_PROTOCOL_SCORES = {
        "TCP": 0.48,
        "UDP": 0.34,
        "ICMP": 0.42,
    }

    def __init__(self, strategy: str = "heuristic"):
        self.strategy = strategy

    def build(self, raw: pd.DataFrame) -> Dict:
        if self.strategy == "observed_labels":
            return self._build_from_observed_labels(raw)
        if self.strategy != "heuristic":
            raise ValueError(f"Unsupported threat-intel library strategy: {self.strategy}")
        return self._build_heuristic(raw)

    def _build_from_observed_labels(self, raw: pd.DataFrame) -> Dict:
        working = raw.copy()
        working["is_malicious"] = working["label"].map(_is_malicious)

        ip_records = []
        for column, role in [("src_ip", "src"), ("dst_ip", "dst")]:
            if column not in working.columns:
                continue
            part = working[[column, "label", "is_malicious"]].copy()
            part.columns = ["indicator", "label", "is_malicious"]
            part["role"] = role
            ip_records.append(part)
        ip_frame = pd.concat(ip_records, ignore_index=True) if ip_records else pd.DataFrame(columns=["indicator", "label", "is_malicious", "role"])

        port_records = []
        for column, role in [("src_port", "src"), ("dst_port", "dst")]:
            if column not in working.columns:
                continue
            part = working[[column, "label", "is_malicious"]].copy()
            part.columns = ["indicator", "label", "is_malicious"]
            part["indicator"] = part["indicator"].astype(str)
            part["role"] = role
            port_records.append(part)
        port_frame = pd.concat(port_records, ignore_index=True) if port_records else pd.DataFrame(columns=["indicator", "label", "is_malicious", "role"])

        protocol_frame = pd.DataFrame({
            "indicator": working.get("protocol", pd.Series(["unknown"] * len(working))).astype(str).str.upper(),
            "label": working["label"].astype(str),
            "is_malicious": working["is_malicious"],
        })

        return {
            "ips": self._aggregate_indicator_frame(ip_frame),
            "ports": self._aggregate_indicator_frame(port_frame),
            "protocols": self._aggregate_indicator_frame(protocol_frame),
            "metadata": {
                "samples": int(len(working)),
                "strategy": "observed_labels",
                "attack_labels": sorted({str(label) for label in working["label"].astype(str) if _is_malicious(label)}),
            },
        }

    def _build_heuristic(self, raw: pd.DataFrame) -> Dict:
        working = raw.copy()
        ip_series = pd.concat(
            [
                working.get("src_ip", pd.Series(dtype=str)).astype(str),
                working.get("dst_ip", pd.Series(dtype=str)).astype(str),
            ],
            ignore_index=True,
        )
        port_series = pd.concat(
            [
                working.get("src_port", pd.Series(dtype=str)).astype(str),
                working.get("dst_port", pd.Series(dtype=str)).astype(str),
            ],
            ignore_index=True,
        )
        protocol_series = working.get("protocol", pd.Series(dtype=str)).astype(str).str.upper()

        ip_counts = ip_series.value_counts()
        port_counts = port_series.value_counts()
        protocol_counts = protocol_series.value_counts()

        ips = {}
        for indicator, count in ip_counts.items():
            ips[str(indicator)] = {
                "risk_score": self._heuristic_ip_score(str(indicator), int(count)),
                "malicious_count": int(count),
                "benign_count": 0,
                "attack_labels": [],
            }

        ports = {}
        for indicator, count in port_counts.items():
            ports[str(indicator)] = {
                "risk_score": self._heuristic_port_score(str(indicator), int(count)),
                "malicious_count": int(count),
                "benign_count": 0,
                "attack_labels": [],
            }

        protocols = {}
        for indicator, count in protocol_counts.items():
            protocols[str(indicator)] = {
                "risk_score": self._heuristic_protocol_score(str(indicator), int(count)),
                "malicious_count": int(count),
                "benign_count": 0,
                "attack_labels": [],
            }

        return {
            "ips": ips,
            "ports": ports,
            "protocols": protocols,
            "metadata": {
                "samples": int(len(working)),
                "strategy": "heuristic",
                "attack_labels": [],
            },
        }

    def _aggregate_indicator_frame(self, frame: pd.DataFrame) -> Dict[str, Dict]:
        if frame.empty:
            return {}

        payload: Dict[str, Dict] = {}
        for indicator, group in frame.groupby("indicator", dropna=False):
            malicious_count = int(group["is_malicious"].sum())
            benign_count = int((~group["is_malicious"]).sum())
            payload[str(indicator)] = {
                "risk_score": _risk_from_counts(malicious_count, benign_count),
                "malicious_count": malicious_count,
                "benign_count": benign_count,
                "attack_labels": sorted({
                    str(label)
                    for label in group.loc[group["is_malicious"], "label"].astype(str)
                }),
            }
        return payload

    def _heuristic_ip_score(self, indicator: str, count: int) -> float:
        try:
            ip_obj = ipaddress.ip_address(indicator)
            base = 0.18 if ip_obj.is_private else 0.55
        except ValueError:
            base = 0.10
        frequency_bonus = min(0.18, np.log1p(count) / 20.0)
        return float(min(0.95, base + frequency_bonus))

    def _heuristic_port_score(self, indicator: str, count: int) -> float:
        if indicator in self.HEURISTIC_PORT_SCORES:
            base = self.HEURISTIC_PORT_SCORES[indicator]
        else:
            try:
                port = int(indicator)
            except ValueError:
                port = 0
            if port <= 0:
                base = 0.05
            elif port <= 1024:
                base = 0.28
            elif port <= 49151:
                base = 0.18
            else:
                base = 0.10
        frequency_bonus = min(0.10, np.log1p(count) / 30.0)
        return float(min(0.95, base + frequency_bonus))

    def _heuristic_protocol_score(self, indicator: str, count: int) -> float:
        base = self.HEURISTIC_PROTOCOL_SCORES.get(indicator.upper(), 0.20)
        frequency_bonus = min(0.08, np.log1p(count) / 40.0)
        return float(min(0.95, base + frequency_bonus))

    @staticmethod
    def save(library: Dict, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            json.dump(library, f, indent=2, ensure_ascii=False)


def compute_threat_intel_response(library: Dict, params: Dict[str, str]) -> Dict[str, float]:
    src_ip = params.get("src_ip", "")
    dst_ip = params.get("dst_ip", "")
    src_port = str(params.get("src_port", ""))
    dst_port = str(params.get("dst_port", ""))
    protocol = str(params.get("protocol", "")).upper()

    ip_db = library.get("ips", {})
    port_db = library.get("ports", {})
    protocol_db = library.get("protocols", {})

    src_ip_item = ip_db.get(src_ip, {})
    dst_ip_item = ip_db.get(dst_ip, {})
    src_port_item = port_db.get(src_port, {})
    dst_port_item = port_db.get(dst_port, {})
    protocol_item = protocol_db.get(protocol, {})

    scores = [
        float(src_ip_item.get("risk_score", 0.0)),
        float(dst_ip_item.get("risk_score", 0.0)),
        float(src_port_item.get("risk_score", 0.0)),
        float(dst_port_item.get("risk_score", 0.0)),
        float(protocol_item.get("risk_score", 0.0)),
    ]
    attack_labels = set()
    for item in [src_ip_item, dst_ip_item, src_port_item, dst_port_item, protocol_item]:
        attack_labels.update(item.get("attack_labels", []))

    malicious_counts = [
        int(src_ip_item.get("malicious_count", 0)),
        int(dst_ip_item.get("malicious_count", 0)),
        int(src_port_item.get("malicious_count", 0)),
        int(dst_port_item.get("malicious_count", 0)),
        int(protocol_item.get("malicious_count", 0)),
    ]

    return {
        "src_ip_score": scores[0],
        "dst_ip_score": scores[1],
        "src_port_score": scores[2],
        "dst_port_score": scores[3],
        "protocol_score": scores[4],
        "mean_score": float(np.mean(scores)),
        "max_score": float(np.max(scores)),
        "indicator_hits": float(sum(score > 0 for score in scores)),
        "malicious_count_sum": float(sum(malicious_counts)),
        "attack_label_diversity": float(len(attack_labels)),
    }


class _ThreatIntelRequestHandler(BaseHTTPRequestHandler):
    library: Dict = {}

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json({"status": "ok"})
            return
        if parsed.path != "/query":
            self._send_json({"error": "not_found"}, status=404)
            return

        query = {key: values[0] for key, values in parse_qs(parsed.query).items()}
        response = compute_threat_intel_response(self.library, query)
        self._send_json(response)

    def log_message(self, format, *args):
        return

    def _send_json(self, payload: Dict, status: int = 200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@dataclass
class MockThreatIntelAPIServer:
    library: Dict
    host: str = "127.0.0.1"
    port: int = 0

    def __post_init__(self):
        handler = type("BoundThreatIntelHandler", (_ThreatIntelRequestHandler,), {})
        handler.library = self.library
        self._server = ThreadingHTTPServer((self.host, self.port), handler)
        self.host, self.port = self._server.server_address
        self._thread: Optional[threading.Thread] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self):
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.stop()


class ThreatIntelAPIClient:
    """Fetch sample-level threat-intel features from the local mock API."""

    def __init__(self, base_url: str, timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._cache: Dict[Tuple[str, str, str, str, str], Dict[str, float]] = {}

    def query(
        self,
        src_ip: str,
        dst_ip: str,
        src_port: str,
        dst_port: str,
        protocol: str,
    ) -> Dict[str, float]:
        key = (str(src_ip), str(dst_ip), str(src_port), str(dst_port), str(protocol).upper())
        if key in self._cache:
            return self._cache[key]

        query_string = urlencode({
            "src_ip": key[0],
            "dst_ip": key[1],
            "src_port": key[2],
            "dst_port": key[3],
            "protocol": key[4],
        })
        with urlopen(f"{self.base_url}/query?{query_string}", timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        self._cache[key] = payload
        return payload

    def enrich_dataframe(
        self,
        raw: pd.DataFrame,
        sample_id_column: str = "sample_id",
    ) -> pd.DataFrame:
        rows = []
        for row in raw.itertuples(index=False):
            payload = self.query(
                src_ip=getattr(row, "src_ip", ""),
                dst_ip=getattr(row, "dst_ip", ""),
                src_port=getattr(row, "src_port", ""),
                dst_port=getattr(row, "dst_port", ""),
                protocol=getattr(row, "protocol", ""),
            )
            rows.append({
                sample_id_column: getattr(row, sample_id_column),
                **payload,
            })
        return pd.DataFrame(rows)
