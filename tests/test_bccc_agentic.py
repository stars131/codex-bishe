"""
Tests for BCCC-CSE-CIC-IDS-2018 sampling and mock threat-intel API.
"""
import shutil
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from src.data.bccc_cicids2018 import BCCCCICIDS2018Adapter
from src.threat_intel.mock_api import (
    MockThreatIntelAPIServer,
    ThreatIntelAPIClient,
    ThreatIntelLibraryBuilder,
    compute_threat_intel_response,
)


def _write_nested_archive(base_dir: Path, outer_name: str, inner_name: str, csv_name: str, csv_text: str):
    outer_path = base_dir / outer_name
    inner_buffer = BytesIO()
    with ZipFile(inner_buffer, "w") as inner_zip:
        inner_zip.writestr(csv_name, csv_text)
    with ZipFile(outer_path, "w") as outer_zip:
        outer_zip.writestr(inner_name, inner_buffer.getvalue())
    return outer_path


def test_bccc_adapter_builds_multimodal_table_from_nested_archives():
    dataset_root = Path("pytest-bccc-agentic")
    dataset_root.mkdir(exist_ok=True)
    dataset_dir = dataset_root / "case-bccc-adapter"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    try:
        csv_text = """flow_id,timestamp,src_ip,src_port,dst_ip,dst_port,protocol,duration,packets_count,handshake_state,delta_start,handshake_duration,label
id1,2018-02-14 09:33:26.718731,18.221.219.4,39922,172.31.69.25,21,TCP,0.1,2,1,not a complete handshake,not a complete handshake,Brute_Force_FTP
id2,2018-02-14 09:35:26.718731,172.31.69.25,21,18.221.219.4,39922,TCP,0.2,3,2,0.001,0.002,Benign
"""
        _write_nested_archive(
            dataset_dir,
            outer_name="Wednesday_14_02_2018.zip",
            inner_name="Wednesday_14_02_2018/wednesday_14_02_2018_BF_FTP.zip",
            csv_name="wednesday_14_02_2018_BF_FTP.csv",
            csv_text=csv_text,
        )

        adapter = BCCCCICIDS2018Adapter(str(dataset_dir))
        raw_df, multimodal_df = adapter.build_multimodal_table(
            sample_per_member=2,
            keywords=["bf_ftp"],
            max_members=1,
        )

        assert len(raw_df) == 2
        assert "sample_id" in raw_df.columns
        assert "flow_duration" in multimodal_df.columns
        assert "flow_packets_count" in multimodal_df.columns
        assert "log_timestamp_hour" in multimodal_df.columns
        assert "log_protocol_tcp" in multimodal_df.columns
        assert "label" in multimodal_df.columns
    finally:
        shutil.rmtree(dataset_dir, ignore_errors=True)


def test_mock_threat_intel_api_returns_numeric_features():
    raw_df = pd.DataFrame({
        "sample_id": ["s1", "s2"],
        "src_ip": ["1.1.1.1", "2.2.2.2"],
        "dst_ip": ["10.0.0.1", "10.0.0.2"],
        "src_port": [12345, 12346],
        "dst_port": [21, 443],
        "protocol": ["TCP", "TCP"],
        "label": ["Attack", "Benign"],
    })
    library = ThreatIntelLibraryBuilder().build(raw_df)

    direct = compute_threat_intel_response(library, {
        "src_ip": "1.1.1.1",
        "dst_ip": "10.0.0.1",
        "src_port": "12345",
        "dst_port": "21",
        "protocol": "TCP",
    })
    assert direct["max_score"] >= 0.0
    assert direct["indicator_hits"] >= 0.0

    with MockThreatIntelAPIServer(library) as server:
        client = ThreatIntelAPIClient(server.base_url)
        enriched = client.enrich_dataframe(raw_df[["sample_id", "src_ip", "dst_ip", "src_port", "dst_port", "protocol"]])

    assert "sample_id" in enriched.columns
    assert "max_score" in enriched.columns
    assert enriched.shape[0] == 2
