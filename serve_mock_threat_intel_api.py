#!/usr/bin/env python3
"""
Serve a saved local mock threat-intel library over HTTP.
"""
import argparse
import json
import time

from src.threat_intel.mock_api import MockThreatIntelAPIServer


def main():
    parser = argparse.ArgumentParser(description="Serve a mock threat-intel API from a saved JSON library")
    parser.add_argument(
        "--library",
        default="data/threat_intel/bccc_cicids2018_mock_library.json",
        help="Path to the saved threat-intel JSON library",
    )
    args = parser.parse_args()

    with open(args.library, "r", encoding="utf-8") as f:
        library = json.load(f)

    with MockThreatIntelAPIServer(library) as server:
        print(f"Mock threat-intel API running at {server.base_url}")
        print("Press Ctrl+C to stop")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
