from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ALLOWED = {"CC0-1.0", "CC-BY-4.0", "CC-BY-SA-4.0", "MIT", "Apache-2.0", "OWNED", "EXPLICIT-PERMISSION", "PUBLIC-DOMAIN"}
REVIEW_REQUIRED = {"CC-BY-NC-4.0", "RESEARCH-ONLY", "UNKNOWN"}


def audit(path: Path) -> dict:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    findings = []
    usable = []
    blocked = []
    for row in rows:
        licence = row.get("license", "UNKNOWN").strip().upper()
        record = {
            "source_id": row.get("source_id", ""),
            "url": row.get("url", ""),
            "license": licence,
            "proof_url": row.get("proof_url", ""),
        }
        if licence in ALLOWED and record["proof_url"]:
            usable.append(record)
        else:
            reason = "licence not approved for commercial ML use"
            if licence in ALLOWED and not record["proof_url"]:
                reason = "missing licence proof URL"
            if licence in REVIEW_REQUIRED:
                reason = "manual legal review required"
            record["reason"] = reason
            blocked.append(record)
        findings.append(record)
    return {"total": len(rows), "usable": usable, "blocked": blocked}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--report", type=Path, default=Path("source_audit.json"))
    args = parser.parse_args()
    report = audit(args.manifest)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"total": report["total"], "usable": len(report["usable"]), "blocked": len(report["blocked"])}, indent=2))
    raise SystemExit(1 if report["blocked"] else 0)
