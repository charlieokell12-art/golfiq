from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "research" / "video_manifest.csv"


def youtube_oembed(url: str) -> dict:
    endpoint = "https://www.youtube.com/oembed?format=json&url=" + quote(url, safe="")
    r = requests.get(endpoint, timeout=15)
    r.raise_for_status()
    data = r.json()
    return {"title": data.get("title", ""), "author": data.get("author_name", ""), "thumbnail": data.get("thumbnail_url", "")}


def enrich_manifest(path: Path = MANIFEST) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            url = row.get("url", "").strip()
            if url and (not row.get("title") or not row.get("creator")):
                try:
                    meta = youtube_oembed(url)
                    row["title"] = row.get("title") or meta["title"]
                    row["creator"] = row.get("creator") or meta["author"]
                except Exception:
                    pass
            rows.append(row)
    return rows


def tokenize(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[A-Za-z0-9']+", text) if len(w) > 2]


def summarize(rows: list[dict]) -> dict:
    formats = Counter((r.get("format") or "unclassified").strip().lower() for r in rows)
    hooks = Counter((r.get("hook_type") or "unclassified").strip().lower() for r in rows)
    words = Counter()
    for r in rows:
        words.update(tokenize(r.get("title", "")))
    return {
        "videos": len(rows),
        "formats": formats.most_common(20),
        "hook_types": hooks.most_common(20),
        "title_words": words.most_common(40),
    }


def main() -> None:
    rows = enrich_manifest()
    print(json.dumps(summarize(rows), indent=2))


if __name__ == "__main__":
    main()
