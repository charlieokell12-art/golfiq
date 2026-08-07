from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "GolfIQ-DatasetBuilder/0.1 (legal media acquisition; contact via repository)"
STRICT_ALLOWED = {
    "cc0",
    "cc0 1.0",
    "public domain",
    "cc by 2.0",
    "cc by 3.0",
    "cc by 4.0",
}


def api(params: dict[str, str]) -> dict:
    query = urllib.parse.urlencode({"format": "json", "formatversion": "2", **params})
    req = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=45) as response:
        return json.load(response)


def clean_html(value: str | None) -> str:
    return re.sub(r"<[^>]+>", " ", value or "").replace("&nbsp;", " ").strip()


def normalized_license(meta: dict) -> str:
    short = clean_html(meta.get("LicenseShortName", {}).get("value"))
    usage = clean_html(meta.get("UsageTerms", {}).get("value"))
    return (short or usage).lower().replace("creative commons ", "cc ").strip()


def is_allowed(license_name: str) -> bool:
    name = " ".join(license_name.lower().split())
    return any(token in name for token in STRICT_ALLOWED)


def search_files(query: str, limit: int = 200) -> list[str]:
    titles: list[str] = []
    offset = 0
    while len(titles) < limit:
        data = api({
            "action": "query",
            "list": "search",
            "srnamespace": "6",
            "srsearch": query,
            "srlimit": str(min(50, limit - len(titles))),
            "sroffset": str(offset),
        })
        batch = data.get("query", {}).get("search", [])
        if not batch:
            break
        titles.extend(item["title"] for item in batch)
        offset += len(batch)
        if len(batch) < 50:
            break
        time.sleep(0.1)
    return titles[:limit]


def file_info(title: str) -> dict | None:
    data = api({
        "action": "query",
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url|mime|size|extmetadata|sha1",
    })
    pages = data.get("query", {}).get("pages", [])
    if not pages or "imageinfo" not in pages[0]:
        return None
    info = pages[0]["imageinfo"][0]
    meta = info.get("extmetadata", {})
    license_name = normalized_license(meta)
    return {
        "title": title,
        "url": info.get("url"),
        "mime": info.get("mime", ""),
        "width": info.get("width"),
        "height": info.get("height"),
        "sha1": info.get("sha1"),
        "license": license_name,
        "license_url": clean_html(meta.get("LicenseUrl", {}).get("value")),
        "artist": clean_html(meta.get("Artist", {}).get("value")),
        "credit": clean_html(meta.get("Credit", {}).get("value")),
        "description": clean_html(meta.get("ImageDescription", {}).get("value")),
        "commons_page": "https://commons.wikimedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"), safe=":()_-"),
        "allowed": is_allowed(license_name),
    }


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=90) as response, destination.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def safe_filename(title: str) -> str:
    name = title.removeprefix("File:")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def acquire(queries: list[str], output: Path, limit_per_query: int, download_media: bool) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    accepted: list[dict] = []
    rejected: list[dict] = []

    for query in queries:
        for title in search_files(query, limit_per_query):
            if title in seen:
                continue
            seen.add(title)
            info = file_info(title)
            if not info:
                continue
            if info["allowed"]:
                info["search_query"] = query
                accepted.append(info)
                if download_media and info.get("url"):
                    download(info["url"], output / "media" / safe_filename(title))
            else:
                rejected.append({"title": title, "license": info["license"], "commons_page": info["commons_page"]})
            time.sleep(0.05)

    (output / "accepted.json").write_text(json.dumps(accepted, indent=2), encoding="utf-8")
    (output / "rejected.json").write_text(json.dumps(rejected, indent=2), encoding="utf-8")
    with (output / "attribution.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["title", "commons_page", "license", "license_url", "artist", "credit", "sha1", "search_query"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in accepted:
            writer.writerow({key: row.get(key, "") for key in fields})
    return {"accepted": len(accepted), "rejected": len(rejected), "unique_examined": len(seen)}


DEFAULT_QUERIES = [
    'golf ball',
    'golf ball grass',
    'golf ball tee',
    'golf driving range',
    'golf practice range',
    'golf swing',
    'golf swing slow motion',
    'golf club impact',
    'golf tee',
    'golf flagstick',
    'golf range marker',
    'golf course sky',
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire commercially usable Wikimedia Commons media with license evidence.")
    parser.add_argument("--output", type=Path, default=Path("datasets/commons-v1"))
    parser.add_argument("--limit-per-query", type=int, default=150)
    parser.add_argument("--query", action="append", default=[])
    parser.add_argument("--download", action="store_true", help="Download accepted media; otherwise only build manifests.")
    args = parser.parse_args()
    queries = args.query or DEFAULT_QUERIES
    print(json.dumps(acquire(queries, args.output, args.limit_per_query, args.download), indent=2))


if __name__ == "__main__":
    main()
