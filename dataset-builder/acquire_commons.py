from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "GolfIQ-DatasetBuilder/0.2 (legal media acquisition; contact via repository)"
STRICT_ALLOWED = {
    "cc0",
    "cc0 1.0",
    "public domain",
    "cc by 2.0",
    "cc by 3.0",
    "cc by 4.0",
}


def _open_with_retry(req: urllib.request.Request, timeout: int, attempts: int = 7):
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 500, 502, 503, 504}:
                raise
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            if retry_after and retry_after.isdigit():
                wait = min(60.0, float(retry_after))
            else:
                wait = min(60.0, 2.0 ** attempt + random.uniform(0.5, 2.0))
            print(f"HTTP {exc.code}; backing off {wait:.1f}s", flush=True)
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            wait = min(45.0, 2.0 ** attempt + random.uniform(0.5, 2.0))
            print(f"Network error; retrying in {wait:.1f}s: {exc}", flush=True)
            time.sleep(wait)
    assert last_error is not None
    raise last_error


def api(params: dict[str, str]) -> dict:
    query = urllib.parse.urlencode({"format": "json", "formatversion": "2", **params})
    req = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": USER_AGENT})
    with _open_with_retry(req, timeout=45) as response:
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
        time.sleep(0.8)
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


def download(url: str, destination: Path) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return True
    temp = destination.with_suffix(destination.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with _open_with_retry(req, timeout=90) as response, temp.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        temp.replace(destination)
        time.sleep(random.uniform(0.5, 1.2))
        return True
    except Exception as exc:
        temp.unlink(missing_ok=True)
        print(f"Skipping download after retries: {url} ({exc})", flush=True)
        return False


def safe_filename(title: str) -> str:
    name = title.removeprefix("File:")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def acquire(queries: list[str], output: Path, limit_per_query: int, download_media: bool) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    accepted: list[dict] = []
    rejected: list[dict] = []
    failed_downloads: list[dict] = []

    for query in queries:
        print(f"Searching Commons: {query}", flush=True)
        try:
            titles = search_files(query, limit_per_query)
        except Exception as exc:
            print(f"Skipping query after retries: {query} ({exc})", flush=True)
            continue
        for title in titles:
            if title in seen:
                continue
            seen.add(title)
            try:
                info = file_info(title)
            except Exception as exc:
                rejected.append({"title": title, "license": "lookup_failed", "reason": str(exc)})
                continue
            if not info:
                continue
            if info["allowed"]:
                info["search_query"] = query
                if download_media and info.get("url"):
                    ok = download(info["url"], output / "media" / safe_filename(title))
                    info["downloaded"] = ok
                    if not ok:
                        failed_downloads.append({"title": title, "url": info["url"]})
                accepted.append(info)
            else:
                rejected.append({"title": title, "license": info["license"], "commons_page": info["commons_page"]})
            time.sleep(random.uniform(0.35, 0.8))

        # Save progress after every query so an interrupted run can still be inspected/resumed.
        (output / "accepted.json").write_text(json.dumps(accepted, indent=2), encoding="utf-8")
        (output / "rejected.json").write_text(json.dumps(rejected, indent=2), encoding="utf-8")
        (output / "failed_downloads.json").write_text(json.dumps(failed_downloads, indent=2), encoding="utf-8")
        time.sleep(random.uniform(2.0, 4.0))

    with (output / "attribution.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["title", "commons_page", "license", "license_url", "artist", "credit", "sha1", "search_query"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in accepted:
            writer.writerow({key: row.get(key, "") for key in fields})

    return {
        "accepted": len(accepted),
        "downloaded": sum(1 for row in accepted if row.get("downloaded", not download_media)),
        "failed_downloads": len(failed_downloads),
        "rejected": len(rejected),
        "unique_examined": len(seen),
    }


DEFAULT_QUERIES = [
    "golf ball",
    "golf ball grass",
    "golf ball tee",
    "golf driving range",
    "golf practice range",
    "golf swing",
    "golf swing slow motion",
    "golf club impact",
    "golf tee",
    "golf flagstick",
    "golf range marker",
    "golf course sky",
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
