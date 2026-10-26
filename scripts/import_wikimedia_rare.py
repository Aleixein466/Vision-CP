"""Fill remaining rare-class candidate gaps with licensed Wikimedia Commons files."""
from __future__ import annotations

import base64
import html
import json
import re
import time
import urllib.parse
import urllib.request
import urllib.error
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

try:
    from dataset_studio import DatasetStudio
except ModuleNotFoundError:
    from scripts.dataset_studio import DatasetStudio

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "datasets" / "reports" / "wikimedia_rare_import.json"
API = "https://commons.wikimedia.org/w/api.php"
SEARCHES = {
    "nail_clipper": ["nail clipper", "nail clippers manicure"],
    "payment_card": ["credit card", "payment card bank card"],
    "keys": ["house keys keyring", "metal keys"],
    "eraser": ["eraser stationery", "rubber eraser"],
    "sweatshirt": ["sweatshirt clothing", "hooded sweatshirt"],
    "usb_drive": ["USB flash drive", "USB memory stick"],
}
ALLOWED_LICENSE_PARTS = ("cc by", "cc-by", "cc0", "public domain", "public-domain")


def plain(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(value or ""))).strip()


def api_search(query: str, pages: int = 3) -> list[dict]:
    results, continuation = [], {}
    for _ in range(pages):
        params = {
            "action": "query", "format": "json", "formatversion": "2", "generator": "search",
            "gsrsearch": query, "gsrnamespace": "6", "gsrlimit": "50", "prop": "imageinfo",
            "iiprop": "url|mime|mediatype|extmetadata|sha1", "iiurlwidth": "1600",
            "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist|Credit|ImageDescription",
            "origin": "*", **continuation,
        }
        url = API + "?" + urllib.parse.urlencode(params)
        request = urllib.request.Request(url, headers={"User-Agent": "VisionAR-AI/1.0 dataset research (local educational project)"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                if exc.code != 429 or attempt == 4:
                    raise
                time.sleep(5 * (attempt + 1))
        results.extend(payload.get("query", {}).get("pages", []))
        if "continue" not in payload:
            break
        continuation = payload["continue"]
        time.sleep(1)
    return results


def candidate_from_page(page: dict, class_name: str) -> dict | None:
    info = (page.get("imageinfo") or [{}])[0]
    if info.get("mediatype") != "BITMAP" or info.get("mime") not in {"image/jpeg", "image/png", "image/webp"}:
        return None
    metadata = {key: value.get("value", "") for key, value in info.get("extmetadata", {}).items()}
    license_name = plain(metadata.get("LicenseShortName", ""))
    if not any(part in license_name.lower() for part in ALLOWED_LICENSE_PARTS):
        return None
    image_url = info.get("thumburl") or info.get("url")
    if not image_url:
        return None
    return {"class_name": class_name, "title": page.get("title", ""), "url": image_url,
            "page_url": info.get("descriptionurl", ""), "mime": info.get("thumbmime") or info.get("mime"),
            "license": f"{license_name} | {plain(metadata.get('LicenseUrl', ''))}",
            "artist": plain(metadata.get("Artist", "") or metadata.get("Credit", ""))[:180],
            "description": plain(metadata.get("ImageDescription", ""))[:240], "sha1": info.get("sha1")}


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "VisionAR-AI/1.0 dataset research"})
    last_error = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                raw = response.read(12 * 1024 * 1024 + 1)
            if len(raw) > 12 * 1024 * 1024:
                raise ValueError("imagen mayor de 12 MB")
            return raw
        except Exception as exc:
            last_error = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 429:
                retry_after = int(exc.headers.get("Retry-After", 10))
                time.sleep(max(retry_after, 10 * (attempt + 1)))
            else:
                time.sleep(2 + attempt)
    raise RuntimeError(str(last_error))


def main() -> None:
    studio = DatasetStudio()
    existing_source_text = "\n".join(record.get("source", "") for record in studio.list_candidates())
    before = studio.stats()
    deficits = {name: max(0, 30 - before["per_class"][name]["pending"] - before["per_class"][name]["reviewed"])
                for name in SEARCHES}
    deficits = {name: value for name, value in deficits.items() if value}
    gathered: dict[str, list[dict]] = {name: [] for name in deficits}
    seen_sha1 = set()
    for name, needed in deficits.items():
        for query in SEARCHES[name]:
            for page in api_search(query):
                item = candidate_from_page(page, name)
                if not item or item["sha1"] in seen_sha1 or item["page_url"] in existing_source_text:
                    continue
                seen_sha1.add(item["sha1"])
                gathered[name].append(item)
                if len(gathered[name]) >= needed + 10:
                    break
            if len(gathered[name]) >= needed + 10:
                break
        print(f"{name}: {len(gathered[name])} archivos licenciados encontrados para déficit {needed}", flush=True)
        time.sleep(2)
    work = [(name, item) for name, items in gathered.items() for item in items[:deficits[name]]]
    counters, errors, accepted = Counter(), [], Counter()

    def import_one(name: str, item: dict) -> dict:
        raw = download(item["url"])
        time.sleep(1)
        data_url = f"data:{item['mime']};base64," + base64.b64encode(raw).decode("ascii")
        source = f"Wikimedia Commons; {item['page_url']}; artist={item['artist']}; description={item['description']}"
        result = studio.import_candidate(data_url, item["title"].removeprefix("File:"), source,
                                         item["license"], reject_near=True)
        return {"result": result, "name": name}

    with ThreadPoolExecutor(max_workers=1) as pool:
        futures = {pool.submit(import_one, name, item): (name, item) for name, item in work}
        for index, future in enumerate(as_completed(futures), 1):
            name, item = futures[future]
            try:
                outcome = future.result()
                result = outcome["result"]
                if result["imported"]:
                    studio.set_intended_classes(result["candidate"]["id"], [name])
                    accepted[name] += 1
                    counters["imported"] += 1
                else:
                    counters["duplicate"] += 1
            except Exception as exc:
                counters["errors"] += 1
                errors.append({"class": name, "title": item["title"], "error": str(exc)})
            if index % 25 == 0 or index == len(futures):
                print(f"{index}/{len(futures)} | importadas={counters['imported']} errores={counters['errors']}", flush=True)
    after = studio.stats()
    report = {"generated_at": datetime.now().astimezone().isoformat(),
              "source": "Wikimedia Commons MediaWiki API", "api": API,
              "deficits_before": deficits, "licensed_results_found": {n: len(v) for n, v in gathered.items()},
              "accepted_by_class": dict(accepted), "result": dict(counters), "errors": errors,
              "stats_after": after}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"accepted": dict(accepted), "result": dict(counters),
                      "pending": after["pending_annotations"], "report": str(REPORT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
