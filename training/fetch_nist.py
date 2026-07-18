"""Fetch NIST SP 800-series PDFs for harvest.py.

NIST publications live at predictable nvlpubs.nist.gov URLs and are richly
bookmarked (deep outlines + printed TOCs — good printed_page training data,
public domain). There is no directory listing, so this probes the URL space:
for each publication number it tries the highest revision first
(NIST.SP.800-<n>r5 .. r1, then unrevised, then the legacy filename) and keeps
the first hit. 404s are expected and cheap.

Usage:
    python training/fetch_nist.py -o corpus/nist              # SP 800-1..230
    python training/fetch_nist.py -o corpus/nist --start 100 --end 200
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REQUEST_DELAY_S = 1.0  # static CDN; gentler than the arXiv API's 3s
_USER_AGENT = "pdf-bookmarker-harvest/0.1"


def candidate_urls(number: int) -> list[str]:
    """Highest revision first, then unrevised, then the legacy path."""
    base = "https://nvlpubs.nist.gov/nistpubs/SpecialPublications"
    urls = [f"{base}/NIST.SP.800-{number}r{r}.pdf" for r in range(5, 0, -1)]
    urls.append(f"{base}/NIST.SP.800-{number}.pdf")
    urls.append(
        "https://nvlpubs.nist.gov/nistpubs/Legacy/SP/"
        f"nistspecialpublication800-{number}.pdf"
    )
    return urls


def try_download(url: str, retries: int = 3) -> bytes | None:
    """Fetch one candidate URL; None on 404 or persistent failure.

    nvlpubs occasionally stalls mid-transfer; a lost candidate must cost a
    retry and a log line, never the whole multi-hour run.
    """
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            print(f"  HTTP {exc.code} for {url.rsplit('/', 1)[-1]} "
                  f"(attempt {attempt}/{retries})", file=sys.stderr)
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            print(f"  network error for {url.rsplit('/', 1)[-1]} "
                  f"(attempt {attempt}/{retries}): {exc}", file=sys.stderr)
        else:
            return data if data.startswith(b"%PDF-") else None
        time.sleep(5 * attempt)
    return None


def fetch(start: int, end: int, out_dir: Path) -> tuple[int, int]:
    """Probe SP 800-<start>..<end>. Returns (downloaded, skipped_existing)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"
    downloaded = skipped = 0
    with open(manifest, "a", encoding="utf-8") as mf:
        for number in range(start, end + 1):
            dest = out_dir / f"sp800-{number}.pdf"
            if dest.exists():
                skipped += 1
                continue
            for url in candidate_urls(number):
                data = try_download(url)
                time.sleep(REQUEST_DELAY_S)
                if data is None:
                    continue
                dest.write_bytes(data)
                mf.write(json.dumps({
                    "publication": f"SP 800-{number}",
                    "url": url,
                    "file": dest.name,
                    "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }) + "\n")
                downloaded += 1
                print(f"fetched SP 800-{number} <- {url.rsplit('/', 1)[-1]}",
                      file=sys.stderr)
                break
    return downloaded, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--out", type=Path, required=True, help="output directory")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=230)
    args = parser.parse_args(argv)

    downloaded, skipped = fetch(args.start, args.end, args.out)
    print(f"downloaded {downloaded}, skipped {skipped} already present", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
