"""Fetch a small seed corpus of arXiv PDFs for harvest.py.

arXiv PDFs compiled with hyperref frequently carry an embedded outline, which
harvest.py turns into labeled training examples (the rest are filtered out by
its get_toc() gate, so over-fetching is fine).

Polite by design: one HTTP request every 3 seconds, per arXiv API guidelines.
Already-downloaded files are skipped, so the command is resumable.

Usage:
    python training/fetch_arxiv.py --query "cat:math.LO" --max 25 -o corpus/arxiv
    python training/fetch_arxiv.py --query 'abs:"lecture notes"' --max 50 -o corpus/arxiv

A manifest.jsonl with per-file provenance (arXiv id, title, fetch date) is
appended in the output directory. NOTE: the arXiv API does not return license
info; check https://arxiv.org/abs/<id> before redistributing, and prefer
CC-licensed papers for a training corpus you intend to publish.
"""
import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

API_URL = "https://export.arxiv.org/api/query"
REQUEST_DELAY_S = 3.0
PAGE_SIZE = 100
_ATOM = "{http://www.w3.org/2005/Atom}"
_USER_AGENT = "pdf-bookmarker-harvest/0.1"


def parse_feed(xml_text: str) -> list[dict]:
    """Parse an arXiv API Atom feed into paper dicts."""
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall(f"{_ATOM}entry"):
        abs_url = entry.findtext(f"{_ATOM}id", "").strip()
        if not abs_url:
            continue
        pdf_url = None
        for link in entry.findall(f"{_ATOM}link"):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href")
                break
        if pdf_url is None:
            pdf_url = abs_url.replace("/abs/", "/pdf/")
        papers.append(
            {
                "arxiv_id": abs_url.rsplit("/abs/", 1)[-1],
                "title": " ".join(entry.findtext(f"{_ATOM}title", "").split()),
                "published": entry.findtext(f"{_ATOM}published", "").strip(),
                "pdf_url": pdf_url,
            }
        )
    return papers


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def search(query: str, max_results: int) -> list[dict]:
    """Page through the arXiv API and return up to max_results papers."""
    papers: list[dict] = []
    start = 0
    while len(papers) < max_results:
        params = urllib.parse.urlencode(
            {
                "search_query": query,
                "start": start,
                "max_results": min(PAGE_SIZE, max_results - len(papers)),
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
        )
        batch = parse_feed(_get(f"{API_URL}?{params}").decode("utf-8"))
        if not batch:
            break
        papers.extend(batch)
        start += len(batch)
        time.sleep(REQUEST_DELAY_S)
    return papers[:max_results]


def _safe_filename(arxiv_id: str) -> str:
    # Old-style ids contain a slash (e.g. "math/0309136v1").
    return arxiv_id.replace("/", "_") + ".pdf"


def fetch(query: str, max_results: int, out_dir: Path) -> tuple[int, int]:
    """Download PDFs for a query. Returns (downloaded, skipped_existing)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"
    papers = search(query, max_results)
    downloaded = skipped = 0
    with open(manifest, "a", encoding="utf-8") as mf:
        for paper in papers:
            dest = out_dir / _safe_filename(paper["arxiv_id"])
            if dest.exists():
                skipped += 1
                continue
            print(f"fetching {paper['arxiv_id']}: {paper['title'][:70]}", file=sys.stderr)
            try:
                dest.write_bytes(_get(paper["pdf_url"]))
            except Exception as exc:
                print(f"  failed: {exc}", file=sys.stderr)
                continue
            mf.write(
                json.dumps(
                    {
                        **paper,
                        "file": dest.name,
                        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            downloaded += 1
            time.sleep(REQUEST_DELAY_S)
    return downloaded, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--query",
        required=True,
        help='arXiv API search query, e.g. "cat:math.LO" or \'abs:"lecture notes"\'',
    )
    parser.add_argument("--max", type=int, default=25, dest="max_results")
    parser.add_argument("-o", "--out", type=Path, required=True, help="output directory")
    args = parser.parse_args(argv)

    downloaded, skipped = fetch(args.query, args.max_results, args.out)
    print(f"downloaded {downloaded}, skipped {skipped} already present", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
