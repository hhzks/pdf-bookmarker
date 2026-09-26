"""Fetch bookmarked web PDFs from CC-MAIN-2021-31-PDF-UNTRUNCATED for harvest.py.

The corpus (Digital Corpora / DARPA SafeDocs) is ~7.9M PDFs from a 2021 Common
Crawl, packed 1,000 to a ZIP: zip N holds N*1000.pdf .. N*1000+999.pdf. It is
the widest source of producers here — Word, InDesign, Acrobat, report
generators — where arXiv and NIST are 87% LaTeX between them.

A ZIP is 1-2.8 GB and most web PDFs carry no outline, so this never downloads
one whole: it reads the ZIP's central directory with HTTP range requests and
fetches members one at a time. Each fetched PDF must pass the same gate as
harvest.py (unencrypted, >= --min-pages, a non-empty get_toc(), a text layer)
or it is discarded without being written.

--metadata points at the corpus's Tika table (tika-20230714.csv.gz, 450 MB,
under metadata/ next to zipfiles/). With it, members that cannot pass — too
short, encrypted, no extracted text, LaTeX-produced — are skipped before any
PDF bytes move. Without it every member is fetched and gated locally.

LaTeX producers are skipped by default because the corpus already has 652 of
them; --include-tex keeps them.

Usage:
    python training/fetch_ccpdf.py --zips 0 -o corpus/ccpdf --max 50
    python training/fetch_ccpdf.py --zips 0-3,17 -o corpus/ccpdf \\
        --metadata corpus/tika-20230714.csv.gz --lang eng

Every member tried is appended to manifest.jsonl with "kept" and a reason, so a
re-run skips members already decided instead of fetching them again.
NOTE: these are crawled web documents under Common Crawl's terms of use, not an
open license; keep the corpus private and check a file's source URL (url_id ->
cc-provenance table) before redistributing it.
"""
import argparse
import csv
import gzip
import io
import json
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitz

from pdf_bookmarker import extractor

BASE_URL = ("https://digitalcorpora.s3.amazonaws.com/corpora/files/"
            "CC-MAIN-2021-31-PDF-UNTRUNCATED")
LAST_ZIP = 7932
_USER_AGENT = "pdf-bookmarker-harvest/0.1"
# pdfTeX, XeTeX, LuaHBTeX, dvips, dvipdfmx — "tex\b" so "Textmaker" is not one.
_TEX_RE = re.compile(r"tex\b|latex|dvips|dvipdfm", re.IGNORECASE)
_MIN_ALPHA_TOKENS = 200  # Tika's count; a scan or an image-only PDF has ~0


def zip_url(index: int) -> str:
    low = index // 1000 * 1000
    return f"{BASE_URL}/zipfiles/{low:04d}-{low + 999:04d}/{index:04d}.zip"


def parse_zip_spec(spec: str) -> list[int]:
    """"0-3,17" -> [0, 1, 2, 3, 17]."""
    indices: set[int] = set()
    for part in spec.split(","):
        first, _, last = part.strip().partition("-")
        indices.update(range(int(first), int(last or first) + 1))
    bad = [i for i in indices if not 0 <= i <= LAST_ZIP]
    if bad:
        raise ValueError(f"zip index out of range 0..{LAST_ZIP}: {sorted(bad)}")
    return sorted(indices)


def is_tex(producer: str) -> bool:
    return bool(_TEX_RE.search(producer))


class HttpRangeFile(io.RawIOBase):
    """A read-only, seekable view of a remote file, one Range GET per read.

    zipfile needs only seek/tell/read, so wrapping this lets it list a
    multi-GB archive and extract single members without downloading the rest.
    """

    def __init__(self, url: str, retries: int = 3):
        self.url = url
        self.retries = retries
        self._pos = 0
        request = urllib.request.Request(
            url, method="HEAD", headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response:
            self.size = int(response.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self.size}[whence]
        self._pos = max(0, base + offset)
        return self._pos

    def readinto(self, buffer) -> int:
        if self._pos >= self.size:
            return 0
        end = min(self._pos + len(buffer), self.size) - 1
        data = self._get_range(self._pos, end)
        buffer[:len(data)] = data
        self._pos += len(data)
        return len(data)

    def _get_range(self, start: int, end: int) -> bytes:
        """S3 occasionally resets a long transfer; a lost range costs a retry,
        never the whole ZIP."""
        request = urllib.request.Request(
            self.url, headers={"User-Agent": _USER_AGENT,
                               "Range": f"bytes={start}-{end}"})
        for attempt in range(1, self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    return response.read()
            except (TimeoutError, urllib.error.URLError, OSError) as exc:
                if attempt == self.retries:
                    raise
                print(f"  range {start}-{end} failed (attempt {attempt}/"
                      f"{self.retries}): {exc}", file=sys.stderr)
                time.sleep(5 * attempt)
        raise AssertionError("unreachable")


def open_remote_zip(index: int) -> zipfile.ZipFile:
    # A small buffer: zipfile's 30-byte local-header reads would otherwise
    # pull a large block of the next member that the following seek discards.
    raw = HttpRangeFile(zip_url(index))
    return zipfile.ZipFile(io.BufferedReader(raw, buffer_size=64 * 1024))


def load_metadata(path: Path, zips: set[int]) -> dict[str, dict]:
    """Tika rows for the requested ZIPs only, keyed by member name."""
    opener = gzip.open if path.suffix == ".gz" else open
    rows: dict[str, dict] = {}
    with opener(path, "rt", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            name = row["file_name"]
            if int(name.split(".")[0]) // 1000 in zips:
                rows[name] = row
    return rows


def metadata_reject(row: dict, *, min_pages: int, include_tex: bool,
                    lang: str | None) -> str | None:
    """Why a member cannot pass the harvest gate, or None if it might."""
    if row.get("parse_status") != "OK":
        return "tika-parse-failed"
    if row.get("encrypted") == "t":
        return "encrypted"
    if int(row.get("num_pages") or 0) < min_pages:
        return "too-short"
    if int(row.get("tika_eval_num_alpha_tokens") or 0) < _MIN_ALPHA_TOKENS:
        return "no-text"
    if lang and row.get("tika_eval_lang") != lang:
        return "language"
    producer = f"{row.get('pdf_producer', '')} {row.get('xmp_creator_tool', '')}"
    if not include_tex and is_tex(producer):
        return "tex"
    return None


def gate(data: bytes, *, min_pages: int, include_tex: bool) -> tuple[str | None, dict]:
    """harvest.py's own gate, applied before the file is written.

    Returns (reject reason or None, provenance fields worth recording).
    """
    if not data.startswith(b"%PDF-"):
        return "not-pdf", {}
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception:
        return "unreadable", {}
    with doc:
        if doc.needs_pass:
            return "encrypted", {}
        meta = doc.metadata or {}
        producer = f"{meta.get('producer') or ''} {meta.get('creator') or ''}".strip()
        info = {"producer": producer, "page_count": doc.page_count}
        if doc.page_count < min_pages:
            return "too-short", info
        if not include_tex and is_tex(producer):
            return "tex", info
        toc = doc.get_toc()
        if not toc:
            return "no-outline", info
        info["outline_entries"] = len(toc)
        if not extractor.has_text_layer(doc):
            return "no-text", info
    return None, info


def _already_tried(manifest: Path) -> set[str]:
    if not manifest.exists():
        return set()
    with open(manifest, encoding="utf-8") as fh:
        return {json.loads(line)["member"] for line in fh if line.strip()}


def fetch(zips: list[int], out_dir: Path, *, max_kept: int, min_pages: int,
          include_tex: bool, metadata: dict[str, dict] | None,
          lang: str | None) -> tuple[int, int]:
    """Returns (kept, rejected) for this run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"
    tried = _already_tried(manifest)
    kept = rejected = 0
    with open(manifest, "a", encoding="utf-8") as mf:
        def record(member: str, index: int, reason: str | None, **fields) -> None:
            mf.write(json.dumps({
                "member": member, "zip": index, "kept": reason is None,
                "reason": reason, **fields,
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, ensure_ascii=False) + "\n")
            mf.flush()

        for index in zips:
            print(f"zip {index:04d}: reading directory", file=sys.stderr)
            try:
                archive = open_remote_zip(index)
            except Exception as exc:
                print(f"  cannot open zip {index:04d}: {exc}", file=sys.stderr)
                continue
            with archive:
                for info in archive.infolist():
                    if kept >= max_kept:
                        return kept, rejected
                    member = Path(info.filename).name
                    if not member.endswith(".pdf") or member in tried:
                        continue
                    row = metadata.get(member) if metadata is not None else None
                    if metadata is not None:
                        reason = ("no-metadata" if row is None else metadata_reject(
                            row, min_pages=min_pages, include_tex=include_tex, lang=lang))
                        if reason:
                            # Not recorded: a re-run with other filters should
                            # reconsider it, and it cost no download.
                            continue
                    try:
                        data = archive.read(info)
                    except Exception as exc:
                        print(f"  {member}: read failed: {exc}", file=sys.stderr)
                        continue
                    reason, fields = gate(data, min_pages=min_pages,
                                          include_tex=include_tex)
                    if row is not None:
                        fields.update(url_id=row.get("url_id"),
                                      lang=row.get("tika_eval_lang"))
                    if reason is None:
                        (out_dir / f"cc-{member}").write_bytes(data)
                        kept += 1
                        print(f"  kept {member} ({fields['outline_entries']} "
                              f"bookmarks, {fields['producer'][:50]})", file=sys.stderr)
                    else:
                        rejected += 1
                    record(member, index, reason, **fields)
                    tried.add(member)
    return kept, rejected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zips", required=True,
                        help=f'ZIP indices 0..{LAST_ZIP}, e.g. "0" or "0-3,17"')
    parser.add_argument("-o", "--out", type=Path, required=True, help="output directory")
    parser.add_argument("--max", type=int, default=100, dest="max_kept",
                        help="stop after keeping this many PDFs (default 100)")
    parser.add_argument("--min-pages", type=int, default=4,
                        help="same default as harvest.py")
    parser.add_argument("--include-tex", action="store_true",
                        help="keep LaTeX-produced PDFs (skipped by default)")
    parser.add_argument("--metadata", type=Path,
                        help="local tika-20230714.csv(.gz): pre-filter before download")
    parser.add_argument("--lang", help='Tika language code, e.g. "eng" (needs --metadata)')
    args = parser.parse_args(argv)
    if args.lang and not args.metadata:
        parser.error("--lang needs --metadata")

    zips = parse_zip_spec(args.zips)
    metadata = None
    if args.metadata:
        print(f"loading {args.metadata} for {len(zips)} zip(s)", file=sys.stderr)
        metadata = load_metadata(args.metadata, set(zips))
    kept, rejected = fetch(zips, args.out, max_kept=args.max_kept,
                           min_pages=args.min_pages, include_tex=args.include_tex,
                           metadata=metadata, lang=args.lang)
    print(f"kept {kept}, rejected {rejected}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
