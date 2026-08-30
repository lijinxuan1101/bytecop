"""Re-fetch WildFake zips that ModelScope's snapshot_download silently drops.

Why this exists
---------------
``snapshot_download`` cannot finish a large WildFake zip. modelscope 1.39.1's
``hub/file_download.py`` resumes with a ``Range`` header, then checks
completeness against the ``Content-Length`` of the *last ranged response* --
which after a resume is only the remaining bytes, never the full file::

    total = int(content_length)      # remaining bytes after a resume
    ...
    if total != downloaded_length:   # therefore always unequal
        os.remove(temp_file.name)    # tens of GB thrown away
        raise FileDownloadError(...)

Two settings make a resume inevitable on a slow link: ``API_FILE_DOWNLOAD_TIMEOUT``
is hardcoded to 60 s (no env override), and ``Retry(total=5)`` counts *cumulative*
stalls for the whole file, never resetting after progress. A 50 GB file over a
~20 MB/s link runs for ~45 min; hitting five 60 s stalls is close to certain.

Result: on 2026-08-29 all 30 small zips arrived intact and all 9 zips over 50 GB
failed -- the ones that must resume. This downloader keeps its own ``.part``
file, resumes from its real length, retries indefinitely with capped backoff,
and verifies against the size in ``Content-Range`` (the authoritative total).

Usage
-----
    python scripts/refetch_wildfake.py --root ~/techjam/raw/WildFake
    python scripts/refetch_wildfake.py --root ... --dry-run   # just report gaps

By default every zip in the upstream manifest that is missing locally, or whose
size differs, is fetched. Pass ``--only SUBSTR`` to narrow that set.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.parse
from pathlib import Path

import requests

REPO_ID = "hy2628982280/WildFake"
REVISION = "master"
ENDPOINT = "https://modelscope.cn"
CHUNK_SIZE = 8 * 1024 * 1024
CONNECT_TIMEOUT = 15
READ_TIMEOUT = 300
BACKOFF_MAX = 120


def _human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}PB"


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _file_url(path: str) -> str:
    quoted = urllib.parse.quote(path, safe="")
    return (f"{ENDPOINT}/api/v1/datasets/{REPO_ID}/repo"
            f"?Revision={REVISION}&FilePath={quoted}")


def upstream_zips() -> dict[str, int]:
    """Map every upstream ``.zip`` path to its authoritative size in bytes."""
    from modelscope.hub.api import HubApi

    files = HubApi().get_dataset_files(
        repo_id=REPO_ID, revision=REVISION, recursive=True,
    )
    return {
        entry["Path"]: entry["Size"]
        for entry in files
        if entry.get("Type") != "tree" and entry["Path"].endswith(".zip")
    }


def _probe_total(session: requests.Session, url: str) -> int:
    """Full size from ``Content-Range`` on a one-byte ranged GET."""
    while True:
        try:
            response = session.get(
                url, headers={"Range": "bytes=0-0"}, stream=True,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            )
            response.raise_for_status()
            content_range = response.headers.get("Content-Range", "")
            response.close()
            if "/" in content_range:
                return int(content_range.rsplit("/", 1)[1])
            raise RuntimeError(f"no Content-Range header: {dict(response.headers)}")
        except Exception as exc:  # noqa: BLE001 -- probe retries on anything
            _log(f"    probe failed ({type(exc).__name__}: {exc}); retry in 10s")
            time.sleep(10)


def fetch_one(session: requests.Session, root: Path, path: str) -> bool:
    """Download one zip to ``root/path``, resuming until byte-complete."""
    dest = root / path
    part = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = _file_url(path)

    total = _probe_total(session, url)
    if dest.exists() and dest.stat().st_size == total:
        _log(f"  already complete ({_human(total)})")
        return True

    attempt = 0
    stall_mark = -1
    while True:
        have = part.stat().st_size if part.exists() else 0
        if have == total:
            break
        if have > total:
            _log(f"  .part exceeds upstream ({have} > {total}); restarting")
            part.unlink()
            have = 0

        attempt += 1
        _log(f"  attempt {attempt}: resuming at {_human(have)} / {_human(total)} "
             f"({100.0 * have / total:.1f}%)")

        try:
            response = session.get(
                url, headers={"Range": f"bytes={have}-"}, stream=True,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            )
            response.raise_for_status()
            if have and response.status_code != 206:
                _log(f"  server ignored Range (status {response.status_code}); "
                     f"restarting from 0")
                response.close()
                part.unlink(missing_ok=True)
                continue

            started = last_report = time.time()
            got = 0
            with open(part, "ab") as handle:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    got += len(chunk)
                    now = time.time()
                    if now - last_report >= 60:
                        current = have + got
                        speed = got / (now - started)
                        eta = (total - current) / speed if speed > 0 else 0
                        _log(f"    {_human(current)} / {_human(total)} "
                             f"({100.0 * current / total:.1f}%)  {_human(speed)}/s  "
                             f"ETA {eta / 3600:.1f}h")
                        last_report = now
            response.close()
        except Exception as exc:  # noqa: BLE001 -- any failure is retryable
            now_have = part.stat().st_size if part.exists() else 0
            # Back off only when an attempt achieved nothing at all.
            wait = min(BACKOFF_MAX, 5 * 2 ** min(attempt, 5)) if now_have == stall_mark else 5
            stall_mark = now_have
            _log(f"    {type(exc).__name__}: {exc}")
            _log(f"    have {_human(now_have)}; retry in {wait}s")
            time.sleep(wait)

    part.replace(dest)
    final = dest.stat().st_size
    ok = final == total
    _log(f"  {'OK' if ok else 'SIZE MISMATCH'}: {_human(final)} / {_human(total)}")
    return ok


def missing_zips(root: Path, upstream: dict[str, int]) -> list[str]:
    """Upstream zips that are absent locally or whose size disagrees."""
    gaps = []
    for path, size in sorted(upstream.items()):
        local = root / path
        if not local.exists() or local.stat().st_size != size:
            gaps.append(path)
    return gaps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--root", type=Path, required=True,
        help="Local WildFake snapshot root (the directory holding Images/).",
    )
    parser.add_argument(
        "--only", default=None,
        help="Restrict to upstream paths containing this substring.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report the gaps and exit without downloading.",
    )
    args = parser.parse_args()

    root = args.root.expanduser()
    _log(f"reading upstream manifest for {REPO_ID}")
    upstream = upstream_zips()
    gaps = missing_zips(root, upstream)
    if args.only:
        gaps = [p for p in gaps if args.only in p]

    _log(f"upstream has {len(upstream)} zips; {len(gaps)} need fetching "
         f"({sum(upstream[p] for p in gaps) / 1e9:.1f} GB)")
    for path in gaps:
        local = root / path
        state = "missing" if not local.exists() else f"{local.stat().st_size} bytes"
        _log(f"  {path}  ({state} vs upstream {upstream[path]})")

    if args.dry_run or not gaps:
        return 0

    session = requests.Session()
    session.headers["User-Agent"] = "wildfake-refetch/1.0"
    failed = []
    for index, path in enumerate(gaps, 1):
        _log(f"[{index}/{len(gaps)}] {path}")
        try:
            if not fetch_one(session, root, path):
                failed.append(path)
        except KeyboardInterrupt:
            _log("interrupted; .part files are kept for resume")
            return 130

    _log("")
    _log(f"complete: {len(gaps) - len(failed)}/{len(gaps)}")
    for path in failed:
        _log(f"  FAILED {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
