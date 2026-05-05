"""MarkdownStore — append-only markdown entries on disk.

Layout: <root>/entries/<YYYY>/<MM>/<DD>/<timestamp>_<runtime>_<short>.md

Append-only by design — entries are never updated in place. If you need to
correct something, write a new entry that supersedes the old one and have
your reader respect supersedence.
"""
from __future__ import annotations

import datetime as _dt
import os
import re
import secrets
from pathlib import Path
from typing import Iterable

from hyperswarm.core.entry import Entry
from hyperswarm.core.store import Store


class MarkdownStore(Store):
    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        root = self.config.get("path", "~/HyperSwarm")
        self.root = Path(os.path.expanduser(root))
        self.entries_dir = self.root / "entries"

    def write(self, entry: Entry) -> str:
        ts = entry.timestamp
        day_dir = self.entries_dir / f"{ts.year:04d}" / f"{ts.month:02d}" / f"{ts.day:02d}"
        day_dir.mkdir(parents=True, exist_ok=True)
        short = secrets.token_hex(3)
        ts_str = ts.strftime("%H%M%S")
        runtime_slug = re.sub(r"[^A-Za-z0-9._-]", "-", entry.runtime or "unknown")
        path = day_dir / f"{ts_str}_{runtime_slug}_{short}.md"
        path.write_text(entry.to_markdown())
        return str(path)

    def read(self, storage_id: str) -> Entry:
        return Entry.from_markdown(Path(storage_id).read_text())

    def list_since(self, since: _dt.datetime) -> Iterable[Entry]:
        if not self.entries_dir.exists():
            return
        # Walk only the day-dirs whose date is >= since.date() to avoid scanning
        # years of cold storage on a "last 24h" query.
        cutoff_date = since.date()
        for year_dir in sorted(self.entries_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            for month_dir in sorted(year_dir.iterdir()):
                if not month_dir.is_dir():
                    continue
                for day_dir in sorted(month_dir.iterdir()):
                    if not day_dir.is_dir():
                        continue
                    try:
                        d = _dt.date(int(year_dir.name), int(month_dir.name), int(day_dir.name))
                    except ValueError:
                        continue
                    if d < cutoff_date:
                        continue
                    for f in sorted(day_dir.iterdir()):
                        if f.suffix != ".md":
                            continue
                        try:
                            entry = Entry.from_markdown(f.read_text())
                        except Exception:
                            continue
                        if entry.timestamp >= since:
                            yield entry
