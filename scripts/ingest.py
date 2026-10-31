"""Build a knowledge-base JSONL from your own documents.

Supported inputs
  *.txt / *.md   one document per file
  *.pdf          text-based PDFs (needs `pip install pypdf`)
  *.csv          one document per row; choose the text column with --text-column

Examples
  python scripts/ingest.py docs/mevzuat --kind mevzuat -o data/corpus.jsonl
  python scripts/ingest.py dilekceler.csv --text-column target_text --kind ictihat -o data/corpus.jsonl --append

Then point the app at it:  CORPUS_PATH=data/corpus.jsonl python -m app.server
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.corpus import KINDS, Passage, chunk_text, load_jsonl, save_jsonl  # noqa: E402


def read_documents(path: Path, text_column: str | None):
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    for f in files:
        suffix = f.suffix.lower()
        if suffix in (".txt", ".md"):
            yield f.stem, f.read_text(encoding="utf-8", errors="ignore")
        elif suffix == ".pdf":
            from pypdf import PdfReader

            yield f.stem, "\n".join(page.extract_text() or "" for page in PdfReader(f).pages)
        elif suffix == ".csv":
            with open(f, encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                column = text_column or (reader.fieldnames or [""])[0]
                for i, row in enumerate(reader, 1):
                    yield f"{f.stem} #{i}", row.get(column) or ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="file or folder to ingest")
    ap.add_argument("-o", "--output", type=Path, default=Path("data/corpus.jsonl"))
    ap.add_argument("--kind", choices=KINDS, default="mevzuat")
    ap.add_argument("--text-column", help="CSV column holding the text")
    ap.add_argument("--max-chars", type=int, default=900)
    ap.add_argument("--append", action="store_true", help="add to an existing output file")
    args = ap.parse_args()

    passages: list[Passage] = load_jsonl(args.output) if args.append and args.output.exists() else []
    seen = {p.id for p in passages}
    added = 0
    for title, text in read_documents(args.source, args.text_column):
        for i, chunk in enumerate(chunk_text(text, args.max_chars), 1):
            pid = hashlib.sha1(f"{title}|{i}|{chunk[:64]}".encode()).hexdigest()[:12]
            if pid in seen:
                continue
            seen.add(pid)
            passages.append(Passage(id=pid, kind=args.kind, title=title, ref=f"{title} (bölüm {i})", text=chunk))
            added += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = save_jsonl(passages, args.output)
    print(f"Added {added} passages; {args.output} now holds {total}.")


if __name__ == "__main__":
    main()
