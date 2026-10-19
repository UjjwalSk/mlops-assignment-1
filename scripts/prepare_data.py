"""Extract the supplied archive and create a validated training CSV."""
from pathlib import Path
import sys
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.heart_risk.data import load_processed_data, validate_frame

import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "heart+disease.zip"
RAW_DIR = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "heart_disease_cleveland.csv"
UCI_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"


def ensure_archive() -> None:
    if not ARCHIVE.exists():
        print(f"Archive {ARCHIVE.name} not found locally. Downloading from UCI Repository: {UCI_URL}...")
        urllib.request.urlretrieve(UCI_URL, ARCHIVE)
        print("Download complete.")


def main() -> None:
    ensure_archive()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with ZipFile(ARCHIVE) as archive:
        archive.extractall(RAW_DIR)
    source = RAW_DIR / "processed.cleveland.data"
    frame = load_processed_data(source)
    validate_frame(frame)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    print(f"Wrote {len(frame)} rows to {OUT}")
    print(f"Class counts: {frame['target'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
