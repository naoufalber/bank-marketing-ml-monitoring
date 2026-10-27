"""Download only the required public UCI CSV, without extracting arbitrary paths."""
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

URL = "https://archive.ics.uci.edu/static/public/222/bank%2Bmarketing.zip"
DESTINATION = Path(__file__).resolve().parents[1] / "data/bank-additional-full.csv"


def extract_csv(archive):
    with ZipFile(BytesIO(archive)) as outer:
        with ZipFile(BytesIO(outer.read("bank-additional.zip"))) as inner:
            return inner.read("bank-additional/bank-additional-full.csv")


def main():
    if DESTINATION.exists():
        print(f"Already available: {DESTINATION}")
        return
    with urlopen(URL, timeout=60) as response:
        payload = response.read()
    csv = extract_csv(payload)
    if not csv.startswith(b'"age";'):
        raise ValueError("Unexpected dataset format; inspect the UCI download")
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_bytes(csv)
    print(f"Downloaded {DESTINATION.name}; see data/README.md for attribution.")


if __name__ == "__main__":
    main()
