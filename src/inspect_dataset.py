import zipfile
from pathlib import Path

ZIP_PATH = Path("data/raw/rtsd-dataset.zip")

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    names = z.namelist()

print("Total files:", len(names))

print("\nTop-level files:")
for name in names[:30]:
    print(name)

print("\nPossible annotation files:")
for name in names:
    lower = name.lower()
    if lower.endswith((".csv", ".json", ".txt", ".xml")):
        print(name)

print("\nPossible image folders:")
folders = sorted(set("/".join(name.split("/")[:2]) for name in names if "/" in name))
for folder in folders[:100]:
    print(folder)