"""Download the RTSD archive from Kaggle unless it is already present.

Auth: ``~/.kaggle/access_token`` (or ``KAGGLE_API_TOKEN``), or the legacy
``~/.kaggle/kaggle.json`` / ``KAGGLE_USERNAME`` + ``KAGGLE_KEY``.
"""

import zipfile

from config import KAGGLE_DATASET, RAW_DIR, RAW_ZIP_PATH

REQUIRED_MEMBERS = ("label_map.json", "train_anno.json", "val_anno.json")


class IncompleteArchiveError(RuntimeError):
    pass


def archive_is_complete() -> bool:
    if not RAW_ZIP_PATH.exists():
        return False
    try:
        with zipfile.ZipFile(RAW_ZIP_PATH) as archive:
            names = set(archive.namelist())
    except zipfile.BadZipFile:
        return False
    return all(member in names for member in REQUIRED_MEMBERS)


def download_archive() -> None:
    # Imported lazily: the kaggle package authenticates on import.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(KAGGLE_DATASET, path=str(RAW_DIR), unzip=False, quiet=False)


def main() -> None:
    if archive_is_complete():
        print(f"Archive already present: {RAW_ZIP_PATH}")
        return
    print(f"Downloading {KAGGLE_DATASET} (~18.4 GB) to {RAW_DIR}")
    download_archive()
    if not archive_is_complete():
        raise IncompleteArchiveError(f"Downloaded archive is incomplete: {RAW_ZIP_PATH}")
    print("Download finished.")


if __name__ == "__main__":
    main()
