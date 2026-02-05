import os
from pathlib import Path

import gdown
from docling.document_converter import DocumentConverter
from file_converter import convert_file_to_txt

converter = DocumentConverter()


def get_files(gdrive_urls: list[str], income_folder: Path):
    if not gdrive_urls:
        raise RuntimeError("Список URL папок пуст")

    for url in gdrive_urls:
        print(f"[INFO] Скачиваем файлы из: {url}")
        gdown.download_folder(url=url, output=str(income_folder), quiet=False)


def convert_files(income_folder: Path, prepared_folder: Path):
    prepared_folder.mkdir(parents=True, exist_ok=True)

    for src in income_folder.iterdir():
        if not src.is_file():
            continue
        try:
            convert_file_to_txt(src, prepared_folder)
        except Exception as e:
            print(f"[ERROR] {src.name}: {e}")
