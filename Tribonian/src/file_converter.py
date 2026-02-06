from pathlib import Path
from typing import Union

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"


def pdf_to_text(pdf_path: Union[str, Path]) -> str:
    """Извлекаем текст из PDF."""
    pdf_path = Path(pdf_path)
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text") + "\n"
    return text


def image_to_text(img_path: Union[str, Path]) -> str:
    """OCR для изображений JPG/PNG/WEBP."""
    img_path = Path(img_path)
    img = Image.open(img_path)
    # Можно настроить язык через lang="rus" при наличии tesseract-русской локали
    text = pytesseract.image_to_string(img, lang="rus+eng")
    return text


def convert_file_to_txt(
    file_path: Union[str, Path], output_folder: Union[str, Path]
) -> Path:
    """Конвертируем PDF или изображение в текстовый файл."""
    file_path = Path(file_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    out_file = output_folder / f"{file_path.stem}.txt"

    if out_file.exists():
        return out_file

    if file_path.suffix.lower() == ".pdf":
        text = pdf_to_text(file_path)
    elif file_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".tiff"]:
        text = image_to_text(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

    out_file.write_text(text, encoding="utf-8")
    return out_file


def convert_folder(input_folder: Union[str, Path], output_folder: Union[str, Path]):
    """Конвертируем все поддерживаемые файлы в папке в TXT."""
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for file_path in input_folder.iterdir():
        if not file_path.is_file():
            continue
        try:
            convert_file_to_txt(file_path, output_folder)
            print(f"[INFO] Конвертирован: {file_path.name}")
        except Exception as e:
            print(f"[ERROR] {file_path.name}: {e}")
