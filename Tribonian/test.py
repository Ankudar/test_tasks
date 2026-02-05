# Практическая задача
# Есть папка с документами: https://drive.google.com/drive/folders/1x6EKNkVw6PlFVTr6cGrsVscmRuwqGrXd?usp=sharing
# Суть: написать программу, которая делает общий саммари по содержимому папки (всей информации внутри).
# • Язык: Python (предпочтительно) или любой другой
# • LLM: используйте бесплатный OpenRouter — регистрация бесплатная, есть бесплатные модели
# • Можно и нужно использовать AI-помощников (Antigravity, Claude Code, Codex, Cursor, Roo и т.д.) — мы это приветствуем
# • Напишите код так, чтобы его функционал можно было расширять или модульно менять

# GDRIVE_URL = "https://drive.google.com/drive/folders/1x6EKNkVw6PlFVTr6cGrsVscmRuwqGrXd?usp=sharing"
# DATA_INCOME_FOLDER = "./Tribonian/data/income/"
# DATA_PREPARED_FOLDER = "./Tribonian/data/prepared/"

import os
from pathlib import Path

import gdown
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()

GDRIVE_URL = os.environ["GDRIVE_URL"]
DATA_INCOME_FOLDER = Path(os.environ["DATA_INCOME_FOLDER"])
DATA_PREPARED_FOLDER = Path(os.environ["DATA_PREPARED_FOLDER"])


DATA_INCOME_FOLDER.mkdir(parents=True, exist_ok=True)
DATA_PREPARED_FOLDER.mkdir(parents=True, exist_ok=True)

converter = DocumentConverter()


def get_files(gdrive_url: str, income_folder: str):
    if not gdrive_url:
        raise RuntimeError("URL папки не задан")

    gdown.download_folder(url=gdrive_url, output=str(income_folder), quiet=False)


def convert_files(income_folder: str, prepared_folder: str):
    for src in income_folder.iterdir():
        if not src.is_file():
            continue

        dst = prepared_folder / f"{src.stem}.txt"

        if dst.exists():
            continue
        try:
            result = converter.convert(src)
            text = result.document.export_to_text()
            dst.write_text(text, encoding="utf-8")
        except Exception as e:
            print(f"[ERROR] {src.name}: {e}")


def get_summary(prepared_folder: str):
    prepared_folder = Path(prepared_folder)

    # Собираем весь текст
    all_text = ""
    for file_path in prepared_folder.iterdir():
        if file_path.suffix.lower() == ".txt":
            all_text += file_path.read_text(encoding="utf-8") + "\n\n"

    if not all_text.strip():
        print("Нет текста для суммаризации")
        return

    # Создаём пайплайн для суммаризации
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Hugging Face умеет суммаризовать до ~1024 токенов за раз
    max_chunk = 1000
    chunks = [
        all_text[i : i + max_chunk * 4] for i in range(0, len(all_text), max_chunk * 4)
    ]

    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
        summaries.append(summary[0]["summary_text"])

    result = "\n\n".join(summaries)
    return result


if __name__ == "__main__":
    # get_files(GDRIVE_URL, DATA_INCOME_FOLDER)
    # convert_files(DATA_INCOME_FOLDER, DATA_PREPARED_FOLDER)
    print(get_summary(DATA_PREPARED_FOLDER))
