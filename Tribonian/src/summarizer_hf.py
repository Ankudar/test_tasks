import os
import re
from pathlib import Path

from dotenv import load_dotenv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Загружаем токен из .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


class Summarizer:
    def __init__(
        self,
        model_name: str = "sarahai/ru-sum",  # русская модель
        device: int = -1,  # -1 — CPU, 0 — GPU
        max_chunk_tokens: int = 512,
        min_chunk_tokens: int = 50,
        max_summary_length: int = 300,
        min_summary_length: int = 50,
    ):
        # Загружаем модель и токенизатор с авторизацией
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_auth_token=HF_TOKEN
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, use_auth_token=HF_TOKEN
        )
        self.summarizer = pipeline(
            "summarization",
            model=model,
            tokenizer=self.tokenizer,
            device=device,
        )
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.max_summary_length = max_summary_length
        self.min_summary_length = min_summary_length

    @staticmethod
    def clean_text(text: str) -> str:
        # базовая очистка
        text = text.replace("\x0c", " ")
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\S\r\n]+", " ", text)

        # убираем повторяющиеся строки
        lines = text.split("\n")
        new_lines = []
        seen_lines = set()
        for line in lines:
            line_clean = line.strip()
            if line_clean and line_clean not in seen_lines:
                new_lines.append(line_clean)
                seen_lines.add(line_clean)

        cleaned_text = "\n".join(new_lines)

        # удаляем повторяющиеся фразы через regex (например, "Глава 1 ОБЩИЕ ПОЛОЖЕНИЯ" и "Глава 1 ОБЩИЕ")
        cleaned_text = re.sub(r"(Глава \d+ [^\n]{5,100})( \1)+", r"\1", cleaned_text)
        return cleaned_text

    def deduplicate_summary(self, summary: str) -> str:
        sentences = re.split(r"(?<=[.!?]) +", summary)
        seen = set()
        new_sentences = []
        for s in sentences:
            s_clean = s.strip()
            if s_clean and s_clean not in seen:
                new_sentences.append(s_clean)
                seen.add(s_clean)
        return " ".join(new_sentences)

    def chunk_text(self, text: str) -> list[str]:
        """Делим текст на токенизированные чанки с поиском конца предложений."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + self.max_chunk_tokens, len(tokens))
            # ищем конец предложения в последних 100 токенах
            for i in range(end, max(start, end - 100), -1):
                chunk_text = self.tokenizer.decode(
                    tokens[start:i], skip_special_tokens=True
                )
                if any(p in chunk_text[-5:] for p in ".!?"):
                    end = i
                    break
            chunk_text = self.tokenizer.decode(
                tokens[start:end], skip_special_tokens=True
            )
            if chunk_text.strip():
                chunks.append(chunk_text)
            start = end
        return chunks

    def summarize_folder(self, folder: Path) -> str:
        folder = Path(folder)
        all_text = ""
        for file_path in folder.iterdir():
            if file_path.suffix.lower() == ".txt":
                text = file_path.read_text(encoding="utf-8")
                text = self.clean_text(text)
                if text:
                    all_text += text + "\n\n"

        if not all_text.strip():
            print("[INFO] Нет текста для суммаризации")
            return ""

        base_chunks = self.chunk_text(all_text)

        # объединяем маленькие чанки
        optimized_chunks = []
        buffer = ""
        for chunk in base_chunks:
            tokens = self.tokenizer.encode(chunk, add_special_tokens=False)
            if len(tokens) < self.min_chunk_tokens:
                buffer += chunk + " "
            else:
                if buffer:
                    optimized_chunks.append(buffer.strip())
                    buffer = ""
                optimized_chunks.append(chunk)
        if buffer:
            optimized_chunks.append(buffer.strip())

        summaries = []
        for i, chunk in enumerate(optimized_chunks):
            tokens = self.tokenizer.encode(chunk, add_special_tokens=False)
            if len(tokens) == 0:
                continue

            # безопасная обрезка по max_chunk_tokens
            if len(tokens) > self.max_chunk_tokens:
                chunk = self.tokenizer.decode(
                    tokens[: self.max_chunk_tokens], skip_special_tokens=True
                )

            try:
                summary = self.summarizer(
                    chunk,
                    max_length=self.max_summary_length,
                    min_length=self.min_summary_length,
                    do_sample=False,
                )
                summary_text = self.deduplicate_summary(summary[0]["summary_text"])
                summaries.append(summary_text)
                print(
                    f"[INFO] Суммаризован chunk {i+1}/{len(optimized_chunks)} (tokens ~{len(tokens)})"
                )
            except Exception as e:
                print(f"[ERROR] Ошибка суммаризации chunk {i+1}: {e}")

        return "\n\n".join(summaries)
