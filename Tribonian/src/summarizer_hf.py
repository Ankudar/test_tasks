import os
import re
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Загружаем токен из .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


class Summarizer:
    def __init__(
        self,
        model_name: str = "sarahai/ru-sum",
        device: int = -1,  # -1 — CPU, 0 — GPU
        max_chunk_tokens: int = 512,
        min_chunk_tokens: int = 100,
        max_summary_length: int = 200,
        min_summary_length: int = 80,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        self.model_name = model_name
        self.device = device
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.max_summary_length = max_summary_length
        self.min_summary_length = min_summary_length
        self.temperature = temperature
        self.top_p = top_p

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

    @staticmethod
    def clean_text(text: str) -> str:
        """Очистка текста от OCR-артефактов и дубликатов."""
        # Удаляем специальные символы и исправляем OCR-ошибки
        text = text.replace("\x0c", " ").replace("“", '"').replace("”", '"')
        text = re.sub(r'[^\w\s.,!?;:\-()\[\]"\'«»—]', " ", text)

        # Исправляем частые OCR-ошибки в юридических текстах
        # можно и нужно расширять, при текущих настройках и слабой локальной модели
        corrections = {
            r"\bнеене\b": "недействительны",
            r"\bФвдеральным\b": "Федеральным",
            r"\bзажоном\b": "законом",
            r"\bсвом\b": "своим",
            r"\bосуцестьления\b": "осуществления",
            r"\bгражданских пров\b": "гражданских прав",
            r"\bоспаривать\b": "оспаривать",
            r"\bпосладстй\b": "последствий",
        }

        for pattern, replacement in corrections.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Нормализуем пробелы
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)

        # Убираем повторяющиеся предложения и абзацы
        lines = text.split("\n")
        unique_lines = []
        seen_lines = set()

        for line in lines:
            line_clean = line.strip()
            if len(line_clean) < 5:
                continue

            line_key = re.sub(r"[^\w\s]", "", line_clean).lower().strip()

            if line_key and line_key not in seen_lines:
                unique_lines.append(line_clean)
                seen_lines.add(line_key)

        return "\n".join(unique_lines)

    def deduplicate_summary(self, summary: str) -> str:
        """Удаление дубликатов из суммаризации."""
        sentences = re.split(r"(?<=[.!?]) +", summary)
        seen = set()
        new_sentences = []

        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue

            s_key = re.sub(r"[^\w\s]", "", s_clean).lower()
            words = s_key.split()

            if len(words) < 3:
                continue

            if s_key not in seen:
                new_sentences.append(s_clean)
                seen.add(s_key)

        return " ".join(new_sentences)

    def split_into_paragraphs(self, text: str, max_tokens: int = 400) -> List[str]:
        """Разделение текста на осмысленные абзацы."""
        paragraphs = []
        current_para = []
        current_tokens = 0

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            line_tokens = len(self.tokenizer.encode(line, add_special_tokens=False))

            # Если строка слишком длинная, делим по предложениям
            if line_tokens > max_tokens:
                sentences = re.split(r"(?<=[.!?]) +", line)
                for sentence in sentences:
                    sent_tokens = len(
                        self.tokenizer.encode(sentence, add_special_tokens=False)
                    )
                    if sent_tokens + current_tokens > max_tokens and current_para:
                        paragraphs.append(" ".join(current_para))
                        current_para = [sentence]
                        current_tokens = sent_tokens
                    else:
                        current_para.append(sentence)
                        current_tokens += sent_tokens
            else:
                if line_tokens + current_tokens > max_tokens and current_para:
                    paragraphs.append(" ".join(current_para))
                    current_para = [line]
                    current_tokens = line_tokens
                else:
                    current_para.append(line)
                    current_tokens += line_tokens

        if current_para:
            paragraphs.append(" ".join(current_para))

        return paragraphs

    def summarize_chunk(self, chunk: str) -> Optional[str]:
        """Суммаризация одного чанка с обработкой ошибок."""
        try:
            tokens = self.tokenizer.encode(chunk, add_special_tokens=False)
            if len(tokens) < 50:  # Минимальный размер для осмысленной суммаризации
                return None

            # Обрезаем до максимального размера
            if len(tokens) > self.max_chunk_tokens:
                chunk = self.tokenizer.decode(
                    tokens[: self.max_chunk_tokens], skip_special_tokens=True
                )

            # Генерируем суммаризацию с параметрами
            summary = self.summarizer(
                chunk,
                max_length=self.max_summary_length,
                min_length=self.min_summary_length,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=1.5,
                no_repeat_ngram_size=3,
            )

            summary_text = self.deduplicate_summary(summary[0]["summary_text"])
            return summary_text.strip()

        except Exception as e:
            print(f"[ERROR] Ошибка суммаризации: {e}")
            return None

    def post_process_summary(self, summary: str) -> str:
        """Пост-обработка финальной суммаризации."""
        # Удаляем повторяющиеся блоки
        sentences = summary.split(". ")
        unique_sentences = []
        seen_phrases = set()

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            words = sentence.split()
            if len(words) < 3:
                continue

            key_phrase = " ".join(words[:4]).lower()

            if key_phrase not in seen_phrases:
                unique_sentences.append(sentence)
                seen_phrases.add(key_phrase)

        # Форматируем результат
        result = ". ".join(unique_sentences)
        if result and not result.endswith("."):
            result += "."

        return result

    def summarize_folder(self, folder: Path) -> str:
        """Суммаризация всех текстов в папке."""
        folder = Path(folder)
        all_text = []

        print(f"[INFO] Чтение файлов из {folder}")

        # Собираем текст из всех файлов
        for file_path in sorted(folder.iterdir()):
            if file_path.suffix.lower() == ".txt":
                try:
                    text = file_path.read_text(encoding="utf-8")
                    cleaned_text = self.clean_text(text)
                    if cleaned_text:
                        all_text.append(cleaned_text)
                        print(f"[INFO] Обработан файл: {file_path.name}")
                except Exception as e:
                    print(f"[ERROR] Ошибка чтения файла {file_path}: {e}")

        if not all_text:
            print("[WARNING] Нет текста для суммаризации")
            return ""

        full_text = "\n\n".join(all_text)

        # Разделяем на абзацы
        paragraphs = self.split_into_paragraphs(full_text, self.max_chunk_tokens)
        print(f"[INFO] Разделено на {len(paragraphs)} абзацев")

        # Суммаризируем каждый абзац
        summaries = []
        for i, para in enumerate(paragraphs):
            print(f"[INFO] Суммаризация абзаца {i+1}/{len(paragraphs)}...")
            summary = self.summarize_chunk(para)
            if summary:
                summaries.append(summary)

        # Объединяем все суммаризации
        combined_summary = "\n".join(summaries)

        # Если суммаризация получилась слишком длинной, делаем финальную суммаризацию
        final_tokens = len(
            self.tokenizer.encode(combined_summary, add_special_tokens=False)
        )
        if final_tokens > 800:
            print("[INFO] Делаем финальную суммаризацию объединенного текста...")
            final_summary = self.summarize_chunk(combined_summary)
            if final_summary:
                combined_summary = final_summary

        # Пост-обработка
        final_result = self.post_process_summary(combined_summary)

        return final_result

    def save_summary(self, summary: str, output_path: Path):
        """Сохранение суммаризации в файл."""
        output_path = Path(output_path)

        # Форматируем вывод
        formatted_summary = "=== СУММАРИЗАЦИЯ ===\n\n"
        formatted_summary += summary

        output_path.write_text(formatted_summary, encoding="utf-8")
        print(f"[INFO] Суммаризация сохранена в {output_path}")
