# Tribonian — конвертация документов и суммаризация (RU)

Проект для:
- скачивания документов из Google Drive
- конвертации PDF / изображений → TXT
- суммаризации русского текста с помощью моделей HuggingFace

Поддерживается CPU и GPU.

## 1. Требования
### Обязательные

- Python
- Git
- Tesseract OCR (для обработки изображений)

### Опционально

- NVIDIA GPU + CUDA (для ускорения суммаризации)

## 2. Клонирование проекта
```
git clone https://github.com/Ankudar/test_tasks.git
cd Tribonian
```

## 3. Создание виртуального окружения
```
Windows
python -m venv .venv
.venv\Scripts\activate
```

```
Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Проверка:

```
python --version
```

## 4. Установка системных зависимостей
### 4.1 Установка Tesseract OCR (обязательно)

#### Windows

Скачать:
https://github.com/UB-Mannheim/tesseract/wiki

Установить (по умолчанию):

C:\Program Files\Tesseract-OCR\tesseract.exe

В коде должен быть указан путь:

```
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
```
Проверка:

```
tesseract --version
```

## 5. Установка Python-зависимостей
### 5.1 Установка PyTorch

Вариант A — CPU (по умолчанию)
```
pip install torch
```

Вариант B — GPU (CUDA 13.0)
```
pip install torch==2.10.0+cu130 --index-url https://download.pytorch.org/whl/cu130
```

Проверка:

```
python -c "import torch; print(torch.cuda.is_available())"
```

### 5.2 Установка остальных зависимостей

```
pip install -r requirements.txt
```

Рекомендуемое содержимое requirements.txt:

```
torch==2.10.0
transformers==5.1.0
huggingface_hub==1.4.0
python-dotenv==1.2.1
gdown==5.2.1
PyMuPDF==1.26.7
pytesseract==0.3.13
pillow==12.1.0
```

## 6. HuggingFace токен (обязательно)

**Некоторые модели требуют авторизацию.**

### 6.1 Получение токена

- Зарегистрироваться на https://huggingface.co
- Settings → Access Tokens
- Создать Read token

### 6.2 Файл .env

В корне проекта создать .env:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxx
DATA_INCOME_FOLDER=./data/income
DATA_PREPARED_FOLDER=./data/prepared
```

**.env не коммитить**

## 7. Подготовка ссылок Google Drive

Файл:
```
Tribonian/urls.txt
```


Пример содержимого:
```
https://drive.google.com/drive/folders/XXXX
https://drive.google.com/drive/folders/YYYY
```

**Каждая ссылка — с новой строки.**

## 8. Запуск пайплайна
### 8.1 Основной запуск
```
python src/main.py
```

Пайплайн:
- (опционально) скачивает файлы из Google Drive
- конвертирует PDF / изображения → TXT
- суммаризирует тексты
- сохраняет результат в Tribonian/result.txt

## 8.2 Переключение CPU / GPU

В main.py:

```
summarizer = Summarizer(device=0)    # GPU
summarizer = Summarizer(device=-1)   # CPU
```

## 9. Выходные файлы

- data/prepared/*.txt — сконвертированные документы
- Tribonian/result.txt — итоговая суммаризация (перезаписывается при каждом запуске)

## 10. Типовые проблемы

### ModuleNotFoundError
```
pip install -r requirements.txt
```

### tesseract is not installed
- проверить путь к tesseract.exe
- перезапустить терминал

### Плохое качество суммаризации
- ограничение модели sarahai/ru-sum
- ожидаемо для длинных юридических текстов
- улучшение возможно только сменой модели или использованием LLM (OpenRouter / GPT)

## 11. Рекомендуемая структура проекта
```
Tribonian/
├── src/
│   ├── main.py
│   ├── file_utils.py
│   ├── file_converter.py
│   └── summarizer_hf.py
├── data/
│   ├── income/
│   └── prepared/
├── urls.txt
├── result.txt
├── requirements.txt
├── .env
└── README.md
```

## 12. Важно

**виртуальное окружение — всегда новое**
**Torch + CUDA ставить первым**
**HuggingFace токен обязателен**