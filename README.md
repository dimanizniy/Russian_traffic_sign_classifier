# Russian Traffic Sign Classifier 🚦

Учебный проект по компьютерному зрению и глубокому обучению для классификации российских дорожных знаков с использованием PyTorch и сверточных нейронных сетей.

---

## О проекте

Проект решает задачу многоклассовой классификации изображений дорожных знаков на датасете RTSD (Russian Traffic Sign Dataset).

Модель обучается распознавать 155 классов дорожных знаков по изображениям, вырезанным из исходных кадров по bounding box-разметке.

Проект включает:

- подготовку датасета;
- обучение модели;
- оценку качества;
- inference для одного изображения;
- веб-интерфейс на Streamlit;
- Docker-конфигурацию.

---

## Используемые технологии

- Python
- PyTorch
- torchvision
- CUDA
- scikit-learn
- pandas
- matplotlib
- Streamlit
- Docker

---

## Архитектура модели

Основная модель — `ResidualCNN`

Дополнительная модель - `CustomCNN`

Для смены модели в файле `config.py` изменить `MODEL_NAME`

Используются:

- residual-блоки;
- BatchNorm;
- ReLU;
- Dropout;
- AdamW;
- CosineAnnealingLR;
- data augmentation;
- weighted CrossEntropyLoss.

Модель обучается без pretrained-весов.

---

## Результаты(ResidualCNN)

### Validation metrics

| Metric | Value |
|---|---|
| Accuracy | 90.85% |
| Macro F1 | 71.83% |
| Weighted F1 | 91.95% |
| Validation samples | 8866 |
| Errors | 811 |

---

## Результаты(CustomCNN)

### Validation metrics

| Metric | Value |
|---|---|
| Accuracy | 80.98% |
| Macro F1 | 58.63% |
| Weighted F1 | 83.51% |
| Validation samples | 8866 |
| Errors | 1686 |

---

## Структура проекта

```text
.
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   ├── checkpoints/
│   ├── metrics/
│   └── plots/
├── reports/
├── src/
│   ├── app.py
│   ├── config.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── models.py
│   ├── predict.py
│   ├── prepare_data.py
│   └── train.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── .gitignore
└── .dockerignore
```
---

## Датасет

Используется RTSD (Russian Traffic Sign Dataset).
Датасет не включён в репозиторий из-за большого размера.
Необходимо самостоятельно скачать архив и поместить его в:
```bash
data/raw/rtsd-dataset.zip
```

---

## Подготовка данных

Подготовка выполняется скриптом:
```bash
python src/prepare_data.py
```

Скрипт:
читает JSON-аннотации;
вырезает дорожные знаки из исходных кадров;
сохраняет изображения в формате ImageFolder.
После обработки структура будет:
```text
data/processed/
├── train/
└── val/
```

---

## Обучение модели
Локальный запуск:
```bash
python src/train.py
```
Оценка модели
```bash
python src/evaluate.py
```

Сохраняются:
```text
outputs/metrics/classification_report.csv
outputs/metrics/top_errors.csv
outputs/metrics/eval_summary.json
outputs/plots/confusion_matrix.png
```

---

## Предсказание для одного изображения
```text
python src/predict.py --image path/to/image.jpg
```
Пример:
```bash
python src/predict.py --image data/processed/val/1_1/example.jpg
```

---

## Веб-интерфейс
Проект содержит Streamlit-интерфейс для тестирования модели.
Запуск:
```bash
streamlit run src/app.py
```
После запуска:
http://localhost:8501

В интерфейсе можно:

- загрузить изображение;
- получить top-5 предсказаний;
- увидеть вероятности классов;
- посмотреть пример эталонного знака из датасета.

---

## Docker

```bash
docker compose build
docker compose up
```

После запуска приложение доступно по адресу:
http://localhost:8501

---

## Особенности проекта
Датасет является сильно несбалансированным, поэтому кроме accuracy используются:

- macro F1-score;
- weighted F1-score;
- confusion matrix;
- classification report.

# Для улучшения качества редких классов используются:

- weighted loss;
- data augmentation;
- residual architecture.

# Возможные улучшения
- Focal Loss
- MixUp / CutMix
- Vision Transformer
- ONNX / TensorRT inference
- Real-time video inference
- Detection pipeline вместо classification crop-изображений