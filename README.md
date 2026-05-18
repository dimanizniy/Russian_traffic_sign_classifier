# Russian Traffic Sign Classifier

Учебный проект по классификации изображений российских дорожных знаков с использованием сверточных нейронных сетей на PyTorch.

## Задача

Цель проекта — разработать модель машинного обучения для распознавания дорожных знаков по изображениям.
Используется датасет RTSD, содержащий изображения российских дорожных знаков.

## Используемые технологии

- Python
- PyTorch
- torchvision
- CUDA
- scikit-learn
- pandas
- matplotlib
- Docker

## Структура проекта

```text
traffic-sign-classifier/
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   ├── checkpoints/
│   ├── metrics/
│   └── plots/
├── reports/
├── src/
│   ├── config.py
│   ├── prepare_data.py
│   ├── dataset.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .gitignore

Датасет

Архив датасета необходимо поместить в:

data/raw/rtsd-dataset.zip

После этого выполнить подготовку данных:

python src/prepare_data.py

Скрипт вырезает дорожные знаки из исходных кадров по bounding box-разметке и формирует структуру:

data/processed/
├── train/
└── val/
Обучение модели
python src/train.py

Основная модель:

ResidualCNN

Результаты сохраняются в:

outputs/
├── checkpoints/best_model.pth
├── metrics/history.json
└── plots/
Оценка модели
python src/evaluate.py

Скрипт сохраняет:

outputs/metrics/classification_report.csv
outputs/metrics/top_errors.csv
outputs/metrics/eval_summary.json
outputs/plots/confusion_matrix.png
Предсказание для одного изображения
python src/predict.py --image path/to/image.jpg

Пример:

python src/predict.py --image data/processed/val/1_1/example.jpg
Docker

Сборка:

docker compose build

Проверка CUDA:

docker compose run --rm sign-classifier python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

Обучение:

docker compose run --rm sign-classifier python src/train.py

Оценка:

docker compose run --rm sign-classifier python src/evaluate.py
Полученный результат

На валидационной выборке:

Accuracy: 0.9956
Macro F1: 0.8864
Weighted F1: 0.9955
Ошибок: 39 из 8866
Особенности

Датасет является несбалансированным: часть классов представлена большим количеством изображений, а часть — малым. Поэтому для анализа используются не только accuracy, но и macro F1.


После этого сделаем `reports/report.md`.