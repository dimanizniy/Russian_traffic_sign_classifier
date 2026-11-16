# Russian Traffic Sign Classifier 🚦

Классификация 155 классов российских дорожных знаков (RTSD) на PyTorch: кропы знаков
по bbox-разметке, свёрточная сеть, воспроизводимый DVC-пайплайн и Streamlit-демо.

---

## Быстрый старт

```bash
make install      # uv sync: .venv из uv.lock (torch с CUDA 13.0)
make pull         # готовая модель и кропы из DVC remote (DagsHub), без обучения
make app          # Streamlit на http://localhost:8501
```

Обучить с нуля: `make gpu-check && make repro` (скачать датасет -> нарезать -> обучить -> оценить).

Нужен токен Kaggle: `~/.kaggle/access_token` (или `KAGGLE_API_TOKEN`, или старые
`~/.kaggle/kaggle.json` / `KAGGLE_USERNAME`+`KAGGLE_KEY`). Архив `watchman/rtsd-dataset`
(~18.4 ГБ) скачивается автоматически этапом `download`, если его нет в `data/raw/`.

Все команды — через `make help`. Окружение управляется только **uv**
(`uv add`, `uv sync`, `uv run ...`), `requirements.txt` больше нет.

---

## Что изменилось во второй версии

| Проблема v1 | Решение v2 |
|---|---|
| macro F1 на 20 п.п. ниже accuracy | найдены и исправлены три причины, см. «Разбор дисбаланса» |
| нет контроля экспериментов | DVC: пайплайн `dvc.yaml`, параметры `params.yaml`, метрики в git, модель и данные в DVC-кэше/remote, `dvc exp` |
| нет пути до версии данных | `dvc.lock` фиксирует хэши архива, кропов и модели для каждого коммита |
| датасет качается руками | этап `download` (Kaggle API), запускается автоматически перед обучением |
| нет управления пайплайном | `Makefile`: install / check / data / train / exp / app / docker |
| pip + requirements.txt | uv + `pyproject.toml` + `uv.lock`, Docker-образ тоже на uv |

Итог на честном сплите: **macro F1 84.96 → 94.48** (accuracy 92.85 → 98.99),
разрыв между accuracy и macro F1 сократился с 8 до 4.5 п.п.

---

## Разбор дисбаланса (почему macro F1 отставал)

1. **Метрика считалась по классам, которых нет в val.** В официальном val 139 из 155
   классов. `classification_report(labels=все_155)` ставит отсутствующим классам F1 = 0,
   и macro F1 занижался примерно на 10%. Теперь `macro_f1` считается по классам,
   присутствующим в val; старая формула сохранена как `macro_f1_all_train_classes`.
2. **Веса `N / n_c` были слишком агрессивными.** Соотношение частот классов —
   22147 : 1, такое же было и у весов. Модель массово «перепредсказывала» редкие классы:
   macro recall 0.95 при precision 0.83. Мягкие варианты (`sqrt_inverse`, sqrt-сэмплер
   `WeightedRandomSampler` без весов в лоссе) дают +8 п.п. macro F1.
3. **Валидация протекала.** Официальный сплит случайный по кадрам видеорегистратора:
   у 97.9% кадров val есть кадр train той же сцены ближе чем за 1 секунду. Добавлен
   честный сплит по поездкам (`data.split: sequence`): кадры группируются в поездки
   (пауза > 60 с), в val целиком уходят ~10% кадров. Это теперь режим по умолчанию.

Аугментация: горизонтальных отражений нет и не должно быть (меняют смысл знака).
Пресет `light` убирает перспективу и сдвиг и ослабляет поворот: на кропах 64 px
связка rotation + shear + perspective давала чёрные углы и размывала пиктограммы.

---

## Результаты экспериментов

Все прогоны — `dvc exp`, параметры и метрики видны в `make exp-show`.
GPU: RTX 3050 Ti Laptop (4 ГБ), ~42 с/эпоха для ResidualCNN 64 px.

### Официальный сплит (с утечкой, для сравнения с v1)

| Эксперимент | Accuracy | Macro P | Macro R | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|
| v1 из README прошлой версии | 90.85 | — | — | 71.83* | 91.95 |
| baseline (настройки v1) | 92.60 | 83.29 | 95.26 | 87.04 (78.05*) | 93.61 |
| веса effective_number | 98.49 | 94.72 | 97.64 | 95.64 | 98.54 |
| веса sqrt_inverse | 99.41 | 99.13 | 98.69 | 98.81 | 99.41 |
| без весов | 99.56 | 99.55 | 98.04 | 98.65 | 99.55 |
| sqrt-сэмплер | 99.56 | 99.58 | 98.77 | 99.10 | 99.56 |

\* старая формула — по всем 155 классам, включая 16 отсутствующих в val.

### Сплит по поездкам (честная оценка, 100 классов в val)

| Эксперимент | Accuracy | Macro P | Macro R | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|
| baseline (настройки v1) | 92.85 | 84.25 | 89.50 | 84.96 | 93.88 |
| без весов | 98.75 | 93.24 | 92.89 | 92.61 | 98.67 |
| веса sqrt_inverse | 98.60 | 94.72 | 93.54 | 93.28 | 98.58 |
| веса effective_number | 97.69 | 92.17 | 91.09 | 90.56 | 97.74 |
| sqrt-сэмплер | 98.87 | 95.05 | 93.29 | 93.33 | 98.80 |
| sqrt-сэмплер + focal | 98.76 | 95.11 | 93.11 | 93.37 | 98.72 |
| сэмплер + focal + light + LS 0.1 + 30 эп. (3 seed) | 98.92 | 94.95 | 93.42 | 93.54 ± 0.33 | 98.86 |
| ResNet18 ImageNet, 96 px, та же схема | 98.64 | 96.08 | 93.42 | 93.91 | 98.60 |
| **сэмплер + CE + light + LS 0.1 + 30 эп. (3 seed)** | 98.99 | 95.79 | 94.28 | **94.48 ± 0.48** | 98.93 |

Текущая модель (`params.yaml`) — последняя строка, seed 42: macro F1 94.84, accuracy 99.06.

Кумулятивная абляция (каждая строка добавляет изменение к предыдущей; получилась
случайно, потому что `dvc exp run -S` не сбрасывает остальные параметры — теперь
`make exp` делает это сам):

| Шаг | Macro F1 |
|---|---|
| сэмплер + focal | 93.37 |
| + light-аугментация | 93.62 |
| + label smoothing 0.1 | 93.88 |
| + lr 3e-3 | 92.61 |
| + 30 эпох | 94.77 |
| → ResNet18 96 px | 93.47 |

**Выводы**

- Главный прирост — отказ от весов `N/n_c`: +8 п.п. macro F1 на обоих сплитах.
- Метрики официального сплита завышены почти дубликатами: при честном сплите лучший
  macro F1 ниже на ~4.5 п.п. Именно его и стоит считать качеством модели.
- light-аугментация, label smoothing 0.1 и 30 эпох вместо 15 дают ещё ~+1 п.п.
  (сэмплер + CE: 93.33 → 94.48);
  lr 3e-3 хуже 1e-3. Focal loss поверх сэмплера не помогает: с CE macro F1 выше
  на ~1 п.п. (94.48 ± 0.48 против 93.54 ± 0.33 по трём seed).
- Предобученный ResNet18 на 96 px (11.3 млн параметров против 2.0 млн) обучается
  столько же (~38 с/эпоха), но лучшую ResidualCNN не обгоняет — на кропах знаков
  ImageNet-признаки мало что добавляют.
- Оставшиеся ошибки — редкие классы: 53 класса целиком попали в train-поездки и в
  честном val отсутствуют; 9 из 100 классов val имеют ≤ 5 примеров. F1 < 0.8 у 6 классов,
  хуже всего `5_18` и `6_8_2` (F1 = 0, см. `outputs/metrics/top_errors.csv`). Следующий шаг — собрать для них больше
  данных или синтетику, а не усложнять модель.

Ограничение: лучшая эпоха выбирается по тому же val, на котором считаются метрики,
отдельного test нет — оценка слегка оптимистична.

---

## Пайплайн и версии данных

```text
download ──> prepare ──> train ──> evaluate
 (Kaggle)   (кропы, сплит)  (models/best_model.pth)  (outputs/metrics, outputs/plots/eval)
```

- `params.yaml` — все гиперпараметры (`data.*` для нарезки и сплита, `train.*` для обучения).
- `dvc.lock` — хэши входов/выходов каждого этапа. Чтобы получить модель и кропы
  конкретного коммита без переобучения: `git checkout <commit> && make pull`.
- Сырой архив (18 ГБ) в DVC-кэш не копируется (`cache: false`), источник правды — Kaggle;
  версионируются кропы `data/processed` (~450 МБ) и модель.
- Remote — DagsHub (`https://dagshub.com/neF1anders/RTSC.dvc`, задан в `.dvc/config`).
  После клона достаточно `make pull`, обучать заново не нужно
  (модель 8 МБ, кропы ~450 МБ; только модель — `uv run dvc pull models/best_model.pth`).
- DagsHub не отдаёт DVC-хранилище анонимно даже у публичного репозитория, поэтому
  нужен токен любого (бесплатного) аккаунта DagsHub — для `pull`, а у владельца и для `push`.
  Токен хранится только локально (`.dvc/config.local`, не в git):
  ```bash
  uv run dvc remote modify dagshub --local auth basic
  uv run dvc remote modify dagshub --local user <dagshub-login>
  uv run dvc remote modify dagshub --local password <token>
  ```

### Эксперименты

```bash
make exp ARGS="-n my-run -S train.lr=0.0005 -S train.epochs=20"
make exp-show                       # таблица всех экспериментов
make exp-apply NAME=seq-best-ce     # перенести эксперимент в workspace
git add -A && git commit            # зафиксировать как новую версию
```

`dvc exp apply` восстанавливает **все** отслеживаемые git файлы на момент запуска
эксперимента — незакоммиченные правки (например, README) перед ним лучше закоммитить.

Параметры, влияющие на дисбаланс: `train.class_weighting`
(`none | inverse | sqrt_inverse | effective_number`), `train.sampler` (`none | weighted`),
`train.loss` (`ce | focal`), `train.label_smoothing`, `train.augmentation` (`baseline | light`).
Модели: `residual_cnn`, `custom_cnn`, `resnet18` (ImageNet-веса).

---

## Структура проекта

```text
.
├── data/                  # raw/ (zip, не в git), processed/ (DVC)
├── models/                # best_model.pth (DVC)
├── outputs/
│   ├── metrics/           # eval_summary.json, train_metrics.json, history.json, отчёты (git)
│   └── plots/             # train/ и eval/ (git)
├── src/
│   ├── download_data.py   # этап download (Kaggle)
│   ├── prepare_data.py    # этап prepare: кропы + сплит
│   ├── splitting.py       # сплит по поездкам
│   ├── train.py           # этап train
│   ├── evaluate.py        # этап evaluate
│   ├── class_balance.py   # веса классов и сэмплера
│   ├── losses.py          # focal loss, выбор лосса
│   ├── dataset.py         # датасет, аугментации, даталоадеры
│   ├── models.py          # CustomCNN, ResidualCNN, ResNet18
│   ├── inference.py       # загрузка чекпоинта, top-k
│   ├── predict.py         # CLI-предсказание
│   ├── app.py             # Streamlit
│   └── config.py          # пути и типизированный params.yaml
├── tests/                 # pytest
├── dvc.yaml, dvc.lock, params.yaml
├── pyproject.toml, uv.lock
├── Makefile, Dockerfile, docker-compose.yml
└── README.md
```

---

## Предсказание и демо

```bash
make predict IMAGE_PATH=data/processed/val/1_1/<file>.jpg
make app
```

## Docker (деплой демо)

Образ на `python:3.12-slim` + uv, ставит зависимости строго из `uv.lock` и включает
текущую `models/best_model.pth` (перед сборкой: `make pull` или `make train`).

```bash
make docker-build
make docker-up      # http://localhost:8501, GPU через nvidia-container-toolkit
make docker-down
```

## Проверки качества

```bash
make check   # ruff check + ruff format --check + mypy + pytest
```
