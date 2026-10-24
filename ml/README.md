# Обучение своей модели (2 головы: color + style)

Цель: обучить модель, которая по фото возвращает **два предсказания**:

- `color`: один из 9 (`Black, White, Grey, Beige, Blue, Brown, Green, Red, Pink`)
- `style`: один из 3 (`Sportswear, Evening, Streetwear`)

Это **multi-task**: общий экстрактор признаков + 2 классификационные головы.

---

## Важная правда про ваши 30 фото Sportswear

Если в датасете есть **только один стиль** (`Sportswear`), модель **не сможет научиться отличать стили** (ей нечего сравнивать).
Но на этих 30 фото мы можем:

- настроить структуру датасета и разметку
- запустить пайплайн обучения и убедиться, что всё “крутится”
- (опционально) потренировать `color`, если вы разметите цвета

Чтобы реально классифицировать стили, вам нужно добавить хотя бы:

- 30+ фото `Evening`
- 30+ фото `Streetwear`

---

## Шаг 1. Подготовьте датасет

1) Сложите фото в `dataset/images/` и переименуйте:

- `0001.jpg`, `0002.jpg`, ...

2) Заполните `dataset/labels.csv`:

```csv
file,color,style
0001.jpg,Black,Sportswear
0002.jpg,Blue,Sportswear
```

Если пока не хотите размечать цвет — всё равно лучше начать размечать.
Без `color` мы можем временно обучать только `style`, но для этого нужны 3 класса.

---

## Шаг 2. Установите зависимости для обучения

```powershell
Set-Location -LiteralPath "c:\Дарина учёба\ИИ\ml"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

---

## Шаг 3. Запуск обучения (когда будут данные минимум по 2–3 стилям)

```powershell
Set-Location -LiteralPath "c:\Дарина учёба\ИИ\ml"
.\.venv\Scripts\Activate.ps1
py train_multitask_vit.py
```

Результаты сохранятся в `ml/runs/latest/`:

- `model.pt` (веса)
- `label_maps.json` (словари label↔id)

---

## Шаг 4. Подключение в ai-service

Когда модель обучена, мы поменяем `ai-service/app.py` так, чтобы он:

- загружал `model.pt`
- принимал фото
- возвращал JSON `{ tags: [...] }` уже из **вашей модели**

Фронт трогать не надо — он уже умеет звать `localhost:8000/analyze`.



---

## Неделя 3 — расширение распознавания стиля

На этом этапе классификация стиля расширена до 9 классов:

- Elegant
- Casual
- Romantic
- Minimalist
- Sporty
- Streetwear
- Boho
- Office
- Evening

### 1. Подготовка датасета

Текущий локальный датасет содержит только Sportswear, Evening и Streetwear.
Скрипт `dataset/prepare_week3_dataset.py` формирует новый сбалансированный датасет:

- Sporty, Streetwear и Evening — из локальных изображений проекта;
- Elegant, Casual, Romantic, Minimalist, Boho и Office — из открытого MIT-датасета
  `lihicarmeli/fashion-stylist-multimodal-v2`.

По умолчанию берётся 60 изображений на каждый класс, всего 540 изображений.

Запуск из корня проекта:

```powershell
py -3.12 -m pip install -r ml\requirements.txt
py -3.12 dataset\prepare_week3_dataset.py
```

Будут созданы:

- `dataset/week3_images/`
- `dataset/week3_labels.csv`

### 2. Обучение ViT

```powershell
cd ml
py -3.12 train_week3_styles.py
```

Модель по-прежнему имеет две головы: color и style.
Для новых внешних изображений обучается только style-голова, потому что их цветовая разметка
не совпадает с 9 цветовыми классами проекта. Color-голова обучается на локальных изображениях.

### 3. Метрики

После каждой эпохи сохраняются:

- Accuracy
- macro-F1
- MAE
- MSE
- train loss
- validation loss

MAE и MSE считаются между вектором вероятностей модели и правильной one-hot меткой.
Для такой задачи они используются как дополнительные показатели ошибки вероятностей:
чем меньше MAE и MSE, тем лучше.

### 4. Графики

После обучения автоматически создаются:

- `dataset_distribution.png` — количество изображений каждого стиля;
- `loss_curve.png` — ошибка обучения и проверки;
- `accuracy_f1_curve.png` — Accuracy и F1;
- `mae_mse_curve.png` — MAE и MSE;
- `style_confusion_matrix.png` — матрица ошибок по стилям;
- `training_history.csv` — значения всех метрик по эпохам;
- `metrics_summary.json` — итог лучшей эпохи.

Файлы сохраняются в `ml/runs/week3/`.
Лучшая модель также сохраняется в `ml/runs/latest/model.pt`.
