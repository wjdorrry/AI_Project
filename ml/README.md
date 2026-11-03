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


---

## Неделя 4 — расширение распознавания цвета

На этом этапе цветовая классификация расширена с 9 до 15 классов:

- Black
- White
- Grey
- Beige
- Blue
- Brown
- Green
- Red
- Pink
- Burgundy
- Purple
- Yellow
- Orange
- Navy
- Cream

### 1. Подготовка цветового датасета

Используется открытый fashion-датасет `HEBA2002/fashion-product-images-small`.
Из него выбираются только изображения одежды (`masterCategory = Apparel`) и только нужные 15 цветов.

По умолчанию берётся 60 изображений на каждый цвет, всего 900 изображений.

Запуск из корня проекта:

```powershell
.\ml\.venv_week3\Scripts\python.exe .\dataset\prepare_week4_color_dataset.py
```

Будут созданы:

- `dataset/week4_color_images/`
- `dataset/week4_color_labels.csv`

### 2. Дообучение цветовой части ViT

Неделя 4 использует лучшую модель недели 3 как исходную.
ViT-база и style-голова замораживаются, чтобы не ухудшить уже обученные 9 стилей.
Обучается новая color-голова на 15 цветах.

```powershell
cd ml
$env:SILLAGE_BATCH_SIZE="4"
.\.venv_week3\Scripts\python.exe .\train_week4_colors.py
```

Если видеопамяти недостаточно, используйте:

```powershell
$env:SILLAGE_BATCH_SIZE="1"
```

### 3. Метрики

После каждой эпохи рассчитываются реальные:

- Color Accuracy
- Color macro-F1
- Color MAE
- Color MSE
- Train loss
- Validation loss

Лучшая эпоха выбирается по Color F1.

### 4. Графики

В `ml/runs/week4/` автоматически сохраняются:

- `dataset_distribution.png` — распределение 15 цветов;
- `loss_curve.png` — train/validation loss;
- `accuracy_f1_curve.png` — Color Accuracy и F1;
- `mae_mse_curve.png` — Color MAE и MSE;
- `color_confusion_matrix.png` — матрица ошибок по 15 цветам;
- `before_after_color_metrics.png` — сравнение качества цвета до и после недели 4;
- `training_history.csv`;
- `metrics_summary.json`.

Лучшая модель также сохраняется в `ml/runs/latest/model.pt` вместе с новой картой из 15 цветовых классов.


---

## Неделя 5 — сезон в системе подбора ароматов

На этом этапе нейросеть не переобучается. ИИ по-прежнему определяет цвет и стиль, а пользователь отдельно выбирает сезон:

- Spring
- Summer
- Autumn
- Winter

Сезон добавлен в экспертную матрицу как третий фактор. Итоговый расчёт выполняется по схеме:

`цвет + стиль + сезон → баллы семейств → конкретные ароматы`

Для каждого сезона используются собственные веса по 11 парфюмерным семействам. Сезон выбирается вручную, потому что система не должна угадывать по фотографии, для какого времени года пользователь хочет получить рекомендацию.

Для проверки влияния сезона добавлен скрипт:

```powershell
py scripts\week5_season_comparison.py
```

Он сравнивает один и тот же образ `Black + Elegant` для четырёх сезонов и сохраняет:

- `analysis/week5/season_comparison.csv`
- `analysis/week5/season_comparison.png`

Так можно показать, что сезон реально меняет итоговый рейтинг парфюмерных семейств.


---

## Неделя 6 — объяснение рекомендации

На этом этапе система показывает не только итоговое семейство и конкретные ароматы, но и объясняет, почему они были выбраны.

Для победившего семейства сохраняется вклад каждого фактора:

- цвет;
- стиль;
- сезон.

Пример:

`Black + Elegant + Winter → Oriental`

Система показывает отдельные веса каждого фактора для семейства-победителя и итоговую сумму. Также у каждого конкретного аромата отображаются семейства, по которым он совпал с рекомендацией.

В интерфейсе добавлен блок `Почему этот аромат?`.

Для демонстрации расчёта добавлен скрипт:

```powershell
py scripts\week6_explanation_demo.py
```

Он сохраняет:

- `analysis/week6/explanation_example.json` — полный расчёт;
- `analysis/week6/winner_contributions.png` — график вклада цвета, стиля и сезона.

Объяснение строится из реальных весов экспертной матрицы и не генерируется случайным текстом.


---

## Дополнительное улучшение цвета после недели 6

Цель этапа — улучшить распознавание 15 цветов без потери уже обученного распознавания стилей.

Изменения:
- новый сбалансированный цветовой датасет по 120 изображений на класс;
- отдельные train / validation / test выборки;
- старт с лучшей модели недели 4;
- разморозка только последнего блока ViT;
- style-голова остаётся замороженной;
- отдельный малый learning rate для backbone;
- итоговое сравнение старой и новой модели на одной и той же test-выборке;
- дополнительная проверка, что качество распознавания стиля не ухудшилось.

Подготовка данных:

```powershell
.\ml\.venv_week3\Scripts\python.exe .\dataset\prepare_week6_color_dataset.py
```

Дообучение:

```powershell
.\ml\.venv_week3\Scripts\python.exe .\ml\train_week6_color_finetune.py
```

Результаты сохраняются в:

`ml/runs/week6_color_finetune/`

Новая модель специально не копируется в `ml/runs/latest` автоматически. Сначала нужно сравнить её с baseline недели 4.
