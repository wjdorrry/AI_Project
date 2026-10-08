# AI-сервис (локально) для Sillage&Style

Этот сервис даёт API **`/analyze`**: загружаете фото → получаете вероятности по тегам (`Black`, `Evening`, `Streetwear` и т.д.).

В **`app.py`** поднимается обученная **ViT** из `../ml/runs/latest/` (две головы: стиль + цвет по пиксельной эвристике). Если весов нет, `/analyze` вернёт пустые теги и пояснение в поле `note`.

## Запуск

Из папки **`ai-service`** (подставьте свой путь, если проект лежит в другом месте):

```powershell
Set-Location -LiteralPath "C:\BRU_3_course\2_sem\ИИ\ИИ\ИИ\ai-service"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

**Запускайте сервер так** (обходит битый лаунчер `uvicorn.exe` на Windows после переноса папки или кириллицы в пути):

```powershell
py -m uvicorn app:app --reload --port 8000
```

Если уже активирован venv, можно так:

```powershell
python -m uvicorn app:app --reload --port 8000
```

Проверка:

- `GET http://localhost:8000/health`
- Swagger: `http://localhost:8000/docs`

## Если в браузере «Не удалось распознать теги»

1. Убедитесь, что в терминале сервер **действительно запущен** (нет traceback при старте).
2. Если при `uvicorn ...` появляется **`Fatal error in launcher`** и путь с `????` — удалите папку **`.venv`**, создайте её заново командами выше и используйте только **`py -m uvicorn ...`** (не голый `uvicorn`).
3. Фронт ходит на `http://localhost:8000` — порт должен совпадать.

### Папка `.venv` битая (`No pyvenv.cfg`), удалить не получается

Часто мешает **другой процесс** (второй терминал, Python в Cursor, старый `uvicorn`): файлы `torch` заняты, `Remove-Item` падает.

**Вариант А (проще всего):** запуск через отдельное окружение **`.venv_ai`** — скрипт в этой папке:

```powershell
Set-Location -LiteralPath "C:\BRU_3_course\2_sem\ИИ\ИИ\ИИ\ai-service"
powershell -ExecutionPolicy Bypass -File .\run_server.ps1
```

Первый раз установит зависимости в `.venv_ai`, потом поднимет сервер.

**Вариант Б:** закройте все терминалы с Python, в диспетчере задач завершите **python.exe**, затем вручную удалите папку `ai-service\.venv` через Проводник и снова выполните блок «Запуск» выше.


## Неделя 1: аккаунты и база пользователей

При первом запуске сервис автоматически создаёт локальную SQLite-базу `ai-service/sillage_style.db`.

API авторизации:

- `POST /auth/register` — регистрация пользователя;
- `POST /auth/login` — вход;
- `GET /auth/me` — данные текущего пользователя;
- `POST /analyze` — теперь требует заголовок `Authorization: Bearer <token>`.

Пароли не сохраняются в открытом виде: в базе хранится scrypt-хэш с индивидуальной солью. Подписывающий секрет для токенов создаётся локально в `.auth_secret` и исключён из Git. Токен действует 7 дней.
