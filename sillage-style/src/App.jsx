import { useCallback, useEffect, useMemo, useState } from "react";
import matrix from "./data/perfumeStyleMatrix.json";
import { computeScores } from "./utils/scoreFamilies";
import { pickPerfumes } from "./utils/pickPerfumes";
import ImageUploader from "./components/ImageUploader.jsx";
import TagPicker from "./components/TagPicker.jsx";
import PreferenceToggle from "./components/PreferenceToggle.jsx";
import ResultDisplay from "./components/ResultDisplay.jsx";
import AuthGate from "./components/AuthGate.jsx";
import { authHeaders } from "./utils/api";
import "./App.css";

/** В dev — относительный путь (прокси Vite → :8000). В preview/production без VITE_AI_URL — прямой URL. */
function analyzeEndpoint() {
  const custom = import.meta.env.VITE_AI_URL;
  if (custom && String(custom).trim()) {
    return `${String(custom).replace(/\/$/, "")}/analyze`;
  }
  if (import.meta.env.DEV) return "/analyze";
  return "http://127.0.0.1:8000/analyze";
}

function MainApp() {
  const [files, setFiles] = useState([]);
  const [selectedTags, setSelectedTags] = useState(() => new Set());
  const [excluded, setExcluded] = useState(() => new Set());
  const [submitted, setSubmitted] = useState(false);
  const [busy, setBusy] = useState(false);
  const [aiBusy, setAiBusy] = useState(false);
  const [aiError, setAiError] = useState("");
  const [aiWarning, setAiWarning] = useState("");
  const [aiMode, setAiMode] = useState("");
  const [aiTopColor, setAiTopColor] = useState([]);
  const [aiTopStyle, setAiTopStyle] = useState([]);
  const [aiPicked, setAiPicked] = useState({ color: "", style: "" });
  /** Каталог из public/perfumes.json (~не в бандле). */
  const [perfumeCatalog, setPerfumeCatalog] = useState({ perfumes: [] });

  useEffect(() => {
    let cancelled = false;
    const base = import.meta.env.BASE_URL;
    const prefix = base.endsWith("/") ? base : `${base}/`;
    fetch(`${prefix}perfumes.json`)
      .then((r) => {
        if (!r.ok) throw new Error(String(r.status));
        return r.json();
      })
      .then((data) => {
        if (!cancelled && data?.perfumes) setPerfumeCatalog(data);
      })
      .catch(() => {
        if (!cancelled) setPerfumeCatalog({ perfumes: [] });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const families = matrix.families;
  const matrixTags = matrix.tags;

  const toggleTag = useCallback((tag) => {
    setSelectedTags((prev) => {
      const next = new Set(prev);
      if (next.has(tag)) next.delete(tag);
      else next.add(tag);
      return next;
    });
    setSubmitted(false);
  }, []);

  const toggleExclude = useCallback((fam) => {
    setExcluded((prev) => {
      const next = new Set(prev);
      if (next.has(fam)) next.delete(fam);
      else next.add(fam);
      return next;
    });
    setSubmitted(false);
  }, []);

  const activeTagList = useMemo(() => [...selectedTags], [selectedTags]);

  const excludedList = useMemo(() => [...excluded], [excluded]);

  const result = useMemo(() => {
    if (!submitted || activeTagList.length === 0) return null;
    return computeScores(matrix, activeTagList, excluded);
  }, [submitted, activeTagList, excluded]);

  const perfumePicks = useMemo(() => {
    const ranked = result?.ranked;
    const list = perfumeCatalog?.perfumes;
    if (!ranked?.length || !list?.length) return [];
    return pickPerfumes(ranked, list, { limit: 8, excludedFamilies: excludedList });
  }, [result, perfumeCatalog, excludedList]);

  const COLOR_TAGS = useMemo(
    () => new Set(["Black", "White", "Grey", "Beige", "Blue", "Brown", "Green", "Red", "Pink", "Burgundy", "Purple", "Yellow", "Orange", "Navy", "Cream"]),
    []
  );
  const STYLE_TAGS = useMemo(() => new Set(["Elegant", "Casual", "Romantic", "Minimalist", "Sporty", "Streetwear", "Boho", "Office", "Evening"]), []);

  const runAiTagging = async () => {
    setAiError("");
    setAiWarning("");
    setAiMode("");
    setAiTopColor([]);
    setAiTopStyle([]);
    setAiPicked({ color: "", style: "" });
    if (files.length === 0) {
      setAiError("Сначала загрузите хотя бы одно фото.");
      return;
    }

    setAiBusy(true);
    try {
      const fd = new FormData();
      fd.append("image", files[0]);

      const res = await fetch(analyzeEndpoint(), {
        method: "POST",
        headers: authHeaders(),
        body: fd,
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data?.detail || `AI service error: ${res.status}`);
      }
      setAiMode(data.model_mode || "");

      const allowed = new Set(matrixTags.map((t) => t.tag));
      const scored = (data.tags || []).filter((x) => allowed.has(x.tag));

      const colors = scored
        .filter((x) => COLOR_TAGS.has(x.tag))
        .sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
      const bestColor = colors[0]?.tag ?? "";

      const styles = scored
        .filter((x) => STYLE_TAGS.has(x.tag))
        .sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
      const bestStyle = styles[0]?.tag ?? "";

      const picked = [bestColor, bestStyle].filter(Boolean);
      if (picked.length === 0) {
        throw new Error(data?.note || "Модель не вернула подходящие теги.");
      }

      const lowConfidence = Boolean(data?.prediction?.low_confidence);
      if (lowConfidence) {
        setAiWarning("Уверенность модели низкая. Проверьте выбранные теги вручную и нажмите «Сопоставить аромат».");
      }

      setAiTopColor(colors.slice(0, 3));
      setAiTopStyle(styles.slice(0, 3));
      setAiPicked({ color: bestColor, style: bestStyle });
      setSelectedTags(new Set(picked));
      // При высокой уверенности сразу показываем рекомендацию. При низкой — ждём подтверждения пользователя.
      setSubmitted(picked.length > 0 && !lowConfidence);
    } catch (e) {
      const hint =
        e?.message?.includes("Failed to fetch") || e?.name === "TypeError"
          ? " Браузер не достучался до сервера (часто сервис не запущен или порт не 8000)."
          : "";
      setAiError(
        `${e?.message || "Не удалось распознать теги."}${hint}`
      );
    } finally {
      setAiBusy(false);
    }
  };

  const runMatch = () => {
    if (activeTagList.length === 0) return;
    setBusy(true);
    window.setTimeout(() => {
      setSubmitted(true);
      setBusy(false);
    }, 520);
  };

  return (
    <div className="app">
      <header className="app-hero">
        <p className="app-brand">Sillage&amp;Style</p>
        <h1 className="app-title">Стиль одежды и семейство аромата</h1>
        <p className="app-lead">
          Экспертная визуализация матрицы весов: теги образа, жёсткие исключения и итоговая рекомендация — всё в браузере, без сервера.
        </p>
      </header>

      <main className="app-grid">
        <div className="app-col app-col--main">
          <ImageUploader files={files} onChange={setFiles} />
          <div className="app-ai">
            <button type="button" className="app-btn app-btn--ghost" disabled={aiBusy} onClick={runAiTagging}>
              {aiBusy ? "Распознаём теги…" : "Распознать теги (ИИ)"}
            </button>
            {aiError && <div className="app-ai-err">{aiError}</div>}
            {aiWarning && <div className="app-ai-warning">{aiWarning}</div>}
            {(aiPicked.color || aiPicked.style) && (
              <div className="app-ai-top">
                <span className="app-ai-top-label">Выбрано ИИ{aiMode ? ` · ${aiMode === "trained_vit" ? "ViT" : "резервная модель"}` : ""}:</span>
                <span className="app-ai-top-list">{[aiPicked.style, aiPicked.color].filter(Boolean).join(" · ")}</span>
              </div>
            )}
            {(aiTopColor.length > 0 || aiTopStyle.length > 0) && (
              <div className="app-ai-top">
                <span className="app-ai-top-label">Топ (для проверки):</span>
                <span className="app-ai-top-list">
                  {aiTopStyle.length > 0 &&
                    `Стиль: ${aiTopStyle.map((x) => `${x.tag} (${Math.round((x.score ?? 0) * 100)}%)`).join(" · ")}`}
                  {aiTopStyle.length > 0 && aiTopColor.length > 0 ? " | " : ""}
                  {aiTopColor.length > 0 &&
                    `Цвет: ${aiTopColor.map((x) => `${x.tag} (${Math.round((x.score ?? 0) * 100)}%)`).join(" · ")}`}
                </span>
              </div>
            )}
          </div>
          <TagPicker matrixTags={matrixTags} selected={selectedTags} onToggle={toggleTag} />
          <PreferenceToggle families={families} excluded={excluded} onToggle={toggleExclude} />

          <div className="app-actions">
            <button type="button" className="app-btn" disabled={activeTagList.length === 0 || busy} onClick={runMatch}>
              {busy ? "Считаем…" : "Сопоставить аромат"}
            </button>
            {activeTagList.length === 0 && <span className="app-hint">Выберите хотя бы один тег</span>}
          </div>
        </div>

        <div className="app-col app-col--side">
          <ResultDisplay
            hasTags={submitted && activeTagList.length > 0}
            winner={result?.winner ?? null}
            ranked={result?.ranked ?? []}
            excludedFamilies={excludedList}
            perfumePicks={perfumePicks}
            perfumeCatalogSize={perfumeCatalog.perfumes?.length ?? 0}
          />
        </div>
      </main>

      <footer className="app-foot">
        <span>Учебный проект · персональный кабинет Sillage&amp;Style</span>
      </footer>
    </div>
  );
}


export default function App() {
  return <AuthGate>{() => <MainApp />}</AuthGate>;
}
