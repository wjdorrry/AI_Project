import { FAMILY_LABEL_RU, FAMILY_DESCRIPTION_RU } from "../content/familyCopy";
import "./ResultDisplay.css";

export default function ResultDisplay({
  winner,
  ranked,
  excludedFamilies,
  hasTags,
  perfumePicks = [],
  perfumeCatalogSize = 0,
}) {
  if (!hasTags) {
    return (
      <aside className="rd rd--empty" aria-live="polite">
        <div className="rd-frame">
          <p className="rd-placeholder">Выберите теги образа и нажмите «Сопоставить аромат».</p>
        </div>
      </aside>
    );
  }

  if (!winner) {
    return (
      <aside className="rd rd--warn" aria-live="polite">
        <div className="rd-frame">
          <h2 className="rd-heading">Нет кандидатов</h2>
          <p className="rd-body">Все семейства исключены ограничениями. Снимите часть переключателей.</p>
        </div>
      </aside>
    );
  }

  const label = FAMILY_LABEL_RU[winner.family] ?? winner.family;
  const desc = FAMILY_DESCRIPTION_RU[winner.family] ?? "";

  const top = ranked.slice(0, 5);

  return (
    <aside className="rd" aria-live="polite">
      <div className="rd-glow" aria-hidden />
      <div className="rd-frame">
        <p className="rd-kicker">Рекомендуемое семейство</p>
        <h2 className="rd-title">{label}</h2>
        <p className="rd-en">{winner.family}</p>
        <p className="rd-desc">{desc}</p>
        <div className="rd-score">
          <span className="rd-score-label">Итоговый балл</span>
          <span className="rd-score-val">{winner.score}</span>
        </div>

        {excludedFamilies.length > 0 && (
          <div className="rd-excluded">
            <span className="rd-excluded-label">Исключено:</span>
            <span className="rd-excluded-list">
              {excludedFamilies.map((f) => FAMILY_LABEL_RU[f] ?? f).join(" · ")}
            </span>
          </div>
        )}

        <div className="rd-bars">
          <h3 className="rd-bars-title">Топ по сумме весов</h3>
          <ul className="rd-bars-list">
            {top.map((row) => {
              const max = Math.max(...top.map((r) => r.score), 1);
              const pct = Math.max(8, (row.score / max) * 100);
              return (
                <li key={row.family} className="rd-bar-row">
                  <span className="rd-bar-name">{FAMILY_LABEL_RU[row.family] ?? row.family}</span>
                  <div className="rd-bar-track">
                    <div
                      className={`rd-bar-fill ${row.family === winner.family ? "rd-bar-fill--win" : ""}`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <span className="rd-bar-num">{row.score}</span>
                </li>
              );
            })}
          </ul>
        </div>

        <div className="rd-perf">
          <h3 className="rd-perf-title">Подборка из базы</h3>
          {perfumePicks.length === 0 ? (
            <p className="rd-perf-empty">
              {perfumeCatalogSize === 0 ? (
                <>
                  Каталог не загружен. Нужен файл <code className="rd-code">public/perfumes.json</code> (команда{" "}
                  <code className="rd-code">py scripts/export_perfumes_to_json.py</code> из{" "}
                  <code className="rd-code">perfume_database_cleaned.xlsx</code>).
                </>
              ) : (
                <>
                  При таких тегах и исключениях подходящих ароматов в базе не нашлось: либо у всех записей в данных есть
                  одно из исключённых семейств, либо совпадений по оставшимся семействам нет. Попробуйте снять часть
                  переключателей «Без …».
                </>
              )}
            </p>
          ) : (
            <ul className="rd-perf-list">
              {perfumePicks.map((p) => (
                <li key={p.id} className="rd-perf-item">
                  <div className="rd-perf-head">
                    <span className="rd-perf-brand">{p.brand}</span>
                    <span className="rd-perf-name">{p.name}</span>
                  </div>
                  {Array.isArray(p.families) && p.families.length > 0 && (
                    <div className="rd-perf-fams">
                      {p.families.map((f) => (
                        <span key={f} className="rd-perf-chip">
                          {FAMILY_LABEL_RU[f] ?? f}
                        </span>
                      ))}
                    </div>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </aside>
  );
}
