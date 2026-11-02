import { FAMILY_LABEL_RU, FAMILY_DESCRIPTION_RU } from "../content/familyCopy";
import "./ResultDisplay.css";

const TAG_LABEL_RU = {
  Spring: "Весна",
  Summer: "Лето",
  Autumn: "Осень",
  Winter: "Зима",
};

function factorLabel(item) {
  return TAG_LABEL_RU[item.tag] ?? item.tag;
}

function signed(value) {
  return value > 0 ? `+${value}` : String(value);
}

export default function ResultDisplay({
  winner,
  ranked,
  excludedFamilies,
  hasTags,
  perfumePicks = [],
  perfumeCatalogSize = 0,
  explanation = null,
}) {
  if (!hasTags) {
    return (
      <aside className="rd rd--empty" aria-live="polite">
        <div className="rd-frame">
          <p className="rd-placeholder">Выберите цвет, стиль и сезон, затем нажмите «Сопоставить аромат».</p>
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

        {explanation && (
          <div className="rd-why">
            <h3 className="rd-why-title">Почему этот аромат?</h3>
            <p className="rd-why-text">
              Семейство «{label}» получило самый высокий итоговый балл. Ниже показано, как каждый фактор повлиял на этот результат.
            </p>

            <ul className="rd-why-list">
              {explanation.factors.map((item) => (
                <li key={`${item.categoryRu}-${item.tag}`} className="rd-why-row">
                  <span className="rd-why-factor">
                    <span className="rd-why-cat">{item.categoryRu}</span>
                    <span className="rd-why-name">{factorLabel(item)}</span>
                  </span>
                  <span className={`rd-why-score ${item.score >= 0 ? "rd-why-score--plus" : "rd-why-score--minus"}`}>
                    {signed(item.score)}
                  </span>
                </li>
              ))}
            </ul>

            {explanation.positive?.length > 0 && (
              <p className="rd-why-summary">
                Сильнее всего результат поддержали:{" "}
                {explanation.positive
                  .slice(0, 3)
                  .map((item) => `${factorLabel(item)} (${signed(item.score)})`)
                  .join(" · ")}.
              </p>
            )}

            {explanation.negative?.length > 0 && (
              <p className="rd-why-summary rd-why-summary--muted">
                Снижали итоговый балл:{" "}
                {explanation.negative
                  .map((item) => `${factorLabel(item)} (${signed(item.score)})`)
                  .join(" · ")}.
              </p>
            )}
          </div>
        )}

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

                  {Array.isArray(p.matchedFamilies) && p.matchedFamilies.length > 0 && (
                    <p className="rd-perf-why">
                      Почему в подборке: совпадает с{" "}
                      {p.matchedFamilies
                        .slice(0, 3)
                        .map((item) => FAMILY_LABEL_RU[item.family] ?? item.family)
                        .join(", ")}.
                    </p>
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
