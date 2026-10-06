import { FAMILY_LABEL_RU, EXCLUSION_PREFIX } from "../content/familyCopy";
import "./PreferenceToggle.css";

export default function PreferenceToggle({ families, excluded, onToggle }) {
  return (
    <section className="pt" aria-labelledby="pt-title">
      <div className="pt-head">
        <h2 id="pt-title" className="pt-title">
          Предпочтения
        </h2>
        <p className="pt-sub">Жёсткие ограничения: отмеченные семейства не участвуют в подборе, даже при высоких баллах по матрице.</p>
      </div>
      <div className="pt-grid" role="group" aria-label="Исключить семейства">
        {families.map((key) => {
          const on = excluded.has(key);
          return (
            <button
              key={key}
              type="button"
              className={`pt-chip ${on ? "pt-chip--on" : ""}`}
              aria-pressed={on}
              onClick={() => onToggle(key)}
            >
              <span className="pt-chip-label">
                {EXCLUSION_PREFIX}
                {FAMILY_LABEL_RU[key] ?? key}
              </span>
            </button>
          );
        })}
      </div>
    </section>
  );
}
