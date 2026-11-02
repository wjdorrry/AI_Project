import "./SeasonPicker.css";

const SEASONS = [
  { tag: "Spring", label: "Весна", hint: "лёгкие, зелёные и цветочные акценты" },
  { tag: "Summer", label: "Лето", hint: "свежесть, цитрусы и водные ноты" },
  { tag: "Autumn", label: "Осень", hint: "древесность, пряность и тепло" },
  { tag: "Winter", label: "Зима", hint: "насыщенные, тёплые и глубокие ноты" },
];

export default function SeasonPicker({ value, onChange }) {
  return (
    <section className="sp" aria-labelledby="sp-title">
      <div className="sp-head">
        <h2 id="sp-title" className="sp-title">Сезон</h2>
        <p className="sp-sub">
          Выберите, для какого времени года подбирается аромат. Сезон участвует в итоговом расчёте вместе с цветом и стилем.
        </p>
      </div>

      <div className="sp-grid">
        {SEASONS.map((season) => {
          const active = value === season.tag;
          return (
            <button
              key={season.tag}
              type="button"
              className={`sp-item ${active ? "sp-item--active" : ""}`}
              aria-pressed={active}
              onClick={() => onChange(season.tag)}
            >
              <span className="sp-name">{season.label}</span>
              <span className="sp-hint">{season.hint}</span>
            </button>
          );
        })}
      </div>
    </section>
  );
}
