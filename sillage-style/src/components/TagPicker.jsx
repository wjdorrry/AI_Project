import "./TagPicker.css";

/** Демо-имитация тегов с «визуального анализа» — без API всё выбирается вручную. */
export default function TagPicker({ matrixTags, selected, onToggle }) {
  const byCat = matrixTags.reduce((acc, t) => {
    const c = t.categoryRu || "Прочее";
    if (!acc[c]) acc[c] = [];
    acc[c].push(t);
    return acc;
  }, {});

  return (
    <section className="tp" aria-labelledby="tp-title">
      <div className="tp-head">
        <h2 id="tp-title" className="tp-title">
          Теги образа
        </h2>
        <p className="tp-sub">Без бэкенда CV: отметьте то, что видите на фото — как будто это вернул API распознавания.</p>
      </div>
      {Object.entries(byCat).map(([cat, items]) => (
        <div key={cat} className="tp-block">
          <h3 className="tp-cat">{cat}</h3>
          <div className="tp-row">
            {items.map((t) => {
              const on = selected.has(t.tag);
              return (
                <button
                  key={t.tag}
                  type="button"
                  className={`tp-tag ${on ? "tp-tag--on" : ""}`}
                  aria-pressed={on}
                  onClick={() => onToggle(t.tag)}
                >
                  {t.tag}
                </button>
              );
            })}
          </div>
        </div>
      ))}
    </section>
  );
}
