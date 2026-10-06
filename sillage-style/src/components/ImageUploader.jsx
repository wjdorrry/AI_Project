import { useCallback, useEffect, useRef, useState } from "react";
import "./ImageUploader.css";

const MAX = 3;

export default function ImageUploader({ files, onChange }) {
  const inputRef = useRef(null);
  const [urls, setUrls] = useState([]);

  useEffect(() => {
    const next = files.map((f) => URL.createObjectURL(f));
    setUrls(next);
    return () => next.forEach((u) => URL.revokeObjectURL(u));
  }, [files]);

  const addFiles = useCallback(
    (list) => {
      const incoming = Array.from(list).filter((f) => f.type.startsWith("image/"));
      const merged = [...files];
      for (const f of incoming) {
        if (merged.length >= MAX) break;
        merged.push(f);
      }
      onChange(merged);
    },
    [files, onChange]
  );

  const onInput = (e) => {
    if (e.target.files?.length) addFiles(e.target.files);
    e.target.value = "";
  };

  const onDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files?.length) addFiles(e.dataTransfer.files);
  };

  const removeAt = (i) => {
    const next = files.filter((_, j) => j !== i);
    onChange(next);
  };

  return (
    <section className="iu" aria-labelledby="iu-title">
      <div className="iu-head">
        <h2 id="iu-title" className="iu-title">
          Образ
        </h2>
        <p className="iu-sub">До {MAX} фото — превью только в браузере, на сервер ничего не уходит.</p>
      </div>

      <div
        className="iu-drop"
        onDragOver={(e) => e.preventDefault()}
        onDrop={onDrop}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          multiple
          className="iu-input"
          onChange={onInput}
          aria-label="Выбрать изображения"
        />
        <span className="iu-drop-icon" aria-hidden>
          ◇
        </span>
        <span className="iu-drop-text">Перетащите сюда или нажмите для выбора</span>
        <span className="iu-drop-hint">PNG, JPG, WebP</span>
      </div>

      {files.length > 0 && (
        <ul className="iu-previews">
          {files.map((file, i) => (
            <li key={`${file.name}-${file.size}-${i}`} className="iu-slot">
              <img src={urls[i]} alt={`Превью ${i + 1}`} className="iu-thumb" />
              <button type="button" className="iu-remove" onClick={() => removeAt(i)} aria-label="Удалить фото">
                ×
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
