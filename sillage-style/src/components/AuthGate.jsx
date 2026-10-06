import { useEffect, useState } from "react";
import { apiJson, getToken, setToken } from "../utils/api";
import "./AuthGate.css";

export default function AuthGate({ children }) {
  const [mode, setMode] = useState("login");
  const [user, setUser] = useState(null);
  const [checking, setChecking] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [form, setForm] = useState({ display_name: "", email: "", password: "" });

  useEffect(() => {
    const token = getToken();
    if (!token) {
      setChecking(false);
      return;
    }
    apiJson("/auth/me")
      .then((data) => setUser(data.user))
      .catch(() => setToken(""))
      .finally(() => setChecking(false));
  }, []);

  const update = (field) => (e) => {
    setForm((prev) => ({ ...prev, [field]: e.target.value }));
  };

  const submit = async (e) => {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      const payload =
        mode === "register"
          ? form
          : { email: form.email, password: form.password };

      const data = await apiJson(`/auth/${mode === "register" ? "register" : "login"}`, {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setToken(data.access_token);
      setUser(data.user);
    } catch (err) {
      setError(err.message || "Не удалось войти.");
    } finally {
      setBusy(false);
    }
  };

  const logout = () => {
    setToken("");
    setUser(null);
    setForm({ display_name: "", email: "", password: "" });
    setMode("login");
  };

  if (checking) {
    return (
      <div className="auth-shell">
        <div className="auth-card auth-card--loading">Проверяем сессию…</div>
      </div>
    );
  }

  if (!user) {
    return (
      <div className="auth-shell">
        <div className="auth-card">
          <p className="auth-brand">Sillage&amp;Style</p>
          <h1 className="auth-title">{mode === "register" ? "Создать аккаунт" : "С возвращением"}</h1>
          <p className="auth-sub">
            {mode === "register"
              ? "Зарегистрируйтесь, чтобы сохранять подборки и позже собрать свою персональную полочку ароматов."
              : "Войдите в личный кабинет, чтобы продолжить работу с вашими рекомендациями."}
          </p>

          <div className="auth-tabs">
            <button type="button" className={mode === "login" ? "is-active" : ""} onClick={() => { setMode("login"); setError(""); }}>
              Вход
            </button>
            <button type="button" className={mode === "register" ? "is-active" : ""} onClick={() => { setMode("register"); setError(""); }}>
              Регистрация
            </button>
          </div>

          <form className="auth-form" onSubmit={submit}>
            {mode === "register" && (
              <label>
                <span>Имя</span>
                <input value={form.display_name} onChange={update("display_name")} minLength={2} maxLength={60} required />
              </label>
            )}
            <label>
              <span>Email</span>
              <input type="email" value={form.email} onChange={update("email")} required />
            </label>
            <label>
              <span>Пароль</span>
              <input type="password" value={form.password} onChange={update("password")} minLength={8} required />
            </label>
            {mode === "register" && <p className="auth-note">Минимум 8 символов, хотя бы одна буква и одна цифра.</p>}
            {error && <div className="auth-error">{error}</div>}
            <button className="auth-submit" type="submit" disabled={busy}>
              {busy ? "Подождите…" : mode === "register" ? "Создать аккаунт" : "Войти"}
            </button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <>
      <div className="account-bar">
        <div>
          <span className="account-label">Личный кабинет</span>
          <strong>{user.display_name}</strong>
          <span className="account-email">{user.email}</span>
        </div>
        <button type="button" onClick={logout}>Выйти</button>
      </div>
      {children({ user })}
    </>
  );
}
