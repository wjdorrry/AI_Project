const TOKEN_KEY = "sillage_access_token";

export function getToken() {
  return localStorage.getItem(TOKEN_KEY) || "";
}

export function setToken(token) {
  if (token) localStorage.setItem(TOKEN_KEY, token);
  else localStorage.removeItem(TOKEN_KEY);
}

export function authHeaders(extra = {}) {
  const token = getToken();
  return token ? { ...extra, Authorization: `Bearer ${token}` } : extra;
}

export async function apiJson(path, options = {}) {
  const headers = authHeaders({
    "Content-Type": "application/json",
    ...(options.headers || {}),
  });

  const res = await fetch(path, { ...options, headers });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const message = data?.detail || "Ошибка сервера.";
    const err = new Error(message);
    err.status = res.status;
    throw err;
  }
  return data;
}
