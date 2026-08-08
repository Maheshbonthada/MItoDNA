const BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

async function req(path, options) {
  const res = await fetch(`${BASE}${path}`, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

export const api = {
  health: () => req("/api/health"),
  models: () => req("/api/models"),
  training: (modelId) => req(`/api/training/${modelId}`),
  evaluation: (modelId) => req(`/api/evaluation/${modelId}`),
  dataset: () => req("/api/dataset"),
  samples: () => req("/api/samples"),
  generate: (payload) =>
    req("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
};
