import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api";

const MODEL_IDS = ["model1", "model2", "model3", "model4"];

export default function TrainingView() {
  const [modelId, setModelId] = useState("model4");
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const s = await api.training(modelId);
        if (!cancelled) setStatus(s);
      } catch (e) {
        if (!cancelled) setError(e.message);
      }
    };
    load();
    const interval = setInterval(load, 15000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [modelId]);

  return (
    <div className="panel">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
        <h2>Training curve</h2>
        <select value={modelId} onChange={(e) => setModelId(e.target.value)}>
          {MODEL_IDS.map((id) => (
            <option key={id} value={id}>{id}</option>
          ))}
        </select>
      </div>
      <div className="desc">
        {modelId === "model1" &&
          "Cross-entropy only baseline."}
        {modelId === "model2" &&
          "CE + batch-pooled GC hinge loss + soft mt-CAI reward. Post-hoc finding: the GC loss logged ~0 for all 28 epochs (batch-average GC rarely left the target band even though individual sequences did) — effectively inert."}
        {modelId === "model3" &&
          "CE + per-sequence GC-matching loss, computed over the unrestricted full vocabulary. Loss curve looked healthy during training but eval got worse on every metric that matters — the model 'solved' the loss by shifting probability onto biologically-invalid codons the constrained decoder can never select."}
        {modelId === "model4" &&
          "CE + per-position GC-matching loss restricted to the true target's synonym class (same masking as the mt-CAI loss). Third attempt at fixing the GC-deviation weakness. Live — refreshes every 15s."}
      </div>

      {error && <div className="error-box">{error}</div>}
      {!status && !error && <div className="loading-line">Loading...</div>}

      {status && (
        <>
          <div className="grid" style={{ marginBottom: 16 }}>
            <div className="stat-card">
              <div className="label">Status</div>
              <div className="value" style={{ fontSize: "1rem" }}>
                <span className={`badge ${status.status}`}>{status.status}</span>
              </div>
            </div>
            <div className="stat-card">
              <div className="label">Epochs</div>
              <div className="value">{status.epochs_completed}/{status.configured_epochs}</div>
            </div>
            <div className="stat-card">
              <div className="label">Best val_loss</div>
              <div className="value">{status.best_val_loss?.toFixed(4) ?? "—"}</div>
              <div className="hint">epoch {status.best_epoch}</div>
            </div>
          </div>

          {status.history.length > 0 && (
            <div style={{ width: "100%", height: 320 }}>
              <ResponsiveContainer>
                <LineChart data={status.history} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#232b36" />
                  <XAxis dataKey="epoch" stroke="#8b98a5" fontSize={12} />
                  <YAxis stroke="#8b98a5" fontSize={12} />
                  <Tooltip contentStyle={{ background: "#161d26", border: "1px solid #232b36", fontSize: 12 }} />
                  <Legend />
                  <Line type="monotone" dataKey="train_loss" stroke="#4fd1c5" dot={false} strokeWidth={2} name="train_loss" />
                  <Line type="monotone" dataKey="val_loss" stroke="#e8b339" dot={false} strokeWidth={2} name="val_loss" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          <div className="loading-line" style={{ marginTop: 10 }}>{status.latest_log_line}</div>
        </>
      )}
    </div>
  );
}
