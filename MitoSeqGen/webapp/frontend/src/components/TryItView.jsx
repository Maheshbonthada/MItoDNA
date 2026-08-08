import { useEffect, useState } from "react";
import { api } from "../api";

const METHOD_LABELS = {
  mt_cai_lookup: "mt-CAI lookup",
  random_synonymous: "Random synonymous",
  most_frequent_codon: "Most frequent codon",
  codontransformer_remap: "CodonTransformer-remap",
};

function MetricsTable({ generated, reference, baselines }) {
  const cols = [
    ["generated", "MitoSeqGen", generated],
    ...(reference ? [["reference", "Natural (reference)", reference]] : []),
    ...(baselines ? Object.entries(baselines).map(([k, v]) => [k, METHOD_LABELS[k] || k, v.metrics]) : []),
  ];
  const rows = [
    ["mt_cai", "mt-CAI", 4],
    ["gc_content", "GC content", 4],
    ["mfe", "MFE (kcal/mol)", 2],
    ["codon_diversity", "Codon diversity (bits)", 3],
    ["bleu", "BLEU-4 vs. reference", 4],
    ["code_compliant", "Code compliant", 0],
  ];
  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            <th>Metric</th>
            {cols.map(([key, label]) => (
              <th key={key} className={key === "generated" ? "highlight-col" : ""}>{label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(([key, label, decimals]) => (
            <tr key={key}>
              <td>{label}</td>
              {cols.map(([colKey, , metrics]) => (
                <td key={colKey} className={colKey === "generated" ? "highlight-col" : ""}>
                  {key === "code_compliant"
                    ? (metrics?.[key] ? "yes" : "no")
                    : metrics?.[key] !== undefined
                      ? metrics[key].toFixed(decimals)
                      : "—"}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function TryItView() {
  const [models, setModels] = useState([]);
  const [samples, setSamples] = useState([]);
  const [modelId, setModelId] = useState("model1");
  const [inputMode, setInputMode] = useState("sample");
  const [sampleId, setSampleId] = useState("");
  const [customProtein, setCustomProtein] = useState("");
  const [includeBaselines, setIncludeBaselines] = useState(true);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.models().then((ms) => {
      setModels(ms);
      const usable = ms.find((m) => m.epochs_completed > 0);
      if (usable) setModelId(usable.model_id);
    });
    api.samples().then((s) => {
      setSamples(s);
      if (s.length) setSampleId(s[0].id);
    });
  }, []);

  const runGenerate = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const payload = { model_id: modelId, include_baselines: includeBaselines };
      if (inputMode === "sample") payload.sample_id = sampleId;
      else payload.protein_sequence = customProtein;
      const r = await api.generate(payload);
      setResult(r);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="panel">
      <h2>Try it — generate a codon-optimized CDS</h2>
      <div className="desc">
        Runs on CPU so it never competes with GPU training in the background. Generation is
        constrained decoding: the output is guaranteed protein-exact and free of internal
        stop codons by construction, not by post-hoc filtering.
      </div>

      <div className="field-row">
        <div className="field">
          <label>Model</label>
          <select value={modelId} onChange={(e) => setModelId(e.target.value)}>
            {models.map((m) => (
              <option key={m.model_id} value={m.model_id} disabled={m.epochs_completed === 0}>
                {m.model_id} — {m.label} {m.epochs_completed === 0 ? "(no checkpoint yet)" : ""}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>Input</label>
          <select value={inputMode} onChange={(e) => setInputMode(e.target.value)}>
            <option value="sample">Held-out test sample</option>
            <option value="custom">Custom protein sequence</option>
          </select>
        </div>
        <label className="checkbox">
          <input type="checkbox" checked={includeBaselines} onChange={(e) => setIncludeBaselines(e.target.checked)} />
          Compare against baselines
        </label>
      </div>

      {inputMode === "sample" ? (
        <div className="field" style={{ marginBottom: 12 }}>
          <label>Sample protein (never seen during training)</label>
          <select value={sampleId} onChange={(e) => setSampleId(e.target.value)}>
            {samples.map((s) => (
              <option key={s.id} value={s.id}>
                {s.gene_name} — {s.species} ({s.protein_sequence.length} aa)
              </option>
            ))}
          </select>
        </div>
      ) : (
        <div className="field" style={{ marginBottom: 12 }}>
          <label>Protein sequence (amino acid letters, max 700)</label>
          <textarea
            value={customProtein}
            onChange={(e) => setCustomProtein(e.target.value)}
            placeholder="MFADRWLFSTVL..."
          />
        </div>
      )}

      <button className="primary" onClick={runGenerate} disabled={loading || (inputMode === "custom" && !customProtein.trim())}>
        {loading ? "Generating (CPU, up to ~10s)..." : "Generate"}
      </button>

      {error && <div className="error-box" style={{ marginTop: 14 }}>{error}</div>}

      {result && (
        <div style={{ marginTop: 18 }}>
          <div className="loading-line" style={{ marginBottom: 6 }}>
            {result.input_label} &middot; generated in {result.generation_seconds.toFixed(2)}s
          </div>
          <div className="field" style={{ marginBottom: 10 }}>
            <label>Generated CDS ({result.generated_cds.length} nt)</label>
            <div className="seq-block">{result.generated_cds}</div>
          </div>
          {result.reference_cds && (
            <div className="field" style={{ marginBottom: 10 }}>
              <label>Reference (real evolved sequence)</label>
              <div className="seq-block">{result.reference_cds}</div>
            </div>
          )}
          <MetricsTable
            generated={result.metrics}
            reference={result.reference_metrics}
            baselines={result.baselines}
          />
        </div>
      )}
    </div>
  );
}
