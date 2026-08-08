import { useEffect, useState } from "react";
import { api } from "../api";

const METHOD_LABELS = {
  mitoseqgen: "MitoSeqGen",
  mt_cai_lookup: "mt-CAI lookup",
  random_synonymous: "Random synonymous",
  most_frequent_codon: "Most frequent codon",
  codontransformer_remap: "CodonTransformer-remap",
  natural_reference: "Natural (ground truth)",
};

const METRIC_ROWS = [
  ["mean_mt_cai", "Mean mt-CAI", 4],
  ["compliance_rate", "Genetic code compliance", 3],
  ["mean_bleu", "BLEU-4 vs. real sequence", 4],
  ["mean_codon_diversity", "Codon diversity (bits)", 3],
  ["mean_mfe", "Mean MFE (kcal/mol)", 2],
  ["mean_mfe_delta_from_natural", "|MFE − natural|", 2],
  ["mean_gc_content", "Mean GC content", 4],
  ["mean_gc_delta_from_natural", "|GC% − natural|", 4],
  ["novel_sequence_rate", "Novel sequence rate", 3],
];

export default function EvaluationView() {
  const [modelId, setModelId] = useState("model1");
  const [models, setModels] = useState([]);
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.models().then(setModels).catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    setReport(null);
    setError(null);
    api.evaluation(modelId).then(setReport).catch((e) => setError(e.message));
  }, [modelId]);

  const evaluated = models.filter((m) => m.has_evaluation);

  return (
    <div className="panel">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
        <h2>Evaluation vs. required baselines</h2>
        <select value={modelId} onChange={(e) => setModelId(e.target.value)}>
          {(evaluated.length ? evaluated : models).map((m) => (
            <option key={m.model_id} value={m.model_id}>{m.model_id}</option>
          ))}
        </select>
      </div>
      <div className="desc">
        Held-out test-set proteins (species never seen in train/val). Per project rule, no
        metric is reported without its baseline — 4 baselines plus the real natural sequence
        are generated for the same proteins.
      </div>

      {error && <div className="error-box">{error}</div>}
      {!report && !error && <div className="loading-line">No evaluation report available for {modelId} yet.</div>}

      {report && (
        <>
          <div className="loading-line" style={{ marginBottom: 10 }}>
            n = {report.sample_size} held-out proteins
          </div>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Metric</th>
                  {Object.keys(report.summary).map((name) => (
                    <th key={name} className={name === "mitoseqgen" ? "highlight-col" : ""}>
                      {METHOD_LABELS[name] || name}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {METRIC_ROWS.map(([key, label, decimals]) => (
                  <tr key={key}>
                    <td>{label}</td>
                    {Object.keys(report.summary).map((name) => (
                      <td key={name} className={name === "mitoseqgen" ? "highlight-col" : ""}>
                        {report.summary[name][key]?.toFixed(decimals) ?? "—"}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h2 style={{ marginTop: 22 }}>Statistical significance (Wilcoxon signed-rank, Bonferroni-corrected)</h2>
          <div className="desc">
            &alpha; = 0.05 / {report.significance_tests.num_comparisons} comparisons = {report.significance_tests.bonferroni_corrected_alpha.toFixed(5)}
          </div>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Baseline</th>
                  <th>Metric</th>
                  <th>MitoSeqGen mean</th>
                  <th>Baseline mean</th>
                  <th>p-value</th>
                  <th>Significant</th>
                </tr>
              </thead>
              <tbody>
                {report.significance_tests.comparisons.map((c, i) => (
                  <tr key={i}>
                    <td>{METHOD_LABELS[c.baseline] || c.baseline}</td>
                    <td>{c.metric}</td>
                    <td>{c.model_mean.toFixed(4)}</td>
                    <td>{c.baseline_mean.toFixed(4)}</td>
                    <td>{c.p_value < 0.0001 ? c.p_value.toExponential(2) : c.p_value.toFixed(4)}</td>
                    <td className={c.significant_after_correction ? "sig-yes" : "sig-no"}>
                      {c.significant_after_correction ? "yes" : "no"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
