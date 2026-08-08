import { useEffect, useState } from "react";
import { api } from "../api";
import StatCard from "./StatCard";

function fmt(n) {
  if (n === null || n === undefined) return "—";
  return n.toLocaleString();
}

export default function Overview() {
  const [dataset, setDataset] = useState(null);
  const [models, setModels] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const [ds, ms] = await Promise.all([api.dataset(), api.models()]);
        if (!cancelled) {
          setDataset(ds);
          setModels(ms);
        }
      } catch (e) {
        if (!cancelled) setError(e.message);
      }
    };
    load();
    const interval = setInterval(load, 20000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  if (error) return <div className="error-box">Failed to reach the API: {error}</div>;
  if (!dataset || !models) return <div className="loading-line">Loading...</div>;

  const qc = dataset.cds_qc || {};
  const trna = dataset.trna_qc || {};
  const readiness = qc.reviewer_readiness || {};

  return (
    <>
      <div className="panel">
        <h2>What this is</h2>
        <div className="desc">
          MitoSeqGen is a constrained sequence-to-sequence transformer that generates
          codon-optimized mammalian mitochondrial mRNA coding sequences (mt-CDS) from a
          target protein, respecting the mitochondrial genetic code (NCBI translation
          table 2: UGA=Trp, AGA/AGG=stop, AUA=Met). Every generated sequence is
          protein-exact and internal-stop-free by construction (constrained decoding),
          and is evaluated against 4 baselines and real natural sequences using
          mt-CAI, MFE (ViennaRNA), GC-content, codon diversity, and BLEU-4, with
          paired Wilcoxon significance testing.
        </div>
      </div>

      <div className="panel">
        <h2>Dataset (NCBI Entrez, vertebrate mitochondrial genomes)</h2>
        <div className="desc">Reviewer-grade 8-layer QC pipeline; phylogenetic (species-level) train/val/test split.</div>
        <div className="grid">
          <StatCard label="CDS records passed QC" value={fmt(qc.total_passed_qc)} hint={`of ${fmt(qc.total_downloaded)} downloaded`} />
          <StatCard label="Unique species" value={fmt(qc.species_count)} hint={readiness.species_diversity?.reviewer_note} />
          <StatCard label="Genes covered" value={fmt(Object.keys(qc.gene_distribution || {}).length)} hint="all 13 mt-protein-coding genes" />
          <StatCard label="tRNA records passed QC" value={fmt(trna.total_passed_qc)} hint={trna.overall_status} />
        </div>
      </div>

      <div className="panel">
        <h2>Train / val / test split</h2>
        <div className="desc">Species-level (phylogenetic) stratification — no species appears in more than one split.</div>
        <div className="grid">
          <StatCard label="Train" value={fmt(dataset.splits.train)} />
          <StatCard label="Validation" value={fmt(dataset.splits.val)} hint="unseen during training" />
          <StatCard label="Test" value={fmt(dataset.splits.test)} hint="unseen species entirely" />
        </div>
      </div>

      <div className="panel">
        <h2>Models</h2>
        <div className="grid">
          {models.map((m) => (
            <div className="stat-card" key={m.model_id}>
              <div className="label">{m.model_id}</div>
              <div className="value" style={{ fontSize: "1rem" }}>{m.label}</div>
              <div className="hint">
                <span className={`badge ${m.status}`}>{m.status}</span>{" "}
                epoch {m.epochs_completed}/{m.configured_epochs} &middot; best val_loss {m.best_val_loss?.toFixed(4) ?? "—"}
              </div>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}
