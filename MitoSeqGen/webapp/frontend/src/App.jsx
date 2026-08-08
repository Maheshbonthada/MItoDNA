import { useState } from "react";
import Overview from "./components/Overview";
import TrainingView from "./components/TrainingView";
import EvaluationView from "./components/EvaluationView";
import TryItView from "./components/TryItView";

const TABS = [
  { id: "overview", label: "Overview", Component: Overview },
  { id: "training", label: "Training", Component: TrainingView },
  { id: "evaluation", label: "Evaluation", Component: EvaluationView },
  { id: "tryit", label: "Try It", Component: TryItView },
];

function App() {
  const [tab, setTab] = useState("overview");
  const Active = TABS.find((t) => t.id === tab).Component;

  return (
    <div className="app">
      <div className="header">
        <h1>MitoSeqGen</h1>
        <div className="subtitle">
          A generative deep-learning model for mammalian mitochondrial mRNA codon
          optimization — constrained transformer generation, multi-baseline evaluation,
          and statistical significance testing, built from scratch on a single 8GB GPU.
        </div>
        <div className="tabs">
          {TABS.map((t) => (
            <button
              key={t.id}
              className={`tab ${tab === t.id ? "active" : ""}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>

      <Active />

      <div className="footer-note">
        Research prototype — not a clinical or production tool. Open source.
      </div>
    </div>
  );
}

export default App;
