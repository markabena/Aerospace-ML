# Aerospace ML

**Machine learning for flight systems: predictive maintenance, telemetry anomaly detection, CFD surrogate modeling, and LLM tools for aviation.**

Built by **Mark Abena**, B.Eng Aerospace Engineering (AFIT Kaduna, 2026). This repository is where aerospace engineering meets applied machine learning. Every project uses real, public flight or engineering data and follows the same standard: a clear problem, a documented method, measurable results, and reproducible code.

---

## Projects

| # | Project | Domain | Methods | Status |
|---|---------|--------|---------|--------|
| 01 | [Turbofan Remaining Useful Life Predictor](projects/01_turbofan_rul/) | Predictive maintenance | pandas, scikit-learn, LSTM (PyTorch) | 🟡 Planned |
| 02 | [Spacecraft Telemetry Anomaly Detection](projects/02_telemetry_anomaly/) | Flight operations | LSTM / autoencoder, time series | 🟡 Planned |
| 03 | [CFD Icing Surrogate Model](projects/03_cfd_icing_surrogate/) | Aerodynamics | Regression, neural surrogates | 🟡 Planned |
| 04 | [Aviation Regulations RAG Assistant](projects/04_aviation_rag/) | Aviation compliance | Embeddings, retrieval, LLMs | 🟡 Planned |

Status key: 🟡 Planned · 🔵 In progress · 🟢 Complete

---

## Repository Structure

```
aerospace-ml/
├── projects/            # One self-contained folder per project
│   └── NN_project_name/
│       ├── README.md    # Problem, data, method, results
│       ├── notebooks/   # Exploration and analysis
│       ├── src/         # Reusable, tested code
│       └── results/     # Figures and metrics
├── data/                # Local datasets (not committed; see data/README.md)
├── docs/                # Learning log and notes
├── tests/               # Repository-wide tests (run in CI)
└── requirements.txt
```

---

## Setup

Requires **Python 3.11**.

```bash
git clone https://github.com/markabena/Aerospace-ML.git
cd Aerospace-ML
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest
```

Deep learning projects (01, 02) are trained on free GPU compute (Kaggle Notebooks / Google Colab).

---

## Tech Stack

Python 3.11 · NumPy · pandas · matplotlib · scikit-learn · PyTorch · Jupyter · MATLAB R2023a (via CSV interchange)

---

## License

MIT. See [LICENSE](LICENSE).
