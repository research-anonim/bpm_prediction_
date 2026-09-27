# Topology-Conditioned Process Drift Prediction

This repository provides the replication code and experimental toolkit for predictive business process monitoring under concept drift and structural process evolution. It implements topology-conditioned Graph Neural Networks (GNNs) and structural prior routing (EOPKG) to predict next activities across changing process versions with dynamic vocabularies.

---

## Quick Start & Reproduction Guide

### 0. Prerequisites
- **Python:** Python **3.10** or **3.11** (tested on Python 3.11.9).
- **OS:** Windows, Linux, or macOS.

---

### 1. Environment Setup

Create and activate a clean virtual environment, then install dependencies:

**On Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on PyTorch / PyG:** If your environment requires prebuilt wheels for PyTorch Geometric (PyG) or CUDA-accelerated PyTorch, please install the appropriate build according to the [official PyG instructions](https://pytorch-geometric.readthedocs.io/).

---

### 2. Extract Dataset

The primary simulation event log (`loan_v1_v5_complex_simulated.xes`) is packaged as a `.zip` archive inside `data/loan/`. Extract it before running experiments:

**Using Python (universal for any OS):**
```bash
python -c "import zipfile; zipfile.ZipFile('data/loan/loan_v1_v5_complex_simulated.zip').extractall('data/loan')"
```

*Or extract `data/loan/loan_v1_v5_complex_simulated.zip` using your preferred archive tool so that `data/loan/loan_v1_v5_complex_simulated.xes` is present.*

---

### 3. Automated Benchmark Execution (Headless CLI)

To execute an ordered queue of benchmark experiments in headless mode:

```bash
python tools/run_benchmark.py
```

#### Selecting Experiments in the Queue:
The queue of experiments to be executed is defined in **`configs/ui/cdlg_benchmark_plan.yaml`**. 

You can edit this YAML file to keep only the specific runs you wish to reproduce. For example, to test one Loan process drift scenario and one CDLG benchmark scenario:

```yaml
runs:
  # Loan dataset drift scenario (fine-tuning over evolving versions)
  - preset: EOPKG-drift-finetune

  # CDLG benchmark scenario (simple1 process tree, EOPKG model)
  - preset: _CDLG-simple1_EOPKG
```

- Pre-check presets availability without running:
  ```bash
  python tools/run_benchmark.py --dry-run
  ```
- Output logs and evaluation metrics will be written to `outputs/cdlg_benchmark/`.

---

### 4. Interactive Experiment UI (Single Run & Deep Inspection)

For full visual inspection of model configurations, graph features, and running individual experiments:

```bash
python main.py experiment-ui
```
*(or `python tools/experiment_ui.py`)*

#### Steps to run in GUI:
1. In the **Presets** dropdown at the top, select the experiment preset (e.g., `EOPKG-drift-finetune` or `_CDLG-simple1_EOPKG`).
2. Click **Apply Preset** to populate all experiment forms and hyperparameters.
3. Review the parameters (data paths, model architecture, GNN layers, drift window).
4. Click **Run Pipeline** at the bottom to execute training and drift evaluation. Live execution logs and progress metrics will be displayed in the console panel.

> **Tip for evaluating different seeds:** To test another experimental seed (e.g., `13`, `32`, `42`, `55`, `67`, `78`, `99`), update the **Seed** field in the form, reflect the seed in the experiment/preset name if desired, and select the corresponding seed in the **Checkpoint path** field (e.g., switching from `_EOPKG-78_...pth` to `_EOPKG-42_...pth`).

---

### 5. Experiment Tracking & Visualization (MLflow UI)

All training runs, hyperparameters, loss curves, and prequential concept drift evaluation metrics are automatically logged locally into `./mlruns`.

To visually explore, compare runs, and inspect detailed metric charts in your browser:

```bash
mlflow ui
```

Then open **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)** in your web browser. 

*(MLflow reads directly from the local `./mlruns` directory — no external database, server, or cloud connection is required).*

---

## 📂 Repository Structure

- `configs/` — Experiment specifications, model hyperparameters, and UI preset definitions.
  - `configs/ui/cdlg_benchmark_plan.yaml` — Active execution plan for `run_benchmark.py`.
  - `configs/ui/experiment_ui_presets.json` — Complete preset configurations for UI and benchmark runs.
- `data/` — Event logs and process topology knowledge graphs.
  - `data/knowledge_graph/` — Pre-built offline process topology graphs (v1–v5 structures per case).
  - `data/loan/` — Loan application process dataset (`.zip` / `.xes`).
  - `data/cdlg/` — Multi-version synthetic benchmark datasets.
- `checkpoints/` — Pre-trained base model weights (`_best.pth`, `_finetune.pth`).
- `src/` — Clean Architecture implementation (`domain`, `application`, `adapters`, `infrastructure`).
- `tools/` — Orchestration and UI tools (`run_benchmark.py`, `experiment_ui.py`).
- `main.py` — Unified CLI entrypoint.

---

## 📚 Scientific Documentation

For deep technical insights and methodological rigor, detailed documentation is available in `docs/`:

- [**Methodology & Anti-Leakage Guardrails**](file:///docs/METHODOLOGY.md) — Research hypotheses, temporal split rigor (`train_cut`), strict statistical snapshot policies (`strict_asof`), multi-seed determinism, and prequential drift evaluation protocols.
- [**System Architecture & Design**](file:///docs/ARCHITECTURE.md) — Hexagonal Architecture layer boundaries, decoupled `IKnowledgeGraphPort` abstractions, and topology-conditioned candidate scoring mechanics (`EOPKGTopologyConditioned`).
- [**Benchmark Suites & Dataset Provenance**](file:///docs/BENCHMARKS.md) — Evolutionary characteristics of the 5-version Loan simulation and the 15 CDLG structural drift benchmark configurations.

