# GPU-Accelerated Portfolio Risk Engine

A containerized Monte Carlo portfolio risk engine benchmarking three compute backends — NumPy CPU, PyTorch CUDA, and a hand-written CuPy raw CUDA kernel — across problem scales from 10k to 10M simulated paths.

Built as an internship portfolio project targeting NVIDIA's Financial Services GPU Benchmarking team.

---

## What it does

| Component | Description |
|-----------|-------------|
| `src/data/fetch.py` | yfinance → Parquet cache; incremental refresh on re-run |
| `src/models/var_cpu.py` | NumPy GBM Monte Carlo VaR/CVaR — CPU baseline |
| `src/models/var_gpu.py` | PyTorch CUDA — single `torch.randn(N,T,A,device='cuda')` replaces the path loop |
| `src/models/var_cuda_kernel.py` | CuPy raw CUDA kernel — one thread per path, `curand` RNG, hand-written Cholesky application |
| `src/models/portfolio_opt.py` | Markowitz efficient frontier; GPU covariance via PyTorch |
| `src/models/forecaster.py` | LSTM: train → checkpoint → infer → feed predicted returns into optimizer |
| `src/bench/run_sweep.py` | Sweep harness: `(n_assets, n_paths, device)` → `results/benchmark_sweep.csv` |
| `src/dashboard/app.py` | Streamlit: speedup curves, throughput, GPU memory, VaR table, forecaster tab |

---

## Quick start

```bash
# Set up environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Optional GPU extras (match your CUDA version)
pip install cupy-cuda12x       # raw CUDA kernel support
pip install cudf-cu12          # RAPIDS GPU DataFrames
pip install cuopt-cu12         # NVIDIA cuOPT solver

# Confirm GPU (on a cloud instance)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Fetch data + smoke test (CPU only — works without a GPU)
python -m src.data.fetch
python -m src.bench.run_sweep --quick --device cpu

# Full sweep (CPU + CUDA + CuPy kernel)
python -m src.bench.run_sweep --device all

# Train the LSTM forecaster
python -m src.models.forecaster --epochs 50 --assets 20

# Launch dashboard
streamlit run src/dashboard/app.py
```

---

## Docker

```bash
# Build
docker build -f docker/Dockerfile -t gpu-portfolio-bench .

# Benchmark (requires NVIDIA Container Toolkit)
docker run --gpus all -v $(pwd)/results:/app/results gpu-portfolio-bench --device both

# Full stack: benchmark + dashboard
docker compose -f docker/docker-compose.yml up
```

---

## Slurm (HPC clusters)

```bash
# Single job — full sweep
sbatch slurm/submit_sweep.sh

# Array job — one job per asset-count bucket, runs in parallel
sbatch slurm/array_sweep.sh
```

---

## Kubernetes

```bash
# Deploy benchmark job
kubectl apply -f k8s/bench-job.yaml

# Deploy dashboard (reads results from shared PVC)
kubectl apply -f k8s/dashboard-deployment.yaml

# Monitor
kubectl logs -f job/gpu-portfolio-bench
```

---

## Project structure

```
gpu-portfolio-bench/
├── src/
│   ├── data/               # yfinance fetch + Parquet cache
│   ├── models/             # var_cpu, var_gpu, var_cuda_kernel, portfolio_opt, forecaster
│   ├── bench/              # benchmark sweep harness
│   └── dashboard/          # Streamlit app
├── docker/                 # Dockerfile (pytorch/pytorch CUDA base) + compose
├── slurm/                  # sbatch scripts (single + array job)
├── k8s/                    # Kubernetes Job + dashboard Deployment
├── notebooks/              # 01_demo.ipynb — end-to-end walkthrough
├── results/                # benchmark_sweep.csv (generated), checkpoints/
├── tests/                  # VaR correctness + forecaster shape/training tests
├── pyproject.toml
└── requirements.txt
```

---

## Tests

```bash
pytest tests/ -v
# GPU tests auto-skip if no CUDA device is present
```

---

## Results

CPU numbers measured on Apple M-series (Python 3.14, NumPy GBM). GPU columns fill in after running on a cloud instance (T4/A100).

| Config | CPU (paths/s) | PyTorch CUDA (paths/s) | CuPy kernel (paths/s) | GPU speedup |
|--------|:-------------:|:---------------------:|:--------------------:|:-----------:|
| 10 assets, 10k paths | 12.1M | — | — | — |
| 10 assets, 1M paths | 9.7M | — | — | — |
| 10 assets, 10M paths | 11.2M | — | — | — |
| 50 assets, 10k paths | 3.8M | — | — | — |
| 50 assets, 1M paths | 4.5M | — | — | — |
| 50 assets, 10M paths | 1.1M | — | — | — |

---

## Phase roadmap

| Phase | Status |
|-------|--------|
| 0 — Setup & environment | ✅ |
| 1a/1b — Data pipeline + Monte Carlo VaR (CPU + GPU) | ✅ |
| 1c — Markowitz efficient frontier | ✅ |
| 1d — LSTM forecasting model | ✅ |
| 2 — Containerization | ✅ |
| 3 — Benchmark sweep at scale | ✅ |
| 4 — Streamlit dashboard | ✅ |
| 5 — Custom CUDA kernel (CuPy) | ✅ |
| 5 — Slurm + Kubernetes orchestration | ✅ |
| 5 — RAPIDS cuDF ETL | ✅ |
| 5 — NVIDIA cuOPT portfolio solver | ✅ |
