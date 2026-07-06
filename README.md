# GPU-Accelerated Portfolio Risk Engine

A containerized Monte Carlo portfolio risk engine benchmarking PyTorch/CUDA against NumPy/CPU across problem scales. Built as an NVIDIA internship portfolio project.

## What it does

- **Monte Carlo VaR/CVaR** — simulates N GBM price paths for a basket of assets (CPU: NumPy, GPU: PyTorch CUDA tensor ops)
- **Markowitz optimization** — GPU-accelerated covariance computation + CVXPY efficient frontier solver
- **Benchmark sweep** — sweeps across `(n_assets, n_paths, device)` configs and logs throughput, GPU memory, and wall time
- **Interactive dashboard** — Streamlit app showing GPU speedup curves, throughput charts, and portfolio visualizations

## Quick start

```bash
# Install deps (local, no GPU required for CPU baseline)
pip install -r requirements.txt

# Fetch price data and run a quick smoke test
python -m src.data.fetch
python -m src.bench.run_sweep --quick --device cpu

# Run the dashboard
streamlit run src/dashboard/app.py
```

## With a GPU (cloud instance)

```bash
# Confirm CUDA is available
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Full benchmark sweep
python -m src.bench.run_sweep --device both
```

## Docker (requires NVIDIA Container Toolkit)

```bash
# Build
docker build -f docker/Dockerfile -t gpu-portfolio-bench .

# Run benchmark
docker run --gpus all -v $(pwd)/results:/app/results gpu-portfolio-bench --device both

# Or via compose
docker compose -f docker/docker-compose.yml up
```

## Project structure

```
gpu-portfolio-bench/
├── src/
│   ├── data/           # yfinance fetch + Parquet cache
│   ├── models/         # var_cpu.py, var_gpu.py, portfolio_opt.py
│   ├── bench/          # benchmark sweep harness
│   └── dashboard/      # Streamlit app
├── docker/             # Dockerfile + docker-compose.yml
├── results/            # benchmark_sweep.csv (generated)
├── tests/              # correctness tests (CPU/GPU VaR agreement)
└── requirements.txt
```

## Results

> Fill in after running the full sweep on a GPU instance.

| Config | CPU (paths/sec) | GPU (paths/sec) | Speedup |
|--------|----------------|----------------|---------|
| 10 assets, 1M paths | — | — | — |
| 50 assets, 1M paths | — | — | — |
| 100 assets, 10M paths | — | — | — |

## Tests

```bash
pytest tests/ -v
```

## Phase roadmap

| Phase | Status |
|-------|--------|
| 0 — Setup & environment | ✅ structure ready |
| 1a/1b — Data pipeline + Monte Carlo VaR | ✅ CPU + GPU |
| 1c — Markowitz optimization | ✅ |
| 1d — LSTM forecasting model | ✅ |
| 2 — Containerization | ✅ Dockerfile + compose |
| 3 — Benchmark sweep at scale | ✅ harness ready |
| 4 — Dashboard | ✅ Streamlit |
| 5 — Stretch (RAPIDS, CUDA kernel, Slurm) | 🔜 |
