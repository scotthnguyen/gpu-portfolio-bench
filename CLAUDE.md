# gpu-portfolio-bench

GPU-accelerated Monte Carlo portfolio risk engine. NVIDIA internship portfolio project.

## What this is

Benchmarks three compute backends (NumPy CPU, PyTorch CUDA, CuPy raw kernel) for Monte Carlo Value-at-Risk simulation across a basket of S&P 500 assets. Also includes Markowitz portfolio optimization, an LSTM return forecaster, a Streamlit dashboard, and Slurm/Kubernetes orchestration scripts.

## Project structure

```
src/data/fetch.py              yfinance → Parquet cache (incremental)
src/models/var_cpu.py          NumPy GBM Monte Carlo VaR/CVaR
src/models/var_gpu.py          PyTorch CUDA equivalent
src/models/var_cuda_kernel.py  CuPy raw CUDA kernel (one thread per path)
src/models/portfolio_opt.py    Markowitz efficient frontier, GPU covariance
src/models/forecaster.py       LSTM: train → checkpoint → infer
src/models/forecast_optimizer.py  LSTM predictions → max-Sharpe weights
src/bench/run_sweep.py         Sweep harness → results/benchmark_sweep.csv
src/dashboard/app.py           Streamlit dashboard (5 tabs)
docker/                        Dockerfile + docker-compose
slurm/                         sbatch scripts (single + array job)
k8s/                           Kubernetes Job + dashboard Deployment
notebooks/01_demo.ipynb        End-to-end Colab walkthrough
tests/                         pytest suite (CPU-only, GPU tests auto-skip)
```

## Running things

```bash
# Smoke test (no GPU needed)
python -m src.bench.run_sweep --quick --device cpu

# Full sweep (GPU instance)
python -m src.bench.run_sweep --device all

# Train LSTM
python -m src.models.forecaster --epochs 50 --assets 20

# Dashboard
streamlit run src/dashboard/app.py

# Tests
pytest tests/ -v
```

## Key design decisions

- PyTorch is the primary GPU library; CuPy used only for the raw kernel demo
- CPU baseline always runs alongside GPU — needed for correctness validation and speedup ratio
- Benchmark results go to `results/benchmark_sweep.csv`; dashboard reads from there
- GPU tests call `pytest.skip()` when `torch.cuda.is_available()` is False — CI runs CPU-only

## Commit style

Short subject lines. No AI tool mentions.

## All phases complete

Every phase from the project plan is implemented. Next steps are running the actual
benchmark sweep on a GPU instance and filling in the results table in README.md.
