# Multi-stage build to keep final image lean
# Stage 1: build deps
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime AS base

WORKDIR /app

# System packages needed for cvxpy solvers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: runtime
FROM base AS runtime
COPY . .

# Default: run the benchmark sweep
# Override CMD to run dashboard: streamlit run src/dashboard/app.py
ENTRYPOINT ["python", "-m", "src.bench.run_sweep"]
CMD ["--device", "both"]
