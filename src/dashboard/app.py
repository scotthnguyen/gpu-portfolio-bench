"""
Streamlit dashboard: GPU vs CPU speedup curves, throughput, efficient frontier.
Run: streamlit run src/dashboard/app.py
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

RESULTS_PATH = Path(__file__).parent.parent.parent / "results" / "benchmark_sweep.csv"

st.set_page_config(page_title="GPU Portfolio Bench", layout="wide")
st.title("GPU-Accelerated Monte Carlo Portfolio Risk Engine")
st.caption("NVIDIA internship project — benchmarking PyTorch CUDA vs NumPy CPU across problem scales")


@st.cache_data
def load_results() -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(RESULTS_PATH)


df = load_results()

if df.empty:
    st.warning("No benchmark results yet. Run `python -m src.bench.run_sweep --quick` first.")
    st.stop()

# ── Filters ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    asset_opts = sorted(df["n_assets"].unique())
    selected_assets = st.multiselect("Number of assets", asset_opts, default=asset_opts)
    df = df[df["n_assets"].isin(selected_assets)]

# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Speedup", "Throughput", "GPU Utilization", "VaR Results", "Efficient Frontier", "Forecaster", "RAPIDS / cuOPT"])

with tab1:
    st.subheader("GPU Speedup vs CPU (by problem size)")
    cpu = df[df["device"] == "cpu"][["n_assets", "n_paths", "elapsed_sec"]].rename(columns={"elapsed_sec": "cpu_sec"})
    gpu = df[df["device"] == "cuda"][["n_assets", "n_paths", "elapsed_sec"]].rename(columns={"elapsed_sec": "gpu_sec"})
    merged = pd.merge(cpu, gpu, on=["n_assets", "n_paths"])
    if merged.empty:
        st.info("Need both CPU and GPU runs to compute speedup.")
    else:
        merged["speedup"] = merged["cpu_sec"] / merged["gpu_sec"]
        fig = px.line(
            merged, x="n_paths", y="speedup", color="n_assets",
            log_x=True, markers=True,
            labels={"n_paths": "Monte Carlo paths", "speedup": "GPU speedup (×)", "n_assets": "Assets"},
            title="GPU Speedup Factor — where CUDA beats NumPy",
        )
        fig.add_hline(y=1, line_dash="dash", line_color="gray", annotation_text="break-even")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Speedup < 1 at small scales = CUDA kernel launch overhead dominates. Crossover visible as paths scale up.")

with tab2:
    st.subheader("Throughput: Paths/second vs Problem Size")
    fig = px.line(
        df, x="n_paths", y="throughput_paths_per_sec", color="device",
        facet_col="n_assets", log_x=True, log_y=True, markers=True,
        labels={"n_paths": "Paths", "throughput_paths_per_sec": "Paths/sec"},
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("GPU Memory Usage (MB)")
    gpu_df = df[df["device"] == "cuda"]
    if gpu_df.empty:
        st.info("No CUDA runs in results.")
    else:
        fig = px.scatter(
            gpu_df, x="n_paths", y="gpu_mem_mb", color="n_assets",
            size="n_assets", log_x=True,
            labels={"gpu_mem_mb": "GPU memory (MB)", "n_paths": "Paths"},
        )
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("VaR / CVaR Estimates")
    st.dataframe(
        df[["device", "n_assets", "n_paths", "VaR", "CVaR", "elapsed_sec", "throughput_paths_per_sec"]]
        .sort_values(["n_assets", "n_paths", "device"])
        .style.format({"VaR": "{:.4f}", "CVaR": "{:.4f}", "elapsed_sec": "{:.3f}", "throughput_paths_per_sec": "{:,.0f}"}),
        use_container_width=True,
    )

with tab5:
    st.subheader("Markowitz Efficient Frontier")
    st.caption("GPU-accelerated covariance (PyTorch) + CVXPY/CLARABEL for QP — traces the risk/return tradeoff across 50 target returns.")

    n_assets_ef = st.slider("Number of assets", min_value=5, max_value=50, value=20, step=5)

    @st.cache_data(show_spinner="Building efficient frontier…")
    def compute_frontier(n: int) -> dict:
        import numpy as np

        from src.data.fetch import compute_log_returns, fetch_prices
        from src.models.portfolio_opt import build_efficient_frontier, max_sharpe_weights

        prices = fetch_prices()
        returns = compute_log_returns(prices).dropna(axis=1)
        asset_names = list(returns.columns[:n])
        R = returns[asset_names].to_numpy(dtype=np.float64)
        ef = build_efficient_frontier(R, asset_names, n_points=50, use_gpu=False)
        mu_ann = R.mean(axis=0) * 252
        cov_ann = np.cov(R.T) * 252
        ms_w = max_sharpe_weights(mu_ann, cov_ann)
        ms_ret = float(ms_w @ mu_ann)
        ms_vol = float(np.sqrt(ms_w @ cov_ann @ ms_w))
        ms_sr = (ms_ret - 0.05) / ms_vol
        return {
            "vols": ef.volatilities,
            "rets": ef.expected_returns[:len(ef.volatilities)],
            "sharpes": ef.sharpe_ratios,
            "ms_ret": ms_ret,
            "ms_vol": ms_vol,
            "ms_sr": ms_sr,
            "ms_weights": ms_w.tolist(),
            "asset_names": asset_names,
        }

    try:
        import numpy as np
        import pandas as pd

        data = compute_frontier(n_assets_ef)
        ef_df = pd.DataFrame({
            "Volatility": data["vols"],
            "Return": data["rets"],
            "Sharpe": data["sharpes"],
        })
        fig = px.scatter(
            ef_df, x="Volatility", y="Return", color="Sharpe",
            color_continuous_scale="Viridis",
            labels={"Volatility": "Annualized Volatility", "Return": "Annualized Return"},
            title=f"Efficient Frontier — {n_assets_ef} assets (Markowitz, CVXPY/CLARABEL)",
        )
        fig.add_scatter(
            x=[data["ms_vol"]], y=[data["ms_ret"]],
            mode="markers", marker=dict(symbol="star", size=16, color="red"),
            name=f"Max Sharpe ({data['ms_sr']:.2f}×)",
        )
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Max Sharpe", f"{data['ms_sr']:.3f}")
        col_b.metric("Expected Return", f"{data['ms_ret']*100:.2f}%")
        col_c.metric("Volatility", f"{data['ms_vol']*100:.2f}%")

        top_n = 10
        top_idx = np.argsort(np.array(data["ms_weights"]))[::-1][:top_n]
        wdf = pd.DataFrame({
            "Asset": [data["asset_names"][i] for i in top_idx],
            "Weight (%)": [data["ms_weights"][i] * 100 for i in top_idx],
        })
        st.markdown(f"**Top {top_n} holdings (max-Sharpe portfolio)**")
        st.dataframe(wdf.style.format({"Weight (%)": "{:.2f}"}), use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Could not build frontier: {e}")

with tab6:
    st.subheader("LSTM Return Forecaster")
    ckpt_path = Path(__file__).parent.parent.parent / "results" / "checkpoints" / "lstm_best.pt"

    if not ckpt_path.exists():
        st.info("No trained model yet. Run:\n```\npython -m src.models.forecaster --epochs 50 --assets 20\n```")
    else:
        import torch
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        st.success(f"Checkpoint loaded — epoch {ckpt['epoch']}, val loss {ckpt['val_loss']:.6f}")
        st.write(f"**Assets:** {len(ckpt['asset_names'])}   **Window:** {ckpt['window']} days   **Hidden:** {ckpt['hidden']}")

        loss_csv = Path(__file__).parent.parent.parent / "results" / "train_losses.csv"
        if loss_csv.exists():
            ldf = pd.read_csv(loss_csv)
            fig = px.line(ldf, x="epoch", y=["train_loss", "val_loss"], labels={"value": "MSE Loss", "variable": ""}, title="Training curves")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Predicted next-day returns (annualized %)**")
        try:
            from src.data.fetch import compute_log_returns, fetch_prices
            from src.models.forecaster import load_model, predict_next_day
            prices = fetch_prices()
            returns = compute_log_returns(prices)[ckpt["asset_names"]].dropna().to_numpy(dtype="float32")
            model, _ = load_model(str(ckpt_path), device="cpu")
            pred = predict_next_day(model, returns[-ckpt["window"]:], device="cpu")
            pred_df = pd.DataFrame({"Asset": ckpt["asset_names"], "Predicted daily return": pred, "Annualized (%)": pred * 252 * 100})
            st.dataframe(pred_df.sort_values("Predicted daily return", ascending=False).style.format({"Predicted daily return": "{:.5f}", "Annualized (%)": "{:.2f}"}), use_container_width=True)
        except Exception as e:
            st.warning(f"Could not run inference: {e}")

with tab7:
    st.subheader("RAPIDS cuDF & cuOPT")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### cuDF ETL Benchmark")
        try:
            from src.data.fetch_rapids import CUDF_AVAILABLE, benchmark_etl
            if CUDF_AVAILABLE:
                st.success("cuDF detected — GPU ETL active")
                with st.spinner("Benchmarking ETL (3 runs each)..."):
                    timings = benchmark_etl(n_runs=3)
                st.metric("pandas", f"{timings['pandas_etl_sec']*1000:.1f} ms")
                st.metric("cuDF", f"{timings['cudf_etl_sec']*1000:.1f} ms", delta=f"{timings['speedup']:.1f}× faster")
            else:
                st.info("cuDF not installed. On a GPU instance: `pip install cudf-cu12`")
                st.caption("cuDF replaces pandas DataFrames with GPU-resident equivalents — same API, GPU throughput.")
        except Exception as e:
            st.warning(str(e))

    with col2:
        st.markdown("#### cuOPT Portfolio Optimizer")
        try:
            from src.models.portfolio_cuopt import CUOPT_AVAILABLE, max_sharpe_cuopt
            if CUOPT_AVAILABLE:
                st.success("cuOPT detected")
            else:
                st.info("cuOPT not installed. On a GPU instance: `pip install cuopt-cu12`")
                st.caption("cuOPT is NVIDIA's GPU-accelerated optimization solver. Falls back to CVXPY/CLARABEL when unavailable.")

            import numpy as np

            from src.data.fetch import compute_log_returns, fetch_prices
            prices = fetch_prices()
            returns = compute_log_returns(prices).dropna(axis=1)
            R = returns.iloc[:, :20].to_numpy(dtype=np.float64)
            mu = R.mean(axis=0) * 252
            cov = np.cov(R.T) * 252
            result = max_sharpe_cuopt(mu, cov)
            st.metric("Solver", result.solver)
            st.metric("Sharpe ratio", f"{result.sharpe:.3f}")
            st.metric("Annualized return", f"{result.expected_return*100:.2f}%")
            st.metric("Annualized vol", f"{result.volatility*100:.2f}%")
            st.metric("Solve time", f"{result.elapsed_sec*1000:.1f} ms")
        except Exception as e:
            st.warning(str(e))
