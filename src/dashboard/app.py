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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Speedup", "Throughput", "GPU Utilization", "VaR Results", "Forecaster"])

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
            import pandas as pd
            ldf = pd.read_csv(loss_csv)
            fig = px.line(ldf, y=["train_loss", "val_loss"], labels={"index": "Epoch", "value": "MSE Loss"}, title="Training curves")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Predicted next-day returns (annualized %)**")
        try:
            from src.data.fetch import fetch_prices, compute_log_returns
            from src.models.forecaster import load_model, predict_next_day
            prices = fetch_prices()
            returns = compute_log_returns(prices)[ckpt["asset_names"]].dropna().to_numpy(dtype="float32")
            model, _ = load_model(str(ckpt_path), device="cpu")
            pred = predict_next_day(model, returns[-ckpt["window"]:], device="cpu")
            import pandas as pd
            pred_df = pd.DataFrame({"Asset": ckpt["asset_names"], "Predicted daily return": pred, "Annualized (%)": pred * 252 * 100})
            st.dataframe(pred_df.sort_values("Predicted daily return", ascending=False).style.format({"Predicted daily return": "{:.5f}", "Annualized (%)": "{:.2f}"}), use_container_width=True)
        except Exception as e:
            st.warning(f"Could not run inference: {e}")
