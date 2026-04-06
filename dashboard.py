"""PhaseBreak Hybrid 2.0 — Streamlit Dashboard.

Usage:
    streamlit run dashboard.py
"""

import json
from pathlib import Path

import streamlit as st

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(page_title="PhaseBreak Hybrid 2.0", page_icon="📊", layout="wide")
st.title("📊 PhaseBreak Hybrid 2.0 — Live Monitoring")


# ─── Tab 1: Current Signals ─────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔴 Signals", "🗺️ Sectors", "📜 History", "📋 Eval"])

with tab1:
    st.subheader("Current Market Signals")

    # Load latest scan
    scan_files = sorted(DATA_DIR.glob("live_scan_*.json"), reverse=True)
    if scan_files:
        with open(scan_files[0]) as f:
            scan = json.load(f)

        st.caption(f"Scan date: {scan.get('scan_date', 'unknown')}")

        bubbles = scan.get("summary", {}).get("bubble", [])
        possibles = scan.get("summary", {}).get("possible", [])

        if bubbles:
            st.error(f"🔴 BUBBLE: {', '.join(bubbles)}")
        if possibles:
            st.warning(f"🟡 POSSIBLE: {', '.join(possibles)}")
        if not bubbles and not possibles:
            st.success("✅ No bubble signals detected")

        # Results table
        results = scan.get("results", [])
        if results:
            rows = []
            for r in results:
                rows.append(
                    {
                        "Ticker": r.get("ticker", ""),
                        "Name": r.get("name", ""),
                        "Price": r.get("price", 0),
                        "Verdict": r.get("verdict", ""),
                        "Quality": r.get("quality_score", 0),
                        "R²": r.get("r_squared", 0),
                        "HMM": r.get("hmm_regime", ""),
                        "Hurst": r.get("hurst", ""),
                        "tc": r.get("tc_date_str", ""),
                    }
                )
            st.dataframe(rows, use_container_width=True)
    else:
        st.info("No scan results found. Run `python monitor_cron.py --dry-run` first.")


with tab2:
    st.subheader("Sector Heatmap")

    heatmap_files = sorted(DATA_DIR.glob("heatmap*.json"), reverse=True)
    if heatmap_files:
        with open(heatmap_files[0]) as f:
            heatmap = json.load(f)

        for sector in heatmap.get("sectors", []):
            heat = sector.get("heat_score", 0)
            name = sector.get("name", "")
            alert = " 🚨" if sector.get("is_alert") else ""
            bubble = sector.get("bubble", 0)
            possible = sector.get("possible", 0)
            total = sector.get("total", 0)

            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.progress(min(heat, 1.0), text=f"{name}{alert}")
            with col2:
                st.metric("Heat", f"{heat:.0%}")
            with col3:
                st.caption(f"B:{bubble} P:{possible}/{total}")
    else:
        st.info(
            "No sector scan results. Run `python -m src.pipeline.sector_scan --output data/heatmap.json`"
        )


with tab3:
    st.subheader("Signal History")

    history_path = DATA_DIR / "signal_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

        if history:
            ticker = st.selectbox("Select ticker", sorted(history.keys()))
            if ticker:
                entries = history[ticker]
                st.line_chart(
                    data={"quality": [e.get("quality", 0) for e in entries]},
                )
                st.dataframe(entries[-10:], use_container_width=True)
        else:
            st.info("Signal history is empty. Run monitor a few times first.")
    else:
        st.info("No signal history. Run `python monitor_cron.py --dry-run` first.")

    # Monitor log
    log_path = DATA_DIR / "monitor_log.json"
    if log_path.exists():
        with open(log_path) as f:
            logs = json.load(f)
        if logs:
            st.subheader("Monitor Run Log")
            st.dataframe(logs[-20:], use_container_width=True)


with tab4:
    st.subheader("Evaluation Results")

    # Crash eval
    crash_path = DATA_DIR / "crash_eval_results.json"
    if crash_path.exists():
        with open(crash_path) as f:
            crash = json.load(f)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Crash Precision", f"{crash.get('crash_precision', 0):.0%}")
        col2.metric("Crash Recall", f"{crash.get('crash_recall', 0):.0%}")
        col3.metric("F1", f"{crash.get('f1', 0):.0%}")
        col4.metric("Accuracy", f"{crash.get('accuracy', 0):.0%}")

        results = crash.get("results", [])
        if results:
            st.dataframe(results, use_container_width=True)
    else:
        st.info("No crash eval results. Run `python -m src.contract.crash_eval`")

    # Blind eval
    for eval_name, eval_file in [
        ("LPPLS+RAG Blind", "eval_results_rag_blind.json"),
        ("LPPLS-only Blind", "eval_results_blind_v3.json"),
    ]:
        eval_path = DATA_DIR / eval_file
        if eval_path.exists():
            with open(eval_path) as f:
                ev = json.load(f)
            st.metric(f"{eval_name} Accuracy", f"{ev.get('accuracy', 0):.0%}")


# ─── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("PhaseBreak Hybrid 2.0")
    st.caption("LPPLS + HMM + Hurst + RAG")
    st.divider()
    st.markdown("**Pipeline:**")
    st.code("yfinance → HMM → LPPLS → Hurst → RAG → Verdict → Persistence → Alert")
    st.divider()
    st.markdown("**Commands:**")
    st.code(
        "python monitor_cron.py --dry-run\npython -m src.contract.crash_eval\nstreamlit run dashboard.py"
    )
