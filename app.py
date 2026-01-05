# app.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from pairs import (
    load_nasdaq_trader_universe,
    fetch_prices,
    run_pairs_backtest_capital_normalized,
    sharpe_like,
)

# -------------------------
# Cache wrappers (Streamlit)
# -------------------------
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def cached_universe() -> pd.DataFrame:
    return load_nasdaq_trader_universe()

@st.cache_data(show_spinner=False)
def cached_prices(tx: str, ty: str, s: str, e: str) -> pd.DataFrame:
    return fetch_prices(tx, ty, s, e)

@st.cache_data(show_spinner=False)
def cached_backtest_capital(px: pd.DataFrame, lookback: int, entry_z: float, exit_z: float,
                            p_threshold: float, notional_per_leg: float, commission_per_share: float,
                            min_commission: float, slippage_bps: float) -> pd.DataFrame:
    return run_pairs_backtest_capital_normalized(
        px,
        lookback=lookback,
        entry_z=entry_z,
        exit_z=exit_z,
        p_threshold=p_threshold,
        notional_per_leg=notional_per_leg,
        commission_per_share=commission_per_share,
        min_commission=min_commission,
        slippage_bps=slippage_bps,
    )

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Pairs Trading Tester", page_icon="📈", layout="wide")

st.title("📈 Pairs Trading Tester")
st.caption("Select two tickers, run a rolling cointegration + z-score mean-reversion backtest.")

with st.sidebar:
    st.header("Universe (Nasdaq Trader)")

    universe_df = cached_universe()

    exchanges = ["All"] + sorted(universe_df["exchange"].dropna().unique().tolist())
    exchange_choice = st.selectbox("Exchange", exchanges, index=0)

    security_type = st.radio("Security type", ["All", "Stocks only", "ETFs only"], horizontal=True)

    df_base = universe_df.copy()
    if exchange_choice != "All":
        df_base = df_base[df_base["exchange"] == exchange_choice]

    if security_type == "Stocks only":
        df_base = df_base[df_base["etf"] != "Y"]
    elif security_type == "ETFs only":
        df_base = df_base[df_base["etf"] == "Y"]

    st.divider()
    st.subheader("Pick your pair")

    colA, colB = st.columns(2)

    with colA:
        qx = st.text_input("Search X (symbol or name)", value="", key="search_x")
        df_x = df_base
        if qx.strip():
            s = qx.strip().lower()
            df_x = df_x[
                df_x["symbol"].str.lower().str.contains(s, na=False)
                | df_x["name"].str.lower().str.contains(s, na=False)
            ]
        if len(df_x) > 3000 and not qx.strip():
            df_x = df_x.head(3000)

        options_x = (df_x["symbol"] + " — " + df_x["name"]).tolist()
        map_x = dict(zip(options_x, df_x["symbol"]))
        opt_x = st.selectbox("Leg X", options_x, index=0 if options_x else None, key="leg_x")
        ticker_x = map_x.get(opt_x, "")

    with colB:
        qy = st.text_input("Search Y (symbol or name)", value="", key="search_y")
        df_y = df_base
        if qy.strip():
            s = qy.strip().lower()
            df_y = df_y[
                df_y["symbol"].str.lower().str.contains(s, na=False)
                | df_y["name"].str.lower().str.contains(s, na=False)
            ]
        if len(df_y) > 3000 and not qy.strip():
            df_y = df_y.head(3000)

        options_y = (df_y["symbol"] + " — " + df_y["name"]).tolist()
        map_y = dict(zip(options_y, df_y["symbol"]))
        opt_y = st.selectbox("Leg Y", options_y, index=1 if len(options_y) > 1 else 0, key="leg_y")
        ticker_y = map_y.get(opt_y, "")

    if ticker_x and ticker_y and ticker_x == ticker_y:
        st.warning("Pick two different tickers for X and Y.")

    st.divider()

    start = st.date_input("Start date", value=pd.to_datetime("2020-01-01"), key="start_date")
    end = st.date_input("End date", value=pd.to_datetime("2025-12-31"), key="end_date")

    st.divider()
    st.subheader("Parameters")
    lookback = st.slider("Lookback window (days)", 30, 500, 120, step=5)
    entry_z = st.slider("Entry z-score", 0.5, 5.0, 2.0, step=0.1)
    exit_z = st.slider("Exit z-score", 0.1, 2.0, 0.5, step=0.1)
    p_threshold = st.slider("Cointegration p-threshold", 0.001, 0.2, 0.05, step=0.001)

    st.divider()
    st.subheader("Capital & Costs")
    notional_per_leg = st.number_input("Notional per leg ($)", min_value=1000.0, value=10000.0, step=1000.0)
    commission_per_share = st.number_input("Commission per share ($)", min_value=0.0, value=0.005, step=0.001)
    min_commission = st.number_input("Min commission per rebalance ($)", min_value=0.0, value=1.0, step=0.5)
    slippage_bps = st.number_input("Slippage/spread (bps of $ turnover)", min_value=0.0, value=1.0, step=0.5)

    st.divider()
    run = st.button("Run Backtest", type="primary", use_container_width=True)

if run:
    try:
        if not ticker_x or not ticker_y:
            st.error("Please choose both tickers.")
            st.stop()
        if ticker_x == ticker_y:
            st.error("Leg X and Leg Y must be different.")
            st.stop()
        if start >= end:
            st.error("Start date must be before end date.")
            st.stop()

        with st.spinner("Downloading data..."):
            px = cached_prices(ticker_x, ticker_y, start.isoformat(), end.isoformat())

        with st.spinner("Running backtest..."):
            res = cached_backtest_capital(
                px,
                lookback,
                entry_z,
                exit_z,
                p_threshold,
                notional_per_leg,
                commission_per_share,
                min_commission,
                slippage_bps,
            )

        traded_days = int((res["sh_y"].abs() > 1e-12).sum())
        pvals = res["pvalue"].dropna()
        coint_rate = float((pvals < p_threshold).mean()) if len(pvals) else float("nan")
        sh = sharpe_like(res["net_ret"])

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Final Net PnL ($)", f"{res['cum_net_pnl'].iloc[-1]:,.2f}")
        c2.metric("Cumulative Return", f"{100 * res['cum_net_ret'].iloc[-1]:.2f}%")
        c3.metric("Tradable fraction (p<th)", "—" if np.isnan(coint_rate) else f"{coint_rate:.2%}")
        c4.metric("Days in position", f"{traded_days}/{len(res)}")
        c5.metric("Sharpe (net returns)", "—" if np.isnan(sh) else f"{sh:.2f}")

        left, right = st.columns([1.2, 1])

        with left:
            st.subheader("Equity Curve (Net $ PnL)")
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=res.index, y=res["cum_net_pnl"], mode="lines", name="Cumulative Net PnL"))
            fig_eq.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_eq, use_container_width=True)

            st.subheader("Z-Score (Spread)")
            fig_z = go.Figure()
            fig_z.add_trace(go.Scatter(x=res.index, y=res["z"], mode="lines", name="z"))
            fig_z.add_hline(y=entry_z, line_dash="dash")
            fig_z.add_hline(y=-entry_z, line_dash="dash")
            fig_z.add_hline(y=exit_z, line_dash="dot")
            fig_z.add_hline(y=-exit_z, line_dash="dot")
            fig_z.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_z, use_container_width=True)

        with right:
            st.subheader("Recent rows")
            st.dataframe(res.tail(30), use_container_width=True)

            st.download_button(
                label="Download results CSV",
                data=res.to_csv().encode("utf-8"),
                file_name=f"pairs_{ticker_x}_{ticker_y}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    except Exception as e:
        st.error(str(e))
else:
    st.info("Set your pair + parameters on the left, then click **Run Backtest**.")
