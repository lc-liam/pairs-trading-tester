# pairs_log.py
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import coint


def load_nasdaq_trader_universe() -> pd.DataFrame:
    """
    Builds a unified US ticker universe from Nasdaq Trader symbol directories:
    - nasdaqlisted.txt (NASDAQ-listed)
    - otherlisted.txt  (NYSE/AMEX/ARCA/BATS/etc listed securities)

    Returns columns:
      symbol, name, exchange, etf, test_issue
    """
    # NASDAQ-listed
    nasdaq_url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    ndq = pd.read_csv(nasdaq_url, sep="|")
    ndq = ndq[ndq["Symbol"].notna()]
    ndq = ndq[~ndq["Symbol"].astype(str).str.contains("File Creation Time", na=False)]
    ndq = ndq.rename(
        columns={
            "Symbol": "symbol",
            "Security Name": "name",
            "Test Issue": "test_issue",
            "ETF": "etf",
        }
    )[["symbol", "name", "test_issue", "etf"]]
    ndq["exchange"] = "NASDAQ"

    # Other-listed (NYSE/AMEX/ARCA/BATS/etc)
    other_url = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
    oth = pd.read_csv(other_url, sep="|")
    oth = oth[oth["ACT Symbol"].notna()]
    oth = oth[~oth["ACT Symbol"].astype(str).str.contains("File Creation Time", na=False)]
    oth = oth.rename(
        columns={
            "ACT Symbol": "symbol",
            "Security Name": "name",
            "Test Issue": "test_issue",
            "ETF": "etf",
            "Exchange": "exchange_code",
        }
    )[["symbol", "name", "exchange_code", "test_issue", "etf"]]

    exch_map = {
        "N": "NYSE",
        "A": "NYSE American",
        "P": "NYSE Arca",
        "Z": "Cboe BZX",
        "V": "IEX",
        "B": "Cboe BYX",
        "Y": "Cboe EDGX",
        "X": "Cboe EDGA",
        "Q": "NASDAQ",
    }
    oth["exchange"] = oth["exchange_code"].map(exch_map).fillna(oth["exchange_code"])
    oth = oth.drop(columns=["exchange_code"])

    # Combine
    uni = pd.concat([ndq, oth], ignore_index=True)

    # Clean
    uni["symbol"] = uni["symbol"].astype(str).str.upper().str.strip()
    uni["name"] = uni["name"].astype(str).str.strip()

    # Filter out test issues
    uni["test_issue"] = uni["test_issue"].fillna("N")
    uni = uni[uni["test_issue"] == "N"].copy()

    # Normalize ETF flag
    uni["etf"] = uni["etf"].fillna("N")

    # Deduplicate symbols
    uni = uni.drop_duplicates(subset=["symbol"], keep="first").reset_index(drop=True)
    return uni


def fetch_prices(ticker_x: str, ticker_y: str, start: str, end: str) -> pd.DataFrame:
    """
    Downloads adjusted close prices for two tickers and returns aligned columns:
      x, y
    indexed by datetime.
    """
    df = yf.download([ticker_x, ticker_y], start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError("No data returned. Check tickers/date range.")

    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Did not receive multi-ticker data. Check tickers.")

    # yfinance output with auto_adjust=True
    if "Close" in df.columns.get_level_values(0):
        px = df["Close"].copy()
    else:
        px = df.xs("Close", axis=1, level=0, drop_level=True)

    px = px.rename(columns={ticker_x: "x", ticker_y: "y"}).dropna()
    if px.shape[0] < 50:
        raise ValueError("Not enough aligned data points.")
    return px


def run_pairs_backtest_capital_normalized(
    px: pd.DataFrame,
    lookback: int,
    entry_z: float,
    exit_z: float,
    p_threshold: float,
    notional_per_leg: float,
    commission_per_share: float,
    min_commission: float,
    slippage_bps: float,
) -> pd.DataFrame:
    """
    Dollar-neutral pairs backtest with:
    - rolling cointegration + hedge ratio
    - z-score entry/exit
    - capital-normalized positions (fixed $ notional per leg)
    - transaction costs (commission + slippage/spread)

    px columns must be: x, y (prices), indexed by datetime.
    """
    df = px.copy()
    df["beta"] = np.nan
    df["pvalue"] = np.nan
    df["spread"] = np.nan
    df["z"] = np.nan

    # Positions in SHARES (capital-normalized)
    df["sh_y"] = 0.0
    df["sh_x"] = 0.0

    # PnL + costs in DOLLARS
    df["gross_pnl"] = 0.0
    df["cost"] = 0.0
    df["net_pnl"] = 0.0
    df["cum_net_pnl"] = 0.0

    # Returns (net) on deployed capital (2 legs)
    df["net_ret"] = 0.0
    df["cum_net_ret"] = 0.0

    lr = LinearRegression()

    # Rolling estimates: beta, pval, z
    for i in range(lookback, len(df)):
        window = df.iloc[i - lookback : i]

        _, pval, _ = coint(window["x"].values, window["y"].values)
        df.iloc[i, df.columns.get_loc("pvalue")] = pval

        x = window["x"].values.reshape(-1, 1)
        y = window["y"].values.reshape(-1, 1)
        lr.fit(x, y)
        beta = float(lr.coef_.ravel()[0])
        df.iloc[i, df.columns.get_loc("beta")] = beta

        spread_series = window["y"] - beta * window["x"]
        spread_now = df["y"].iloc[i] - beta * df["x"].iloc[i]
        z = (spread_now - spread_series.mean()) / (spread_series.std(ddof=0) + 1e-12)

        df.iloc[i, df.columns.get_loc("spread")] = spread_now
        df.iloc[i, df.columns.get_loc("z")] = z

    # Trading + sizing
    in_trade = 0  # +1 long Y/short X ; -1 short Y/long X ; 0 flat
    deployed_capital = 2.0 * float(notional_per_leg)

    for i in range(1, len(df)):
        z = df["z"].iloc[i]
        pval = df["pvalue"].iloc[i]
        beta = df["beta"].iloc[i]

        prev_sh_y = df["sh_y"].iloc[i - 1]
        prev_sh_x = df["sh_x"].iloc[i - 1]

        sh_y = prev_sh_y
        sh_x = prev_sh_x

        if np.isnan(z) or np.isnan(pval) or np.isnan(beta):
            sh_y = 0.0
            sh_x = 0.0
        else:
            tradable = (pval < p_threshold)

            # exit
            if in_trade != 0 and abs(z) <= exit_z:
                in_trade = 0

            # enter
            if in_trade == 0 and tradable:
                if z <= -entry_z:
                    in_trade = +1
                elif z >= entry_z:
                    in_trade = -1

            px_x = float(df["x"].iloc[i])
            px_y = float(df["y"].iloc[i])

            if in_trade == 0:
                sh_y = 0.0
                sh_x = 0.0
            else:
                sign_beta = 1.0 if beta >= 0 else -1.0
                sh_y = in_trade * (notional_per_leg / px_y)
                sh_x = (-in_trade) * sign_beta * (notional_per_leg / px_x)

        # Costs on trades
        d_sh_y = sh_y - prev_sh_y
        d_sh_x = sh_x - prev_sh_x

        turnover = abs(d_sh_y) * float(df["y"].iloc[i]) + abs(d_sh_x) * float(df["x"].iloc[i])
        shares_traded = abs(d_sh_y) + abs(d_sh_x)

        commission = 0.0
        if shares_traded > 0:
            commission = max(min_commission, commission_per_share * shares_traded)

        slip_cost = (slippage_bps / 10000.0) * turnover
        cost = commission + slip_cost

        df.iloc[i, df.columns.get_loc("sh_y")] = sh_y
        df.iloc[i, df.columns.get_loc("sh_x")] = sh_x
        df.iloc[i, df.columns.get_loc("cost")] = cost

        # PnL from i-1 -> i on previous shares
        dx = float(df["x"].iloc[i] - df["x"].iloc[i - 1])
        dy = float(df["y"].iloc[i] - df["y"].iloc[i - 1])

        gross = prev_sh_x * dx + prev_sh_y * dy
        net = gross - cost

        df.iloc[i, df.columns.get_loc("gross_pnl")] = gross
        df.iloc[i, df.columns.get_loc("net_pnl")] = net

        net_ret = net / (deployed_capital + 1e-12)
        df.iloc[i, df.columns.get_loc("net_ret")] = net_ret

    df["cum_net_pnl"] = df["net_pnl"].cumsum()
    df["cum_net_ret"] = (1.0 + df["net_ret"]).cumprod() - 1.0
    return df


def sharpe_like(daily_ret: pd.Series) -> float:
    daily_ret = daily_ret.dropna()
    if len(daily_ret) < 10:
        return float("nan")
    return float(daily_ret.mean() / (daily_ret.std(ddof=0) + 1e-12) * np.sqrt(252))
