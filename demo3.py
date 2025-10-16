
# agent_pipeline_refactor_final.py
# Single-file agent pipeline — final version (Cách B robust preprocess)
# - handles MultiIndex columns from yfinance
# - safer financial serialization
# - selective dropna instead of global dropna
# - guarded metric computation and training size checks
# - ReportAgent.generate returns report object (interactive=False by default)

import yfinance as yf
import pandas as pd
import numpy as np
import json, os
from prophet import Prophet
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# =================== UTILS / FEATURES ===================
def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))


def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# =================== MEMORY (file-based) ===================
class Memory:
    """Simple file-backed storage. Values should be JSON-serializable.
    Use orient='records' for DataFrames or fallback to .to_json() strings.
    """

    def __init__(self, fn="agent_memory.json"):
        self.fn = fn
        if not os.path.exists(fn):
            with open(fn, "w") as f:
                json.dump({}, f)

    def _read(self):
        with open(self.fn, "r") as f:
            try:
                return json.load(f)
            except Exception:
                return {}

    def _write(self, data):
        with open(self.fn, "w") as f:
            json.dump(data, f, indent=4, default=str)

    def store(self, key, val):
        data = self._read()
        data[key] = val
        self._write(data)

    def retrieve(self, key):
        data = self._read()
        return data.get(key, None)


# =================== AGENT BASE ===================
class Agent:
    def __init__(self, name, memory=None):
        self.name = name
        self.memory = memory

    def log(self, text):
        logging.info(f"[{self.name}] {text}")


# =================== DataAgent ===================
class DataAgent(Agent):
    def fetch(self, ticker, period="1y", interval="1d", persist: bool = False):
        """Fetch price history and basic financials from yfinance

        Returns
        -------
        pd.DataFrame
            DataFrame with at least `Date` and `Close` columns (may be MultiIndex before normalization)
        """
        self.log(f"Fetching {ticker} price data ({period}, {interval})...")
        df = yf.download(ticker, period=period, interval=interval, progress=False)

        # yf.download may return MultiIndex columns and/or have Date as index
        # keep df as returned; callers are responsible for normalization

        # try to fetch financial statements and store them safely
        try:
            tk = yf.Ticker(ticker)
            fin = tk.financials
            q_fin = tk.quarterly_financials
            if persist and fin is not None and not fin.empty:
                # convert to records for JSON safety
                fin_rec = fin.reset_index().fillna(0).to_dict(orient="records")
                if self.memory is not None:
                    self.memory.store(f"{ticker}_financials", fin_rec)
            if persist and q_fin is not None and not q_fin.empty:
                q_fin_rec = q_fin.reset_index().fillna(0).to_dict(orient="records")
                if self.memory is not None:
                    self.memory.store(f"{ticker}_q_financials", q_fin_rec)
        except Exception as e:
            self.log("Warning: could not fetch full financials: " + str(e))

        if persist and self.memory is not None:
            self.memory.store("last_fetch_time", str(datetime.now()))

        return df


# =================== PreprocessingAgent ===================
class PreprocessingAgent(Agent):
    def preprocess(self, df: pd.DataFrame, ticker: str = None, persist: bool = False) -> pd.DataFrame:
        """
        Robust preprocess: handle MultiIndex columns, ensure a Date column exists (try index -> columns -> infer),
        then compute features.
        """
        self.log("Start preprocessing & feature engineering...")
        df = df.copy()

        # 1) If MultiIndex columns, try to extract single ticker slice
        if isinstance(df.columns, pd.MultiIndex):
            try:
                if 'Ticker' in df.columns.names and ticker is not None:
                    df = df.xs(ticker, axis=1, level='Ticker')
                elif ticker is not None:
                    df = df.xs(ticker, axis=1, level=1)
                else:
                    first_tk = df.columns.levels[1][0]
                    df = df.xs(first_tk, axis=1, level=1)
                df = df.reset_index()
                df.columns = [str(c) for c in df.columns]
                self.log(f"Preprocess: extracted ticker slice for '{ticker if ticker else first_tk}' from MultiIndex columns.")
            except Exception as e:
                self.log(f"Preprocess MultiIndex extraction failed: {e}. Attempting flatten fallback.")
                flat = [f"{a}_{b}" for a, b in df.columns]
                df.columns = flat
                df = df.reset_index()
                # try rename first close-like column to 'Close'
                for c in df.columns:
                    if c.lower().startswith('close_'):
                        df = df.rename(columns={c: 'Close'})
                        break

        # 2) Ensure we have a Date column. Try multiple methods.
        if 'Date' not in df.columns:
            # if the index is a DatetimeIndex, reset it and name column 'Date'
            if isinstance(df.index, pd.DatetimeIndex):
                try:
                    df = df.reset_index()
                    # if reset_index created a column with different name, rename the first column to 'Date'
                    if df.columns[0] != 'Date':
                        df = df.rename(columns={df.columns[0]: 'Date'})
                    self.log("Preprocess: used DatetimeIndex -> reset_index to create 'Date' column.")
                except Exception as e:
                    self.log(f"Preprocess: failed to reset DatetimeIndex: {e}")
            else:
                # try to find any column that looks like datetime
                found = False
                for c in df.columns:
                    try:
                        if np.issubdtype(df[c].dtype, np.datetime64):
                            df = df.rename(columns={c: 'Date'})
                            found = True
                            self.log(f"Preprocess: found datetime-like column '{c}' -> renamed to 'Date'.")
                            break
                    except Exception:
                        continue
                # try lowercase 'date'
                if not found:
                    for c in df.columns:
                        if str(c).lower() == 'date':
                            df = df.rename(columns={c: 'Date'})
                            found = True
                            self.log(f"Preprocess: found column named '{c}', normalized to 'Date'.")
                            break
                # if still not found, as last resort, try first column (with warning)
                if not found:
                    self.log(f"Preprocess: 'Date' column not found. Attempting fallback to first column -> '{df.columns[0]}'. This may be unsafe.")
                    df = df.reset_index() if not df.index.name else df.reset_index()
                    try:
                        df = df.rename(columns={df.columns[0]: 'Date'})
                    except Exception:
                        pass

        # Final check
        if 'Date' not in df.columns:
            raise KeyError("Input data does not contain 'Date' column after normalization. Provide ticker or check fetch output.")

        # convert Date to datetime
        try:
            df['Date'] = pd.to_datetime(df['Date'])
        except Exception as e:
            raise ValueError(f"Unable to convert Date column to datetime: {e}")

        # 3) ensure Close exists
        if 'Close' not in df.columns:
            raise KeyError("Input data does not contain 'Close' column after normalization. Provide ticker or check fetch output.")

        # 4) proceed with feature engineering
        df = df.dropna(subset=['Close']).sort_values('Date').reset_index(drop=True)

        df['Return'] = df['Close'].pct_change()
        df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))

        for w in [5, 10, 20, 50]:
            df[f"MA_{w}"] = df['Close'].rolling(w).mean()

        df['RSI_14'] = rsi(df['Close'], 14)
        macd_line, macd_sig, macd_hist = macd(df['Close'])
        df['MACD'] = macd_line
        df['MACD_HIST'] = macd_hist

        df['Vol_20'] = df['LogRet'].rolling(20).std()
        df['Vol_60'] = df['LogRet'].rolling(60).std()

        df['Mom_10'] = df['Close'] / df['Close'].shift(10) - 1

        required_for_model = ['Close', 'Return', 'MA_20', 'RSI_14', 'MACD_HIST', 'Vol_20']
        existing_required = [c for c in required_for_model if c in df.columns]
        df = df.dropna(subset=existing_required).reset_index(drop=True)

        if persist and self.memory is not None:
            self.memory.store("last_preprocess_time", str(datetime.now()))
        return df


# =================== PredictionAgent ===================
class PredictionAgent(Agent):
    def __init__(self, name, memory=None):
        super().__init__(name, memory)
        self.classifier = None

    def make_labels(self, df: pd.DataFrame, horizon: int = 5, threshold: float = 0.0) -> pd.DataFrame:
        df = df.copy()
        df['FutureClose'] = df['Close'].shift(-horizon)
        df['FutureRet'] = df['FutureClose'] / df['Close'] - 1
        df['Label'] = (df['FutureRet'] > threshold).astype(int)
        df = df.dropna().reset_index(drop=True)
        return df

    def train_classifier(self, df: pd.DataFrame, feature_cols: list = None, horizon: int = 5, persist: bool = False) -> dict:
        df_lab = self.make_labels(df, horizon=horizon)
        if feature_cols is None:
            feature_cols = ['Return', 'MA_5', 'MA_10', 'MA_20', 'RSI_14', 'MACD_HIST', 'Vol_20', 'Mom_10']

        # keep only available feature columns
        feature_cols = [c for c in feature_cols if c in df_lab.columns]
        X = df_lab[feature_cols]
        y = df_lab['Label']

        # time-aware split
        split = int(0.8 * len(X))
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        X_test, y_test = X.iloc[split:], y.iloc[split:]

        # sanity checks
        min_train = 20
        min_test = 5
        if len(X_train) < min_train or len(X_test) < min_test:
            raise ValueError(f"Not enough data to train/test classifier: train={len(X_train)}, test={len(X_test)}")
        if len(np.unique(y_train)) < 2:
            raise ValueError("Training labels contain only one class; cannot fit classifier")

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))
        ])
        pipe.fit(X_train, y_train)
        self.classifier = pipe

        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, 'predict_proba') else preds

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, zero_division=0)
        auc = float('nan')
        if len(np.unique(y_test)) == 2:
            try:
                auc = roc_auc_score(y_test, probs)
            except Exception:
                auc = float('nan')

        metrics = {"accuracy": acc, "f1": f1, "auc": auc, "n_test": len(X_test)}
        if persist and self.memory is not None:
            self.memory.store("clf_metrics", metrics)

        self.log(f"Trained classifier: acc={acc:.4f}, f1={f1:.4f}, auc={auc}")
        last_features = X.iloc[-1:]
        return {"metrics": metrics, "last_features": last_features}

    def predict_short_term(self, last_features: pd.DataFrame) -> dict:
        if self.classifier is None:
            raise ValueError("Classifier not trained")
        prob = self.classifier.predict_proba(last_features)[:, 1][0]
        label = int(prob > 0.5)
        self.log(f"Short-term prediction prob={prob:.3f}, label={label}")
        return {"prob": float(prob), "label": label}

    def forecast_long_term(self, df: pd.DataFrame, periods: int = 30, persist: bool = False) -> pd.DataFrame:
        """Prophet forecast as long-term forecasting. Uses df with Date & Close columns."""
        self.log("Running Prophet long-term forecast...")
        dfp = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        dfp = dfp.dropna()
        if len(dfp) < 10:
            raise ValueError("Not enough data for Prophet fit")
        m = Prophet(daily_seasonality=False, yearly_seasonality=True)
        m.fit(dfp)
        future = m.make_future_dataframe(periods=periods)
        fc = m.predict(future)
        if persist and self.memory is not None:
            try:
                self.memory.store("last_forecast_head", fc[['ds', 'yhat']].tail(periods).to_dict())
            except Exception:
                self.memory.store("last_forecast_head_str", fc[['ds', 'yhat']].tail(periods).to_json())
        return fc


# =================== RiskManagementAgent ===================
class RiskManagementAgent(Agent):
    def evaluate(self, df: pd.DataFrame, classifier_metrics: dict, forecast_df: pd.DataFrame,
                 threshold_acc: float = 0.52, threshold_sharpe: float = 0.5) -> dict:
        self.log("Risk evaluation started...")
        acc = classifier_metrics.get("accuracy", 0) if classifier_metrics else 0
        n_test = classifier_metrics.get("n_test", 0) if classifier_metrics else 0
        flag = False
        reasons = []

        if acc < threshold_acc:
            flag = True
            reasons.append(f"Classifier accuracy {acc:.3f} < {threshold_acc}")

        df_eval = df.copy().reset_index(drop=True)
        window = min(60, len(df_eval) - 1)
        if window <= 1:
            mean_ret = 0.0
            vol = 0.0
            sharpe = 0.0
        else:
            df_bt = df_eval.tail(window).copy()
            df_bt['NextRet'] = df_bt['Close'].pct_change().shift(-1)
            mean_ret = df_bt['NextRet'].mean()
            vol = df_bt['NextRet'].std()
            sharpe = (mean_ret / (vol + 1e-9)) * np.sqrt(252) if vol > 0 else 0

        self.log(f"Backtest approx mean_ret={mean_ret:.5f}, vol={vol:.5f}, sharpe={sharpe:.3f}")
        if sharpe < threshold_sharpe:
            flag = True
            reasons.append(f"Sharpe {sharpe:.3f} < {threshold_sharpe}")

        # forecast uncertainty: check if prophet fc has yhat_lower/yhat_upper
        if isinstance(forecast_df, pd.DataFrame) and 'yhat_upper' in forecast_df.columns and 'yhat_lower' in forecast_df.columns:
            fut = forecast_df[forecast_df['ds'] > df['Date'].max()]
            if not fut.empty:
                avg_width = (fut['yhat_upper'] - fut['yhat_lower']).mean()
                rel_width = avg_width / df['Close'].iloc[-1]
                self.log(f"Forecast relative interval width ~ {rel_width:.4f}")
                if rel_width > 0.1:
                    reasons.append("Forecast uncertainty high (>10% relative width)")
                    flag = True

        result = {"flag": flag, "reasons": reasons, "metrics": {"sharpe": sharpe, "mean_ret": mean_ret, "vol": vol}}
        if self.memory is not None:
            self.memory.store("risk_eval", result)
        return result


# =================== ReportAgent ===================
class ReportAgent(Agent):
    def generate(self, ticker: str, df_raw: pd.DataFrame, df_proc: pd.DataFrame,
                 forecast_df: pd.DataFrame, clf_pred: dict, risk_eval: dict, interactive: bool = False) -> dict:
        """Generate enhanced report with visualization and decision support."""
        self.log("Generating enhanced report with decision support...")

        saved = False  # ✅ fix lỗi NameError

        # --- Retrieve stored metrics ---
        clf_metrics = self.memory.retrieve("clf_metrics") if self.memory is not None else None
        short_term_prob = clf_pred.get("prob", None) if clf_pred else None
        short_term_label = clf_pred.get("label", None) if clf_pred else None

        # --- Prepare historical data ---
        hist = df_proc[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
        fc_future = pd.DataFrame()
        if isinstance(forecast_df, pd.DataFrame) and "ds" in forecast_df.columns and "yhat" in forecast_df.columns:
            fc_future = forecast_df[forecast_df["ds"] > hist["ds"].max()]

        # --- Compute simple trend slope on forecast ---
        slope = 0.0
        if not fc_future.empty:
            y0 = fc_future["yhat"].iloc[0]
            y1 = fc_future["yhat"].iloc[-1]
            days = (fc_future["ds"].iloc[-1] - fc_future["ds"].iloc[0]).days + 1
            slope = (y1 - y0) / max(days, 1)

        # --- Decision rules ---
        action = "HOLD"
        if short_term_prob is not None:
            if short_term_prob > 0.6 and slope > 0:
                action = "BUY"
            elif short_term_prob < 0.4 and slope < 0:
                action = "SELL"

        # --- Print decision summary ---
        print("\n========== AI AGENT DECISION REPORT ==========")
        print(f"Ticker: {ticker}")
        print(f"Short-term prediction: prob={short_term_prob:.3f} | label={short_term_label}")
        print(f"Forecast trend slope: {slope:.5f}")
        print(f"Suggested action: {action}")
        print(f"Risk evaluation: {risk_eval}")
        print("=============================================\n")

        # --- Plot visualization (do not call fig.show() here to avoid duplicate rendering) ---
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Scatter(x=hist["ds"], y=hist["y"], mode="lines", name="Historical Close"), row=1, col=1)

        if not fc_future.empty:
            fig.add_trace(go.Scatter(x=fc_future["ds"], y=fc_future["yhat"], mode="lines", name="Forecast (Prophet)",
                                     line=dict(color="orange", dash="dash")), row=1, col=1)
            if "yhat_upper" in fc_future.columns and "yhat_lower" in fc_future.columns:
                fig.add_trace(go.Scatter(
                    x=pd.concat([fc_future["ds"], fc_future["ds"][::-1]]),
                    y=pd.concat([fc_future["yhat_upper"], fc_future["yhat_lower"][::-1]]),
                    fill="toself", fillcolor="rgba(255,165,0,0.2)", line=dict(color="rgba(255,165,0,0)"),
                    hoverinfo="skip", showlegend=True, name="Forecast Interval"
                ), row=1, col=1)
        else:
            self.log("No future forecast available to plot (fc_future empty).")

        # Short-term prediction marker
        if clf_pred is not None and "prob" in clf_pred:
            last_date = hist["ds"].iloc[-1]
            last_price = hist["y"].iloc[-1]
            color = "green" if clf_pred["label"] == 1 else "red"
            symbol = "triangle-up" if clf_pred["label"] == 1 else "triangle-down"
            text = f"Prob={clf_pred['prob']:.2f}, Action={action}"
            fig.add_trace(go.Scatter(
                x=[last_date], y=[last_price],
                mode="markers+text", name="Short-term Signal",
                marker=dict(size=12, color=color, symbol=symbol),
                text=[text], textposition="bottom right"
            ), row=1, col=1)

        # Volatility subplot
        if isinstance(df_proc, pd.DataFrame) and "Vol_20" in df_proc.columns:
            fig.add_trace(go.Scatter(x=df_proc["Date"], y=df_proc["Vol_20"], mode="lines", name="Vol_20"), row=2, col=1)

        fig.update_layout(title=f"{ticker} - Forecast & Decision Support (Action: {action})",
                          xaxis_title="Date", yaxis_title="Price")

        # --- Save summary for later retrieval ---
        summary = {
            "ticker": ticker,
            "short_term_prob": short_term_prob,
            "short_term_label": short_term_label,
            "forecast_slope": slope,
            "suggested_action": action,
            "risk_eval": risk_eval,
            "clf_metrics": clf_metrics
        }
        if self.memory is not None:
            self.memory.store(f"{ticker}_decision_summary", summary)

        return {"summary": summary, "fig": fig, "saved": saved}


# =================== ORCHESTRATOR ===================
class Orchestrator:
    def __init__(self, agents: dict):
        self.agents = agents

    def run_for(self, ticker, period="1y", interval="1d", horizon=5, fc_periods=30, persist: bool = True):
        self.agents['data'].log(f"Orchestrator starting for {ticker}")

        raw = self.agents['data'].fetch(ticker, period=period, interval=interval, persist=persist)

        try:
            proc = self.agents['pre'].preprocess(raw, ticker=ticker, persist=persist)
        except AssertionError as e:
            self.agents['data'].log(f"Preprocess failed: {e}")
            return
        except KeyError as ke:
            self.agents['data'].log(f"Preprocess failed: {ke}")
            return

        # forecast on raw to avoid losing days from heavy dropna
        try:
            fc = self.agents['pred'].forecast_long_term(raw, periods=fc_periods, persist=persist)
        except Exception as e:
            self.agents['data'].log(f"Forecast failed: {e}")
            fc = pd.DataFrame()

        # forecast prefer on processed data (proc) because it has Date & Close normalized
        fc = pd.DataFrame()
        try:
            if isinstance(proc, pd.DataFrame) and not proc.empty:
                self.agents['data'].log("Forecasting on preprocessed data (preferred).")
                fc = self.agents['pred'].forecast_long_term(proc, periods=fc_periods, persist=persist)
            else:
                self.agents['data'].log("Preprocessed data empty — trying forecast on raw data.")
                fc = self.agents['pred'].forecast_long_term(raw, periods=fc_periods, persist=persist)
        except Exception as e:
            self.agents['data'].log(f"Forecast failed: {e}")
            fc = pd.DataFrame()

        # debug: log forecast head & columns
        if isinstance(fc, pd.DataFrame) and not fc.empty:
            self.agents['data'].log(f"Forecast produced {len(fc)} rows; columns: {list(fc.columns)[:10]}")
        else:
            self.agents['data'].log("Forecast DataFrame is empty.")

        # train/predict classifier (may fail if not enough data)
        clf_pred = None
        try:
            clf_info = self.agents['pred'].train_classifier(proc, horizon=horizon, persist=persist)
            clf_pred = self.agents['pred'].predict_short_term(clf_info['last_features'])
        except Exception as e:
            self.agents['pred'].log(f"Classifier training/prediction skipped due to: {e}")

        risk = self.agents['risk'].evaluate(proc, self.agents['pred'].memory.retrieve("clf_metrics") if self.agents['pred'].memory is not None else {}, fc)
        report = self.agents['report'].generate(ticker, raw, proc, fc, clf_pred, risk, interactive=False)

        # show figure if running interactively in a notebook or dev env
        try:
            report['fig'].show()
        except Exception:
            pass


# =================== MAIN RUN (example) ===================
if __name__ == "__main__":
    mem = Memory()
    agents = {
        "data": DataAgent("DataAgent", mem),
        "pre": PreprocessingAgent("PreprocessingAgent", mem),
        "pred": PredictionAgent("PredictionAgent", mem),
        "risk": RiskManagementAgent("RiskAgent", mem),
        "report": ReportAgent("ReportAgent", mem)
    }
    orchestrator = Orchestrator(agents)

    # Example run for multiple tickers (sequential). Adjust period / horizon as needed.
    tickers = ["GOOG"]
    for t in tickers:
        orchestrator.run_for(ticker=t, period="1y", interval="1d", horizon=5, fc_periods=30, persist=True)




