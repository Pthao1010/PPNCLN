# demo2_optimized.py
"""
Optimized AI Agent pipeline (refactor of demo2/combine_test2_final)
Features:
 - Ensemble of LGBM + RandomForest with optional probability calibration
 - Prophet forecasting with configurable changepoint / seasonality
 - Allow cutting historical data at a cutoff date for controlled experiments
 - Robust Preprocessing to handle yfinance MultiIndex/Adj Close variants
 - Clear logging and single fig.show()
"""

import os, json, time, logging, warnings
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from prophet import Prophet

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV

import plotly.graph_objs as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# ---------------- CONFIG ----------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # optional LLM (OpenRouter/Gemini)
OPENROUTER_ENDPOINT = "https://api.openrouter.ai/v1/chat/completions"

DEFAULT_PERIOD = "1y"
DEFAULT_INTERVAL = "1d"

FORECAST_DAYS = 30
HORIZON = 5  # classification horizon in days
CALIBRATE_PROBS = True
ENSEMBLE_WEIGHT_BY_ACC = True  # if False => equal average
PROPHET_CHANGEP = 0.01
PROPHET_SEASONALITY_MODE = "multiplicative"

# ---------------- small utils / indicators ----------------
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

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

# ---------------- Memory ----------------
class Memory:
    def __init__(self):
        self._store = {}
    def store(self, k, v): self._store[k] = v
    def retrieve(self, k): return self._store.get(k)

# ---------------- Base Agent ----------------
class Agent:
    def __init__(self, name, memory=None):
        self.name = name
        self.memory = memory
    def log(self, text):
        logging.info(f"[{self.name}] {text}")

# ---------------- DataAgent ----------------
class DataAgent(Agent):
    def fetch(self, ticker, period=DEFAULT_PERIOD, interval=DEFAULT_INTERVAL):
        self.log(f"Fetching {ticker} ({period},{interval})")
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            if df is None or df.empty:
                self.log("No data returned.")
                return pd.DataFrame()
            df = df.reset_index()
            df.columns = [str(c) for c in df.columns]
            return df
        except Exception as e:
            self.log(f"Fetch error: {e}")
            return pd.DataFrame()

# ---------------- PreprocessingAgent ----------------
class PreprocessingAgent(Agent):
    def preprocess(self, df: pd.DataFrame, ticker: str = None, cutoff_date: str = None, persist: bool = False) -> pd.DataFrame:
        self.log("Start preprocessing & feature engineering...")
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty in preprocess")

        df = df.copy()

        # flatten MultiIndex if exists
        if isinstance(df.columns, pd.MultiIndex):
            try:
                cols = []
                for a,b in df.columns:
                    cols.append(f"{a}_{b}" if b != '' else str(a))
                df.columns = cols
            except Exception:
                df.columns = [str(c) for c in df.columns]

        # ensure Date col
        if 'Date' not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                if df.columns[0] != 'Date':
                    df = df.rename(columns={df.columns[0]:'Date'})
            else:
                # find any datetime-like column
                for c in df.columns:
                    if np.issubdtype(df[c].dtype, np.datetime64):
                        df = df.rename(columns={c:'Date'})
                        break
                if 'Date' not in df.columns:
                    df = df.reset_index()
                    if df.columns[0] != 'Date':
                        df = df.rename(columns={df.columns[0]:'Date'})

        # detect Close column flexibly
        close_col = None
        for c in df.columns:
            if str(c).strip().lower() in ('close','adj close','adj_close'):
                close_col = c; break
        if close_col is None:
            for c in df.columns:
                if 'close' in str(c).lower():
                    close_col = c; break
        if close_col is None:
            self.log(f"Columns: {list(df.columns)[:40]}")
            raise KeyError("No Close-like column found")
        if close_col != 'Close':
            df = df.rename(columns={close_col: 'Close'})

        # convert Date
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').dropna(subset=['Close']).reset_index(drop=True)

        # cutoff_date optional
        if cutoff_date:
            try:
                cut = pd.to_datetime(cutoff_date)
                df = df[df['Date'] <= cut].reset_index(drop=True)
                self.log(f"Cut data to {cut.date()}, rows now {len(df)}")
                if df.empty:
                    raise ValueError("No rows after cutoff_date filter")
            except Exception as e:
                self.log(f"cutoff_date parse failed: {e}")

        # features
        df['Return'] = df['Close'].pct_change()
        df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))

        for w in [3,5,8,10,12,20,50,100]:
            df[f"MA_{w}"] = df['Close'].rolling(window=w, min_periods=1).mean()
            df[f"EMA_{w}"] = ema(df['Close'], span=w)

        df['RSI_14'] = rsi(df['Close'],14)
        macd_line, macd_sig, macd_hist = macd(df['Close'])
        df['MACD_HIST'] = macd_hist

        df['Vol_20'] = df['LogRet'].rolling(20).std()
        df['Mom_5'] = df['Close'].pct_change(5)

        # drop rows missing core features
        required = ['Close','Return','MA_20','RSI_14','MACD_HIST','Vol_20']
        existing = [c for c in required if c in df.columns]
        df = df.dropna(subset=existing).reset_index(drop=True)
        if df.empty:
            raise ValueError("Preprocessed dataframe empty after feature engineering")

        if self.memory is not None and persist:
            self.memory.store("last_preprocess_time", str(datetime.now()))
        self.log(f"Preprocess done. rows={len(df)} cols={len(df.columns)}")
        return df

# ---------------- PredictionAgent (train LGBM+RF + ensemble + calibration) ----------------
class PredictionAgent(Agent):
    def __init__(self, name, memory=None):
        super().__init__(name, memory)
        self.pipes = {}      # trained pipelines
        self.calibrators = {}  # optionally calibrated wrappers
        self.weights = {}    # ensemble weights
        self.best_pipe = None

    def make_labels(self, df: pd.DataFrame, horizon=HORIZON):
        df2 = df.copy()
        df2['FutureClose'] = df2['Close'].shift(-horizon)
        df2['FutureRet'] = df2['FutureClose'] / df2['Close'] - 1
        df2['Label'] = (df2['FutureRet'] > 0).astype(int)
        df2 = df2.dropna().reset_index(drop=True)
        return df2

    def train_models(self, df: pd.DataFrame, horizon=HORIZON):
        self.log("Training LGBM + RF models...")
        df_lab = self.make_labels(df, horizon=horizon)
        feature_cols = ['Return','MA_5','MA_10','MA_20','RSI_14','MACD_HIST','Vol_20','Mom_5']
        feature_cols = [c for c in feature_cols if c in df_lab.columns]
        if len(df_lab) < 30 or len(feature_cols) < 4:
            raise ValueError("Not enough data/features to train")

        X = df_lab[feature_cols]; y = df_lab['Label']
        split = int(0.8 * len(X))
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        models = {
            "LGBM": LGBMClassifier(n_estimators=200, max_depth=6, random_state=42),
            "RF": RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
        }

        results = {}
        best_acc = -1.0
        best_name = None

        for name, mdl in models.items():
            try:
                pipe = Pipeline([('scaler', StandardScaler()), (name, mdl)])
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_test)
                probs = pipe.predict_proba(X_test)[:,1] if hasattr(pipe, 'predict_proba') else np.zeros(len(preds))
                acc = float(accuracy_score(y_test, preds))
                f1 = float(f1_score(y_test, preds, zero_division=0))
                auc = float(roc_auc_score(y_test, probs)) if len(np.unique(y_test))==2 else float('nan')
                results[name] = {"acc": acc, "f1": f1, "auc": auc, "n_test": len(X_test)}
                self.log(f"{name}: acc={acc:.4f}, f1={f1:.4f}, auc={auc if not np.isnan(auc) else 'nan'}")
                self.pipes[name] = pipe
                if acc > best_acc:
                    best_acc = acc; best_name = name; self.best_pipe = pipe
            except Exception as e:
                self.log(f"{name} training failed: {e}")

        if not self.pipes:
            raise RuntimeError("No models trained")

        # calibration on X_test if desired (use part of test as calibration target)
        if CALIBRATE_PROBS:
            for name, pipe in list(self.pipes.items()):
                try:
                    # calibrate with CalibratedClassifierCV using prefit
                    calib = CalibratedClassifierCV(pipe, method='sigmoid', cv='prefit')
                    calib.fit(X_test, y_test)
                    self.calibrators[name] = calib
                    self.log(f"{name} calibrated on holdout")
                except Exception as e:
                    self.log(f"Calibration failed for {name}: {e}")

        # ensemble weights
        if ENSEMBLE_WEIGHT_BY_ACC:
            accs = np.array([results[n]['acc'] if n in results else 0.0 for n in self.pipes.keys()], dtype=float)
            if accs.sum() > 0:
                ws = accs / accs.sum()
            else:
                ws = np.ones(len(accs)) / len(accs)
            self.weights = dict(zip(list(self.pipes.keys()), ws.tolist()))
        else:
            n = len(self.pipes); self.weights = dict(zip(list(self.pipes.keys()), [1.0/n]*n))

        # store metrics and last_features
        self.memory.store("model_metrics", results)
        last_features = X.iloc[-1:].copy()
        return {"metrics": results, "last_features": last_features, "best_name": best_name}

    def predict_ensemble(self, last_features: pd.DataFrame):
        names = list(self.pipes.keys())
        probs = {}
        for name in names:
            try:
                if name in self.calibrators:
                    p = float(self.calibrators[name].predict_proba(last_features)[:,1][0])
                else:
                    p = float(self.pipes[name].predict_proba(last_features)[:,1][0])
            except Exception:
                p = float(self.pipes[name].predict_proba(last_features)[:,1][0])
            probs[name] = p
        # weighted average
        total = 0.0; denom = 0.0
        for n,w in self.weights.items():
            total += w * probs.get(n,0.0); denom += w
        combined = total / (denom + 1e-9)
        label = int(combined > 0.5)
        return {"prob_ml": combined, "label": label, "probs_by_model": probs}

    def predict_short_term(self, last_features: pd.DataFrame = None, df: pd.DataFrame = None):
        if last_features is None:
            if df is None: raise ValueError("Need last_features or df for prediction")
            # try to reconstruct last_feature set
            cols = ['Return','MA_5','MA_10','MA_20','RSI_14','MACD_HIST','Vol_20','Mom_5']
            cols = [c for c in cols if c in df.columns]
            last_features = df[cols].iloc[-1:].copy()
        res = self.predict_ensemble(last_features)
        self.log(f"Short-term ML combined prob={res['prob_ml']:.4f}, label={res['label']}")
        return res

    def forecast_long_term(self, df: pd.DataFrame, periods=FORECAST_DAYS, changep=PROPHET_CHANGEP,
                           seasonality_mode=PROPHET_SEASONALITY_MODE, do_tune: bool = True):
        """
        Prophet forecast with optional tuning.
        - df: original preprocessed df with Date & Close
        - periods: number of days to forecast
        - changep: fallback changepoint_prior_scale
        - seasonality_mode: 'additive' or 'multiplicative'
        - do_tune: if True, run tune_prophet and use best scale (may be slow)
        """
        self.log(f"Running Prophet forecast (initial changep={changep}, seasonality={seasonality_mode})...")
        dfp = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'}).dropna()
        if len(dfp) < 20:
            raise ValueError("Not enough data for Prophet")

        # Optional tuning (may be slow). tune_prophet should accept original df or dfp; here we pass original df for compatibility
        best_scale = None
        if do_tune:
            try:
                best_scale = tune_prophet(df)  # tune_prophet returns None or numeric
                self.log(f"Prophet tuning returned best_scale={best_scale}")
            except Exception as e:
                self.log(f"Prophet tuning failed: {e}")
                best_scale = None

        # Decide final changepoint_prior_scale
        final_cp = best_scale if (best_scale is not None) else changep

        # Build Prophet model with chosen params
        m = Prophet(daily_seasonality=False, yearly_seasonality=True, seasonality_mode=seasonality_mode,
                    changepoint_prior_scale=float(final_cp))
        m.fit(dfp)

        future = m.make_future_dataframe(periods=periods, freq='D')
        fc = m.predict(future)

        # store head if memory
        if self.memory is not None:
            try:
                self.memory.store("last_forecast_head", fc[['ds', 'yhat']].tail(periods).to_dict())
            except Exception:
                self.memory.store("last_forecast_head_str", fc[['ds', 'yhat']].tail(periods).to_json())
        
# Apply growth constraint: limit daily change to ±{MAX_DAILY_CHANGE*100:.1f}%
        max_daily_change = 0.2

        last_price = df['Close'].iloc[-1]
        fc = fc.copy()
        prev_price = last_price
        for i in range(len(fc)):
            if fc['ds'].iloc[i] > df['Date'].max():
                raw_pred = fc.at[i, 'yhat']
                max_up = prev_price * (1 + max_daily_change)
                max_down = prev_price * (1 - max_daily_change)
                capped_pred = min(max_up, max(max_down, raw_pred))
                fc.at[i, 'yhat'] = capped_pred
                prev_price = capped_pred
        return fc


# =================== MODULE A: BacktesterAgent
# ===================
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class BacktesterAgent(Agent):
    """
    Thực hiện walk-forward validation (cuộn cửa sổ thời gian)
    để kiểm định mô hình ML theo thời gian thực tế.
    """

    def walk_forward_test(self, df, feature_cols, label_col='Label', horizon=5, step=30, model_class= RandomForestClassifier):
        """
        df: DataFrame đã có Label (dùng make_labels của PredictionAgent)
        feature_cols: các cột đặc trưng (feature)
        horizon: dự báo sau bao nhiêu ngày
        step: số ngày dịch mỗi lần kiểm tra (ví dụ 30 = mỗi tháng)
        model_class: lớp ML (RandomForestClassifier, LGBMClassifier,...)
        """
        if len(df) < 100:
            raise ValueError("Dữ liệu quá ngắn để chạy backtest.")

        results = []
        self.log(f"Bắt đầu Walk-forward test (window step={step}, horizon={horizon})...")
        for start in range(0, len(df) - step * 2, step):
            train = df.iloc[start:start + step * 3]  # 3 tháng train
            test = df.iloc[start + step * 3:start + step * 4]  # 1 tháng test
            if len(train) < 50 or len(test) < 10:
                continue

            X_train, y_train = train[feature_cols], train[label_col]
            X_test, y_test = test[feature_cols], test[label_col]

            if len(np.unique(y_train)) < 2:  # cần ít nhất 2 lớp
                continue

            model = model_class(n_estimators=100, max_depth=6, random_state=42)
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
            preds = (probs > 0.5).astype(int)

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, zero_division=0)
            auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) == 2 else np.nan

            results.append({"start": df['Date'].iloc[start], "acc": acc, "f1": f1, "auc": auc})

        res_df = pd.DataFrame(results)
        if not res_df.empty:
            self.log(f"Walk-forward Accuracy mean: {res_df['acc'].mean():.3f}, F1 mean: {res_df['f1'].mean():.3f}")
            # Vẽ biểu đồ kết quả
            res_df.plot(x='start', y=['acc', 'f1'], kind='line', title='Walk-forward Metrics over Time')
        else:
            self.log("Không đủ dữ liệu để tính backtest.")

        if model_class is None:
            raise ValueError("model_class cannot be None. Pass a valid classifier class.")

        return res_df


# =================== MODULE B: Prophet Tuning ===================
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation


def tune_prophet(df, period=30, changepoints=[0.001, 0.01, 0.05, 0.1, 0.3], plot_result=True):
    """
    Tối ưu tham số Prophet (changepoint_prior_scale) bằng cross-validation
    """
    print("[Prophet Tuning] Running cross-validation...")
    best_score = float('inf')
    best_scale = None
    results = []

    for c in changepoints:
        try:
            m = Prophet(changepoint_prior_scale=c, yearly_seasonality=True)
            m.fit(df.rename(columns={'Date': 'ds', 'Close': 'y'}))
            df_cv = cross_validation(m, initial='180 days', period='30 days', horizon=f'{period} days')
            df_p = performance_metrics(df_cv)
            rmse = df_p['rmse'].mean()
            results.append({'scale': c, 'rmse': rmse})
            print(f"Scale={c} -> RMSE={rmse:.4f}")
            if rmse < best_score:
                best_score = rmse
                best_scale = c
        except Exception as e:
            print(f"Failed for scale={c}: {e}")
            continue

    res_df = pd.DataFrame(results)
    if plot_result and not res_df.empty:
        res_df.plot(x='scale', y='rmse', marker='o', title='Prophet Tuning (RMSE vs changepoint scale)')
    print(f"✅ Best scale={best_scale}, RMSE={best_score:.4f}")
    return best_scale


# ---------------- RiskManagementAgent ----------------
class RiskManagementAgent(Agent):
    def evaluate(self, df: pd.DataFrame, classifier_metrics: dict = None, forecast_df: pd.DataFrame = None, threshold_acc: float = 0.52, threshold_sharpe: float = 0.5):
        self.log("Risk evaluation started...")
        acc = float(classifier_metrics.get("accuracy", 0.0) if classifier_metrics else 0.0)
        reasons = []; flag = False
        if acc < threshold_acc:
            flag = True; reasons.append(f"Classifier accuracy {acc:.3f} < {threshold_acc}")

        # approx backtest metrics
        df_eval = df.copy().reset_index(drop=True)
        window = min(60, max(0, len(df_eval)-1))
        mean_ret = vol = sharpe = 0.0
        if window > 1:
            df_bt = df_eval.tail(window).copy()
            df_bt['NextRet'] = df_bt['Close'].pct_change().shift(-1)
            mean_ret = float(df_bt['NextRet'].mean() or 0.0)
            vol = float(df_bt['NextRet'].std() or 0.0)
            if vol > 0: sharpe = float((mean_ret/vol)*np.sqrt(252))
        self.log(f"Backtest approx mean_ret={mean_ret:.6f}, vol={vol:.6f}, sharpe={sharpe:.3f}")
        if sharpe < threshold_sharpe:
            flag = True; reasons.append(f"Sharpe {sharpe:.3f} < {threshold_sharpe}")

        # forecast interval check
        if isinstance(forecast_df, pd.DataFrame) and {'yhat_upper','yhat_lower','ds'}.issubset(set(forecast_df.columns)):
            try:
                fut = forecast_df[forecast_df['ds'] > df['Date'].max()]
                if not fut.empty:
                    avg_width = float((fut['yhat_upper'] - fut['yhat_lower']).mean())
                    rel = avg_width / (df['Close'].iloc[-1] + 1e-9)
                    self.log(f"Forecast relative width ~ {rel:.4f}")
                    if rel > 0.1:
                        flag = True; reasons.append("Forecast uncertainty high (>10% rel width)")
            except Exception as e:
                self.log(f"Forecast interval check failed: {e}")

        result = {"flag": flag, "reasons": reasons, "metrics": {"sharpe": sharpe, "mean_ret": mean_ret, "vol": vol, "accuracy": acc, "n_test": int(classifier_metrics.get("n_test",0) if classifier_metrics else 0)}}
        if self.memory is not None: self.memory.store("risk_eval", result)
        return result



# ---------------- ReportAgent ----------------
class ReportAgent(Agent):
    def generate(self, ticker, df_raw, df_proc, forecast_df, clf_pred, risk_eval, interactive: bool=False):
        self.log("Generating report & visualization...")
        prob = clf_pred.get("prob_ml", clf_pred.get("prob", 0.5))
        label = clf_pred.get("label", None)

        # compute forecast slope
        slope = 0.0
        try:
            if isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty:
                fut = forecast_df[forecast_df['ds'] > df_proc['Date'].max()]
                if not fut.empty:
                    y0 = float(fut['yhat'].iloc[0]); y1 = float(fut['yhat'].iloc[-1])
                    days = max(1, (fut['ds'].iloc[-1] - fut['ds'].iloc[0]).days + 1)
                    slope = (y1 - y0) / days
        except Exception:
            slope = 0.0

        # decision rule
        action = "HOLD"
        if prob is None: action = "HOLD"
        else:
            if prob > 0.6 and slope > 0: action = "BUY"
            elif prob < 0.4 and slope < 0: action = "SELL"
            else: action = "HOLD"

        # print summary
        print("\n===== DECISION REPORT =====")
        print(f"Ticker: {ticker}")
        print(f"ML prob: {prob:.3f}")
        print(f"Forecast slope: {slope:.5f}")
        print(f"Suggested action: {action}")
        print(f"Risk: {risk_eval}")
        print("===========================\n")

        # build plot
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3])
        fig.add_trace(go.Scatter(x=df_proc['Date'], y=df_proc['Close'], mode='lines', name='Historical Close'), row=1, col=1)
        # forecast part
        if isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty:
            fut = forecast_df[forecast_df['ds'] > df_proc['Date'].max()]
            if not fut.empty:
                fig.add_trace(go.Scatter(x=fut['ds'], y=fut['yhat'], mode='lines', name='Forecast (Prophet)', line=dict(dash='dash', color='orange')), row=1, col=1)
                if 'yhat_upper' in fut.columns and 'yhat_lower' in fut.columns:
                    fig.add_trace(go.Scatter(x=pd.concat([fut['ds'], fut['ds'][::-1]]),
                                             y=pd.concat([fut['yhat_upper'], fut['yhat_lower'][::-1]]),
                                             fill='toself', fillcolor='rgba(255,165,0,0.2)', line=dict(color='rgba(255,165,0,0)'),
                                             hoverinfo='skip', showlegend=True, name='Forecast Interval'), row=1, col=1)
        # short-term marker
        if label is not None:
            fig.add_trace(go.Scatter(x=[df_proc['Date'].iloc[-1]], y=[df_proc['Close'].iloc[-1]], mode='markers+text',
                                     marker=dict(size=12, color='green' if label==1 else 'red', symbol='triangle-up' if label==1 else 'triangle-down'),
                                     text=[f"Prob={prob:.2f}, Action={action}"], textposition='bottom right', name='Signal'), row=1, col=1)
        # vol subplot
        if 'Vol_20' in df_proc.columns:
            fig.add_trace(go.Scatter(x=df_proc['Date'], y=df_proc['Vol_20'], mode='lines', name='Vol_20'), row=2, col=1)
        fig.update_layout(title=f"{ticker} — Action: {action}", height=700)

        # LLM explanation (optional)
        llm_out = self.ask_llm_safe(ticker, prob, slope, action, risk_eval)
        print("LLM Explanation:", llm_out if isinstance(llm_out, str) else json.dumps(llm_out))

        if self.memory is not None:
            self.memory.store(f"{ticker}_report_summary", {"ticker": ticker, "prob": prob, "action": action, "risk": risk_eval})

        return {"summary": {"ticker":ticker,"prob":prob,"action":action,"risk":risk_eval}, "fig": fig, "llm": llm_out}

    def ask_llm_safe(self, ticker, prob, slope, action, risk, temperature=0.1, max_tokens=140):
        if not OPENROUTER_API_KEY:
            return {"error": "OPENROUTER_API_KEY not set"}
        prompt = (f"You are concise financial analyst. Stock={ticker}. "
                  f"Short-term rise probability={prob:.3f}. Forecast slope={slope:.5f}. Action={action}. Risk={risk}. "
                  "Give 2-3 short sentences explaining the drivers.")
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
        payload = {"model":"google/gemini-2.5-pro","temperature":float(temperature),"messages":[{"role":"user","content":prompt}],"max_tokens":max_tokens}
        try:
            r = requests.post(OPENROUTER_ENDPOINT, headers=headers, json=payload, timeout=20)
        except Exception as e:
            return {"error": f"request failed: {e}"}
        if r.status_code != 200:
            return {"error": f"status {r.status_code}", "raw": r.text[:800]}
        try:
            j = r.json()
        except Exception as e:
            return {"error": f"invalid json: {e}", "raw": r.text[:800]}
        # try to extract text safely
        choices = j.get("choices") or j.get("output") or []
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                if isinstance(first.get("message"), dict) and first["message"].get("content"):
                    return first["message"]["content"]
                if first.get("text"):
                    return first.get("text")
                if first.get("content"):
                    return first.get("content")
            return json.dumps(first)[:800]
        if isinstance(j.get("result"), str):
            return j.get("result")
        return {"error": "unexpected response shape", "raw": j}

# ---------------- Orchestrator ----------------
class Orchestrator:
    def __init__(self, agents: dict):
        self.agents = agents

    def run_for(self, ticker, period=DEFAULT_PERIOD, interval=DEFAULT_INTERVAL, horizon=HORIZON, fc_periods=FORECAST_DAYS, persist: bool=True, cutoff_date: str = None):
        self.agents['data'].log(f"Orchestrator starting for {ticker}")
        raw = self.agents['data'].fetch(ticker=ticker, period=period, interval=interval)
        if raw is None or raw.empty:
            self.agents['data'].log("No raw data; skipping")
            return

        try:
            proc = self.agents['pre'].preprocess(raw, ticker=ticker, cutoff_date=cutoff_date, persist=persist)
        except Exception as e:
            self.agents['pre'].log(f"Preprocess failed: {e}")
            return

        # forecast (Prophet)
        try:
            fc = self.agents['pred'].forecast_long_term(proc, periods=fc_periods)
            self.agents['data'].log(f"Forecast rows: {len(fc)} cols: {list(fc.columns)[:8]}")
        except Exception as e:
            self.agents['data'].log(f"Forecast failed: {e}")
            fc = pd.DataFrame()

        # train models
        clf_pred = {"prob_ml": 0.5, "label": None}
        clf_metrics = {}
        try:
            train_res = self.agents['pred'].train_models(proc, horizon=horizon)
            clf_pred = self.agents['pred'].predict_short_term(last_features=train_res.get("last_features"))
            # build classifier summary
            metrics_all = self.agents['pred'].memory.retrieve("model_metrics") or train_res.get("metrics", {})
            # choose best acc for reporting
            best_acc = 0.0; best_entry = {"n_test":0}
            for k,v in (metrics_all.items() if isinstance(metrics_all, dict) else []):
                acc_v = float(v.get("acc",0.0) or 0.0)
                if acc_v > best_acc:
                    best_acc = acc_v; best_entry = v
            clf_metrics = {"accuracy": best_acc, "n_test": int(best_entry.get("n_test",0) or 0)}
        except Exception as e:
            self.agents['pred'].log(f"Classifier training/prediction skipped due to: {e}")

        # attach forecast slope
        try:
            if isinstance(fc, pd.DataFrame) and not fc.empty:
                fut = fc[fc['ds'] > proc['Date'].max()]
                if not fut.empty:
                    y0 = float(fut['yhat'].iloc[0]); y1 = float(fut['yhat'].iloc[-1])
                    days = max(1,(fut['ds'].iloc[-1] - fut['ds'].iloc[0]).days + 1)
                    fc_slope = (y1 - y0) / days
                else:
                    fc_slope = 0.0
            else:
                fc_slope = 0.0
        except Exception:
            fc_slope = 0.0
        clf_pred["forecast_slope"] = fc_slope


        # risk
        try:
            risk = self.agents['risk'].evaluate(proc, classifier_metrics=clf_metrics, forecast_df=fc)
        except Exception as e:
            self.agents['risk'].log(f"Risk eval failed: {e}")
            risk = {"flag": True, "reasons":[str(e)], "metrics":{}}

        # after proc computed and before final return
        try:
            df_lab = self.agents['pred'].make_labels(proc, horizon=horizon)
            features = ['Return', 'MA_5', 'MA_10', 'MA_20', 'RSI_14', 'MACD_HIST', 'Vol_20', 'Mom_10']
            features = [c for c in features if c in df_lab.columns]
            back_df = self.agents['back'].walk_forward_test(df_lab, features, label_col='Label', horizon=horizon)
            if not back_df.empty:
                print("Walk-forward mean acc:", back_df['acc'].mean())
                back_df.to_csv(f"walkforward_{ticker}.csv", index=False)
        except Exception as e:
            self.agents['back'].log(f"Walk-forward failed: {e}")

        # report
        try:
            report = self.agents['report'].generate(ticker, raw, proc, fc, clf_pred, risk, interactive=False)
            if report.get("fig") is not None:
                # show once
                report["fig"].show()
        except Exception as e:
            self.agents['report'].log(f"Report generation failed: {e}")
            return

# ---------------- MAIN ----------------
def main():
    mem = Memory()
    agents = {
        "data": DataAgent("DataAgent", mem),
        "pre": PreprocessingAgent("PreprocessingAgent", mem),
        "pred": PredictionAgent("PredictionAgent", mem),
        "risk": RiskManagementAgent("RiskAgent", mem),
        "report": ReportAgent("ReportAgent", mem),
        "back": BacktesterAgent("Backtester", mem)
    }
    orch = Orchestrator(agents)

    # Example: you can specify cutoff_date to limit historic data for controlled experiments
    # e.g., cutoff_date = "2025-08-31"
    cutoff_date = "2025-08-31"
    # cutoff_date = "2025-08-31"

    TICKERS = ["MDLZ", "GIS", "UNH", "ABBV", "DUK", "SO", "T", "MSFT", "AAPL", "BRK-B"]  # or ["IBM","AAPL"]

    for t in TICKERS:
        orch.run_for(t, period="1y", interval="1d", horizon=HORIZON, fc_periods=FORECAST_DAYS, persist=True, cutoff_date=cutoff_date)
        time.sleep(1)

if __name__ == "__main__":
    main()
"""
2025-09-30: CL, MCD
2025-10-31: MCD
2025-08-31: WMT, IBM, T, ABBV
"""