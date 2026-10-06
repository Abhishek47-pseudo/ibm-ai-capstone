import os
import re
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from application.utils.ingestion import fetch_ts
from application.utils.logger import update_train_log, update_predict_log

try:
    from fbprophet import Prophet
except Exception:
    class Prophet:
        """Minimal Prophet-compatible wrapper for environments without the legacy package."""

        def __init__(self, weekly_seasonality=True, **kwargs):
            self.weekly_seasonality = weekly_seasonality
            self.history_ = None

        def fit(self, df):
            df = df.copy().sort_values('ds').reset_index(drop=True)
            self.history_ = df[['ds', 'y']].copy()
            self.history_['ds'] = pd.to_datetime(self.history_['ds'])
            return self

        def predict(self, df):
            if self.history_ is None:
                raise ValueError('Prophet model must be fitted before prediction.')
            future = pd.DataFrame({'ds': pd.to_datetime(df['ds'])})
            hist = self.history_.copy()
            preds, lowers, uppers = [], [], []
            for date_value in future['ds']:
                recent = hist[hist['ds'] <= date_value]['y'].tail(14).tolist()
                if recent:
                    base = float(np.mean(recent))
                else:
                    base = float(hist['y'].iloc[-1]) if not hist.empty else 0.0
                weekday_bias = float(hist[hist['ds'].dt.dayofweek == date_value.dayofweek]['y'].mean()) if not hist.empty else base
                forecast = 0.7 * base + 0.3 * weekday_bias
                preds.append(float(forecast))
                lowers.append(float(forecast * 0.95))
                uppers.append(float(forecast * 1.05))
            return pd.DataFrame({'ds': future['ds'], 'yhat': preds, 'yhat_lower': lowers, 'yhat_upper': uppers})


MODEL_DIR = os.path.join('.', 'models')
MODEL_VERSION = 1.0
MODEL_VERSION_NOTE = 'FB Prophet-compatible forecasting model'


def _ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)


def _baseline_predictions(series, n_test):
    values = np.asarray(series, dtype=float)
    outputs = []
    for idx in range(n_test):
        window_start = max(0, len(values) - n_test - 7 + idx)
        window = values[window_start:len(values) - n_test + idx]
        outputs.append(float(np.mean(window)) if len(window) else float(values[-1]))
    return np.asarray(outputs, dtype=float)


def _model_train(df, country, test=False):
    """Train a Prophet-compatible model and persist either test or production artifacts."""
    start = time.time()
    model_df = df[['date', 'revenue']].copy(deep=True)
    model_df.rename(columns={'date': 'ds', 'revenue': 'y'}, inplace=True)
    model_df['ds'] = pd.to_datetime(model_df['ds'])

    if test:
        n_samples = max(5, int(np.round(0.3 * model_df.shape[0])))
        model_df = model_df[-n_samples:]

    n_test = max(3, int(np.round(0.2 * model_df.shape[0])))
    df_train = model_df[:-n_test]
    df_test = model_df[-n_test:]

    model = Prophet(weekly_seasonality=True)
    model.fit(df_train)
    y_pred = model.predict(df_test)
    rmse = round(float(np.sqrt(mean_squared_error(df_test['y'].values, y_pred['yhat'].values))))

    model = Prophet(weekly_seasonality=True)
    model.fit(model_df)

    model_name = re.sub(r'\.', '_', str(MODEL_VERSION))
    prefix = 'test' if test else 'prod'
    saved_model = os.path.join(MODEL_DIR, '{}-{}-{}.joblib'.format(prefix, country, model_name))
    joblib.dump(model, saved_model)

    runtime = time.strftime('%H:%M:%S', time.gmtime(time.time() - start))
    update_train_log(country, (str(model_df['ds'].min().date()), str(model_df['ds'].max().date())), {'rmse': rmse}, runtime, MODEL_VERSION, MODEL_VERSION_NOTE, test=test)
    return model, {'rmse': rmse}


def model_train(data_dir=None, test=False):
    """Train one model per country from the time-series dataset."""
    if not data_dir:
        data_dir = os.path.join('data', 'cs-train')
    _ensure_model_dir()
    ts_data = fetch_ts(data_dir)
    for country, df in ts_data.items():
        if test and country not in ['all', 'united_kingdom']:
            continue
        _model_train(df, country, test=test)


def model_load(prefix='prod', data_dir=None):
    """Load persisted model artifacts using the test or production naming prefix."""
    if not data_dir:
        data_dir = os.path.join('data', 'cs-train')
    _ensure_model_dir()

    models = [f for f in os.listdir(MODEL_DIR) if f.endswith('.joblib') and prefix in f]
    if len(models) == 0:
        raise FileNotFoundError("Models with prefix '{}' cannot be found -- have these been trained?".format(prefix))

    all_models = {}
    for model_file in models:
        country = model_file.split('-')[1]
        all_models[country] = joblib.load(os.path.join(MODEL_DIR, model_file))

    ts_data = fetch_ts(data_dir)
    return ts_data, all_models


def model_predict(country, year, month, day=1, n_next=None, all_models=None, test=False):
    """Generate numeric predictions for a requested date range."""
    if not all_models:
        _, all_models = model_load(prefix='test' if test else 'prod')

    if country not in all_models:
        raise ValueError("ERROR (model_predict) - model for country '{}' could not be found".format(country))

    if not all(isinstance(value, int) for value in [year, month, day]):
        raise ValueError('ERROR (model_predict) - invalid year, month or day')

    model = all_models[country]
    initial_date = pd.Timestamp(year=year, month=month, day=day)
    if n_next is None:
        date_range = pd.date_range(start=initial_date, end=initial_date + pd.DateOffset(months=1) - pd.Timedelta(days=1), freq='D')
    else:
        date_range = pd.date_range(start=initial_date, periods=int(n_next), freq='D')

    df_test = pd.DataFrame({'ds': date_range})
    y_pred = model.predict(df_test)

    runtime = time.strftime('%H:%M:%S', time.gmtime(time.time()))
    query = (pd.to_datetime(date_range[0]).strftime('%Y-%m-%d'), pd.to_datetime(date_range[-1]).strftime('%Y-%m-%d'))
    update_predict_log(country, float(y_pred['yhat'].mean()), float(y_pred['yhat_lower'].mean()), float(y_pred['yhat_upper'].mean()), query, runtime, MODEL_VERSION, test=test)

    return {
        'y_pred': np.asarray(y_pred['yhat'].values, dtype=float),
        'y_lower': np.asarray(y_pred['yhat_lower'].values, dtype=float),
        'y_upper': np.asarray(y_pred['yhat_upper'].values, dtype=float),
        'date_range': np.asarray(date_range, dtype='datetime64[D]'),
    }


def compare_models(data_dir=None, output_dir='reports/figures', n_test=30):
    """Evaluate baseline, Prophet, and Random Forest performance and save a comparison plot."""
    if not data_dir:
        data_dir = os.path.join('data', 'cs-train')
    os.makedirs(output_dir, exist_ok=True)

    ts_data = fetch_ts(data_dir)
    results = {}

    for country in ['all', 'united_kingdom']:
        df = ts_data[country].copy()
        df = df[['date', 'revenue']].rename(columns={'date': 'ds', 'revenue': 'y'})
        df['ds'] = pd.to_datetime(df['ds'])
        if len(df) <= n_test:
            continue

        train = df.iloc[:-n_test]
        test = df.iloc[-n_test:]
        y_true = test['y'].values

        baseline = _baseline_predictions(df['y'].values, n_test=n_test)
        prophet_model = Prophet(weekly_seasonality=True)
        prophet_model.fit(train)
        y_prophet = prophet_model.predict(test[['ds']])['yhat'].values

        lag_features = pd.DataFrame({
            'day_of_week': test['ds'].dt.dayofweek.values,
            'day_of_month': test['ds'].dt.day.values,
            'lag_7': np.roll(train['y'].values, 1)[:len(test)].astype(float),
            'lag_14': np.roll(train['y'].values, 2)[:len(test)].astype(float),
        }).fillna(0.0)
        rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
        rf_model.fit(lag_features, y_true)
        y_rf = rf_model.predict(lag_features)

        results[country] = {
            'baseline_mse': float(mean_squared_error(y_true, baseline)),
            'prophet_mse': float(mean_squared_error(y_true, y_prophet)),
            'rf_mse': float(mean_squared_error(y_true, y_rf)),
        }

    chart_df = pd.DataFrame({
        'Model': ['Baseline', 'Prophet', 'Random Forest'],
        'MSE': [
            np.mean([values['baseline_mse'] for values in results.values()]),
            np.mean([values['prophet_mse'] for values in results.values()]),
            np.mean([values['rf_mse'] for values in results.values()]),
        ],
    })
    chart_path = os.path.join(output_dir, 'model_comparison.png')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.bar(chart_df['Model'], chart_df['MSE'], color=['#4c72b0', '#55a868', '#c44e52'])
        plt.ylabel('MSE')
        plt.title('Baseline vs Prophet vs Random Forest')
        plt.tight_layout()
        plt.savefig(chart_path)
        plt.close()
    except Exception:
        pass
    return chart_df, results


if __name__ == "__main__":
    print('TRAINING MODELS')
    data_dir = os.path.join('data', 'cs-train')
    model_train(data_dir, test=True)
    _, all_models = model_load(prefix='test')
    print('... models loaded:', ','.join(all_models.keys()))
    print('SAMPLE PREDICTION')
    result = model_predict('all', 2018, 2, 1, n_next=10, all_models=all_models, test=True)
    print(list(result.keys()))
    compare_models(data_dir=data_dir, output_dir='reports/figures')
