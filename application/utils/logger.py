import csv
import os
import time
import uuid
from datetime import date

import pandas as pd


def _ensure_log_dir():
    os.makedirs(os.path.join('.', 'logs'), exist_ok=True)
    return os.path.join('.', 'logs')


def update_train_log(country, data_range, eval_metric, runtime, MODEL_VERSION, MODEL_VERSION_NOTE, test=False):
    """Write a CSV row for a model training event."""
    _ensure_log_dir()
    today = date.today()
    logfile = os.path.join('logs', 'train-test.log' if test else 'train-{}-{}.log'.format(today.year, today.month))
    header = ['timestamp', 'unique_id', 'country', 'data_range', 'eval_metric', 'model_version', 'model_version_note', 'runtime']
    write_header = not os.path.exists(logfile)

    with open(logfile, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)
        writer.writerow([time.time(), str(uuid.uuid4()), country, str(data_range), str(eval_metric), MODEL_VERSION, MODEL_VERSION_NOTE, runtime])


def update_predict_log(country, y_pred, y_lower, y_upper, query, runtime, MODEL_VERSION, test=False):
    """Write a CSV row for a prediction event."""
    _ensure_log_dir()
    today = date.today()
    logfile = os.path.join('logs', 'predict-test.log' if test else 'predict-{}-{}.log'.format(today.year, today.month))
    header = ['timestamp', 'unique_id', 'country', 'y_pred', 'y_lower', 'y_upper', 'query', 'model_version', 'runtime']
    write_header = not os.path.exists(logfile)

    with open(logfile, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)
        writer.writerow([time.time(), str(uuid.uuid4()), country, y_pred, y_lower, y_upper, str(query), MODEL_VERSION, runtime])


def read_training_metrics(log_dir=None):
    """Return the most recent training metrics as a list of dictionaries."""
    if not log_dir:
        log_dir = _ensure_log_dir()
    files = sorted([os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.startswith('train-') and f.endswith('.log')])
    records = []
    for path in files:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if df.empty or 'eval_metric' not in df.columns:
            continue
        for _, row in df.iterrows():
            try:
                metric = {'country': row.get('country', 'unknown'), 'rmse': float(eval(row['eval_metric'], {'__builtins__': {}}, {}))} 
            except Exception:
                metric = {'country': row.get('country', 'unknown'), 'rmse': None}
            records.append(metric)
    return records


if __name__ == "__main__":
    from application.model import MODEL_VERSION, MODEL_VERSION_NOTE
    update_train_log('Brazil', "('2000-01-01', '2000-01-01')", {'rmse': 4.2}, '00:00:01', MODEL_VERSION, MODEL_VERSION_NOTE, test=True)
    update_predict_log('Brazil', 1, -1, 3, "('2000-01-01', '2000-01-01')", '00:00:01', MODEL_VERSION, test=True)