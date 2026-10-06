from flask import Flask, jsonify, request, send_from_directory
import os
import re
import numpy as np

from dashboards import create_dash_app

from application.model import model_train, model_load, model_predict
from application.utils.logger import read_training_metrics

server = Flask(__name__)
app = create_dash_app(server)


def _get_request_payload():
    payload = request.get_json(silent=True)
    if payload is None and request.method in {'GET', 'POST'}:
        payload = request.args.to_dict(flat=True)
    if payload is None:
        payload = {}
    return payload


@server.route('/predict', methods=['GET', 'POST'])
def predict():
    """Return predictions for the requested country or for all countries combined."""
    payload = _get_request_payload()
    query = payload.get('query', payload)
    mode = payload.get('mode', query.get('mode', 'prod')) if isinstance(query, dict) else 'prod'
    test = mode == 'test'

    if not isinstance(query, dict):
        return jsonify({'status': 'error', 'error': 'Request must contain a query object'}), 400

    if 'country' not in query:
        return jsonify({'status': 'error', 'error': 'country is required'}), 400

    country = str(query['country'])
    year = int(query.get('year', 2018))
    month = int(query.get('month', 1))
    day = int(query.get('day', 1))
    n_next = int(query.get('n_next', 30))

    try:
        _, models = model_load(prefix='test' if test else 'prod')
        result = model_predict(country=country, year=year, month=month, day=day,
                              n_next=n_next, all_models=models, test=test)
    except Exception as exc:
        return jsonify({'status': 'error', 'error': str(exc)}), 400

    serialized = {}
    for key, item in result.items():
        if isinstance(item, np.ndarray):
            serialized[key] = item.tolist()
        else:
            serialized[key] = item

    serialized['country'] = country
    serialized['status'] = 'ok'
    return jsonify(serialized)


@server.route('/train', methods=['GET', 'POST'])
def train():
    """Train model artifacts, with isolated test artifacts when mode='test'."""
    payload = _get_request_payload()
    test = bool(payload.get('mode') == 'test')
    model_train(test=test)
    return jsonify(True)


@server.route('/logs/', methods=['GET'])
def list_logs():
    """List available log files."""
    log_dir = os.path.join('.', 'logs')
    if not os.path.isdir(log_dir):
        return jsonify([])
    return jsonify(sorted(os.listdir(log_dir)))


@server.route('/logs/<filename>', methods=['GET'])
def logs(filename):
    """Download a log file by name."""
    if not re.search(r'\.log$', filename, flags=re.IGNORECASE):
        return jsonify({'status': 'error', 'error': 'Requested file is not a log file'}), 400

    log_dir = os.path.join('.', 'logs')
    if not os.path.isdir(log_dir):
        return jsonify({'status': 'error', 'error': 'Logs directory is missing'}), 404

    file_path = os.path.join(log_dir, filename)
    if not os.path.exists(file_path):
        return jsonify({'status': 'error', 'error': 'Log file not found'}), 404

    return send_from_directory(log_dir, filename, as_attachment=True, mimetype='text/csv')


@server.route('/metrics', methods=['GET'])
def metrics():
    """Expose actual training metrics from the persisted training logs."""
    try:
        summary = read_training_metrics(log_dir=os.path.join('.', 'logs'))
        return jsonify({'status': 'ok', 'metrics': summary})
    except Exception:
        return jsonify({'status': 'ok', 'metrics': []})


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=False)