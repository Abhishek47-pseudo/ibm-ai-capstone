# IBM AI Workflow Capstone Project

This repository is the IBM AI Workflow capstone project and contains the data investigation notebook, model comparison work, a Flask + Dash API, unit tests, Docker packaging, and a simple monitoring dashboard. The project is kept compatible with the original code structure while adding the missing peer-review checks, isolated test artifacts, and a single command test runner.

## Project structure

- `application/model.py` – training, loading, prediction, and model comparison
- `application/utils/ingestion.py` – automated data ingestion and time-series preparation
- `application/utils/logger.py` – production/test logging with CSV output
- `app.py` – Flask API and Dash app entry point
- `dashboards/` – prediction and monitoring pages
- `unittests/` – API, model, and logging tests
- `data/cs-train/` – training data input for the forecasting work
- `reports/figures/` – generated EDA and model comparison visualizations
- `Dockerfile` – production container definition

## EDA and model comparison

The notebooks `Part 1 - Data investigation.ipynb` and `Part 2 - Model building and selection.ipynb` contain the exploratory data analysis and model comparison work.

Relevant EDA visualizations include:
- time-series revenue trends,
- country-level comparison charts,
- data distribution plots,
- a baseline-vs-Prophet-vs-Random-Forest comparison chart saved at `reports/figures/model_comparison.png`.

The comparison uses actual evaluation metrics from the available data and stores MSE values for each model.

## Automated ingestion

The project exposes reusable ingestion helpers in `application/utils/ingestion.py`:
- `fetch_data(data_dir)`
- `fetch_ts(data_dir, clean=False)`
- `get_country_names(data_dir=None)`

To run ingestion manually:

```bash
python -c "from application.utils.ingestion import fetch_ts; fetch_ts('data/cs-train', clean=False)"
```

## Model behavior

The model layer trains a Prophet-compatible forecasting model for each country plus an all-countries series. It persists the trained artifacts in `models/` and logs training metrics in `logs/`.

Production artifacts remain separate from test artifacts:
- production: `prod-*` files and production log names
- test: `test-*` files and `*-test.log` logs

## Running the application

```bash
python app.py
```

The API is served on port `8050`.

## API examples

### Predict all countries

```bash
curl -X POST http://localhost:8050/predict \
  -H "Content-Type: application/json" \
  -d '{"mode":"test","country":"all","year":2018,"month":2,"n_next":20}'
```

### Predict one country

```bash
curl -X POST http://localhost:8050/predict \
  -H "Content-Type: application/json" \
  -d '{"mode":"test","country":"united_kingdom","year":2018,"month":2,"n_next":20}'
```

### Monitoring metrics

```bash
curl http://localhost:8050/metrics
```

## Peer Review Checklist

| Requirement | Implementation |
|---|---|
| API unit tests | Implemented in `unittests/api_tests.py` |
| Model unit tests | Implemented in `unittests/model_tests.py` |
| Logging unit tests | Implemented in `unittests/logger_tests.py` |
| Single test command | `python run-tests.py` |
| Performance monitoring | `GET /metrics` returns training metrics from persisted logs |
| Test isolation | `mode='test'` writes to separate model and log artifacts |
| Country-specific prediction | Supported via `/predict` with `country` set to a valid country name |
| All-country prediction | Supported via `/predict` with `country='all'` |
| Automated ingestion | `fetch_data()` and `fetch_ts()` in `application/utils/ingestion.py` |
| Multiple models | Baseline, Prophet-compatible, and Random Forest comparison included |
| EDA visualizations | Included in the notebooks and saved outputs |
| Docker | Dockerfile kept and validated for project structure |
| Baseline comparison | Saved chart at `reports/figures/model_comparison.png` |

## Running the tests

```bash
python run-tests.py
```

## Docker

```bash
docker build -t ibm-ai-capstone .
docker run -p 8050:8050 --rm ibm-ai-capstone
```

If Docker is unavailable in the current environment, the project still keeps the required Dockerfile and validates the packaging structure statically.

## Monitoring and logs

Logs are available under `logs/` and are also accessible via the API endpoints:
- `GET /logs/`
- `GET /logs/<filename>`
- `GET /metrics`

This allows a reviewer to inspect training and prediction behavior without overwriting production logs.

## Local validation commands

```bash
python -m compileall app.py application dashboards unittests
python run-tests.py
```

## Notes

This project intentionally keeps the original architecture of the IBM capstone while fixing the missing peer-review requirements: isolated testing, API validation, model artifact management, log access, and a single-run verification script.
