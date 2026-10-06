import os
from ast import literal_eval

import pandas as pd
from dash import dcc, html

from application.model import model_load, model_train
from application.utils.ingestion import get_country_names


# Layout utils
def Header(app):
    return html.Div([get_header(app), get_menu()], className='row')


def get_header(app):
    return html.Div(html.H1('AAVAIL'), className='six columns')


def get_menu():
    return html.Div(
        [
            html.Div(className='five columns'),
            html.Div(html.H4(dcc.Link('Prediction', href='/')), className='two columns'),
            html.Div(html.H4(dcc.Link('Monitor', href='/monitor/')), className='two columns'),
        ],
        className='row',
    )


def make_dash_table(df, header=False):
    """Return a Dash HTML table definition from a pandas DataFrame."""
    table = []
    if header:
        html_row = []
        for col in df.columns:
            html_row.append(html.Td([col]))
        table.append(html.Tr(html_row))
    for _, row in df.iterrows():
        html_row = []
        for value in row:
            html_row.append(html.Td([value]))
        table.append(html.Tr(html_row))
    return table


def read_metrics_from_log(logdir=None):
    if not logdir:
        logdir = os.path.join('.', 'logs')

    candidates = []
    for filename in sorted(os.listdir(logdir)):
        if not filename.endswith('.log') or not filename.startswith('train') or 'test' in filename:
            continue
        path = os.path.join(logdir, filename)
        try:
            df = pd.read_csv(path, delimiter=',', on_bad_lines='skip')
        except Exception:
            continue
        if df.empty or not {'timestamp', 'country', 'eval_metric'}.issubset(df.columns):
            continue
        candidates.append((filename, df))

    if not candidates:
        return pd.DataFrame(columns=['Country', 'RMSE'])

    filename, df = sorted(candidates, key=lambda item: item[0])[-1]
    df = df.sort_values(by='timestamp', ascending=True).groupby('country').tail(1)[['country', 'eval_metric']]
    df['RMSE'] = df['eval_metric'].apply(lambda x: literal_eval(x)['rmse'])
    df.drop(columns=['eval_metric'], inplace=True)
    df.rename(columns={'country': 'Country'}, inplace=True)
    return df


try:
    data, models = model_load()
except Exception:
    model_train()
    data, models = model_load()

country_mapping = {'all': 'All', **get_country_names()}
country_names = []
for key, country in country_mapping.items():
    if key in data.keys():
        country_names.append({'label': country, 'value': key})

metrics = read_metrics_from_log()
if not metrics.empty:
    metrics['Country'] = metrics['Country'].apply(lambda x: country_mapping.get(x, x)) 