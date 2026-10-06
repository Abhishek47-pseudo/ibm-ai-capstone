import json

import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output, callback, dcc, html
import dash

from dashboards.pages import monitor, prediction
from dashboards.utils.utils import data, models
from application.model import model_predict

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


def create_dash_app(server):
    app = dash.Dash(
        __name__,
        server=server,
        routes_pathname_prefix='/',
        external_stylesheets=external_stylesheets,
        suppress_callback_exceptions=True,
    )

    app.layout = html.Div(
        [dcc.Location(id='url', refresh=False), html.Div(id='page-content')],
        className='container',
    )

    @app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
    def display_page(pathname):
        if pathname in ['/monitor/', '/monitor']:
            return monitor.create_layout(app)
        return prediction.create_layout(app)

    @app.callback(Output('pred-country-message', 'children'), Input('country-dropdown', 'value'))
    def update_output(value):
        result = '45 days prediction for '
        if value:
            result += prediction.country_mapping[value]
        return '{}'.format(result)

    @app.callback(Output('intermediate-value', 'children'), Input('country-dropdown', 'value'))
    def save_data(value):
        df = data[value]
        return json.dumps((value, df.to_json(date_format='iso', orient='split')))

    @app.callback(Output('pred-plot', 'figure'), Input('intermediate-value', 'children'))
    def update_graph(data_json):
        country, country_data = json.loads(data_json)
        dff = pd.read_json(country_data, orient='split')

        last_train_date = pd.to_datetime(dff.date.values[-1])
        pred_data = model_predict(
            country,
            last_train_date.year,
            last_train_date.month,
            last_train_date.day,
            all_models=models,
            n_next=45,
        )

        fig = go.Figure([
            go.Scatter(x=dff['date'], y=dff['revenue'], name='True values'),
            go.Scatter(x=pred_data['date_range'], y=pred_data['y_lower'], fill=None, line_color='gray', name='Prediction lower limit'),
            go.Scatter(x=pred_data['date_range'], y=pred_data['y_upper'], fill='tonexty', line_color='gray', name='Prediction upper limit'),
            go.Scatter(x=pred_data['date_range'], y=pred_data['y_pred'], name='Prediction'),
        ])
        fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation='h'))
        return fig

    return app