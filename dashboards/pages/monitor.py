from dash import html

from dashboards.utils.utils import Header, make_dash_table, metrics


def create_layout(app):
    return html.Div(
        [
            html.Div([Header(app)]),
            html.Div([
                html.H3('Monitor'),
                html.H5('Last logged evaluation metric'),
                html.Div([
                    html.Table(make_dash_table(metrics, header=True)),
                ]),
            ]),
            html.Div(id='intermediate-value', style={'display': 'none'}),
        ]
    )

