from dash import dcc, html

from dashboards.utils.utils import Header, country_mapping, country_names


def create_layout(app):
    return html.Div(
        [
            html.Div([Header(app)]),
            html.Div([
                html.H3('Prediction'),
                html.Div([
                    html.Div([
                        html.Div('Select country', className='three columns'),
                    ], className='row'),
                    dcc.Dropdown(
                        options=country_names,
                        value='all',
                        id='country-dropdown',
                        clearable=False,
                    ),
                ]),
                html.Br(),
                html.Div([
                    html.Div(html.H5('45-days prediction', id='pred-country-message')),
                    dcc.Graph(id='pred-plot'),
                ]),
            ]),
            html.Div(id='intermediate-value', style={'display': 'none'}),
        ]
    )

