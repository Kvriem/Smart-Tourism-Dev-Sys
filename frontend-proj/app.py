# app.py

from dash import Dash, html, dcc, Input, Output, callback, callback_context
import dash
import dash_bootstrap_components as dbc
from pages import page1
import pandas as pd

# Load shared data
df = page1.load_data()

# Initialize Dash app
app = Dash(
    __name__,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
        dbc.themes.LUX,
        "https://use.fontawesome.com/releases/v5.15.4/css/all.css",
        "https://cdn.jsdelivr.net/npm/daterangepicker/daterangepicker.css"
    ],
    external_scripts=[
        "https://cdn.jsdelivr.net/npm/moment@2.29.4/moment.min.js",
        "https://cdn.jsdelivr.net/npm/daterangepicker/daterangepicker.min.js"
    ],
    suppress_callback_exceptions=True
)

# Add custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Hotel Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            :root {
                --primary-color: #4361ee;
                --secondary-color: #3a0ca3;
                --accent-color: #4cc9f0;
                --success-color: #4bb543;
                --warning-color: #ff9e00;
                --danger-color: #ef233c;
                --light-color: #f8f9fa;
                --dark-color: #212529;
            }
            
            body {
                font-family: 'Inter', system-ui, sans-serif;
                background-color: var(--light-color);
            }
            
            .sidebar {
                background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
                color: white;
                min-height: 100vh;
                width: 280px;
                transition: all 0.3s ease;
                position: fixed;
                top: 0;
                left: 0;
                z-index: 1000;
                padding: 1rem;
            }
            
            .sidebar.collapsed {
                width: 80px;
            }
            
            .sidebar-title {
                font-size: 1.5rem;
                font-weight: 600;
                margin-bottom: 2rem;
                white-space: nowrap;
                overflow: hidden;
            }
            
            .sidebar .nav-link {
                color: rgba(255, 255, 255, 0.8);
                border-radius: 8px;
                margin: 0.5rem 0;
                transition: all 0.2s ease;
                padding: 0.75rem 1rem;
            }
            
            .sidebar .nav-link:hover {
                background: rgba(255, 255, 255, 0.1);
                color: white;
            }
            
            .sidebar .nav-link.active {
                background: rgba(255, 255, 255, 0.2);
                color: white;
            }
            
            .toggle-btn {
                background: white !important;
                border-radius: 50% !important;
                width: 24px;
                height: 24px;
                padding: 0 !important;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            }
            
            .content-area {
                margin-left: 280px;
                transition: all 0.3s ease;
                min-height: 100vh;
                background: var(--light-color);
            }
            
            .content-area.collapsed {
                margin-left: 80px;
            }
            
            .nav-section {
                margin-top: 2rem;
            }
            
            .nav-section-title {
                color: rgba(255, 255, 255, 0.6);
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin: 1rem 0 0.5rem;
                padding-left: 1rem;
            }
            
            @media (max-width: 768px) {
                .sidebar {
                    width: 80px;
                }
                .content-area {
                    margin-left: 80px;
                }
                .sidebar-title span {
                    display: none;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Sidebar with enhanced navigation
sidebar = html.Div([
    html.Div([
        html.Div([
            html.H2([
                html.I(className="fas fa-hotel me-2"),
                html.Span("Analytics Hub", className="sidebar-title-text")
            ], className="sidebar-title"),
            html.Button(
                html.I(className="fas fa-angle-left", id="sidebar-toggle-icon"),
                className="btn btn-sm btn-light position-absolute toggle-btn",
                style={"right": "-12px", "top": "50%", "transform": "translateY(-50%)"},
                id="sidebar-toggle"
            )
        ], className="position-relative")
    ], className="sidebar-header"),
    
    # Enhanced Navigation menu
    html.Div([
        html.H6("MAIN MENU", className="nav-section-title"),
        dbc.Nav([
            dbc.NavItem(dbc.NavLink(
                [html.I(className="fas fa-tachometer-alt me-2"), 
                 html.Span("Dashboard", className="nav-text")],
                href="/",
                active="exact",
                className="nav-link-custom"
            )),
            dbc.NavItem(dbc.NavLink(
                [html.I(className="fas fa-chart-line me-2"),
                 html.Span("Analytics", className="nav-text")],
                href="/analytics",
                className="nav-link-custom"
            )),
        ], vertical=True, pills=True, className="mb-4"),
        
        html.H6("INSIGHTS", className="nav-section-title"),
        dbc.Nav([
            dbc.NavItem(dbc.NavLink(
                [html.I(className="fas fa-star me-2"),
                 html.Span("Reviews", className="nav-text")],
                href="/reviews",
                className="nav-link-custom"
            )),
            dbc.NavItem(dbc.NavLink(
                [html.I(className="fas fa-users me-2"),
                 html.Span("Customers", className="nav-text")],
                href="/customers",
                className="nav-link-custom"
            )),
        ], vertical=True, pills=True),
    ], className="sidebar-nav"),
    
    # Footer section in sidebar
    html.Div([
        html.Hr(className="sidebar-divider"),
        dbc.Nav([
            dbc.NavItem(dbc.NavLink(
                [html.I(className="fas fa-cog me-2"),
                 html.Span("Settings", className="nav-text")],
                href="/settings",
                className="nav-link-custom"
            )),
        ], vertical=True, pills=True),
    ], className="sidebar-footer")
], id="sidebar", className="sidebar")

# Content area with modal
content = html.Div([
    dcc.Location(id='url', refresh=False),
    sidebar,
    html.Div(id="page-content", className="content-area"),
    html.Div(id='modal-container', children=[
        dbc.Modal([
            dbc.ModalHeader(html.H5("📊 Chart Insights", className="fw-bold")),
            dbc.ModalBody(html.Div(id="modal-body-content")),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-modal", className="ms-auto")
            )
        ], id="chart-modal", size="lg", centered=True, is_open=False)
    ]),
    dcc.Store(id='sidebar-state', data='expanded')
])

# App layout
app.layout = content

# Route callback
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == "/" or pathname == "/sentiment":
        return page1.layout(df)
    elif pathname in ["/analytics", "/reviews", "/customers", "/settings"]:
        return html.H3("This page is under development", className="p-4")
    else:
        return html.H3("404 Page Not Found", className="p-4")

# Sidebar toggle callback
@app.callback(
    [Output('sidebar', 'className'),
     Output('page-content', 'style'),
     Output('sidebar-state', 'data'),
     Output('sidebar-toggle-icon', 'className')],
    [Input('sidebar-toggle', 'n_clicks')],
    [Input('sidebar-state', 'data')]
)
def toggle_sidebar(n_clicks, state):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "sidebar", {"marginLeft": "280px"}, "expanded", "fas fa-angle-left"
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if triggered_id == 'sidebar-toggle':
        new_state = 'collapsed' if state == 'expanded' else 'expanded'
        margin_left = "80px" if new_state == 'collapsed' else "280px"
        icon_class = 'fas fa-angle-right' if new_state == 'collapsed' else 'fas fa-angle-left'
        return f"sidebar {new_state}", {"marginLeft": margin_left}, new_state, icon_class
    
    margin_left = "80px" if state == 'collapsed' else "280px"
    return f"sidebar {state}", {"marginLeft": margin_left}, state, "fas fa-angle-left"

if __name__ == '__main__':
    app.run(debug=True)