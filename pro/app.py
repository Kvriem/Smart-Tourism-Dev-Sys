import dash
from dash import dcc, html, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
from datetime import datetime
import threading
import time
# Import page components with better formatting
from pages.overview_page import (
    create_overview_content,
    analyze_word_details,
    create_modal_content,
    analyze_nationality_details,
    create_nationality_modal_content
)
from pages.Hotels_page import create_hotels_content
from pages.recommendations_page import create_recommendations_content
from pages.geography_page import layout as geography_layout
from cache_utils import clear_cache_metadata
from performance_cache import performance_cache
from data_processing_optimized import get_cities_fast, get_basic_stats_fast

# --- App Setup ---
app = dash.Dash(__name__,                external_stylesheets=[
                    dbc.themes.BOOTSTRAP,
                    'assets/app.css',
                    'assets/overview.css',
                    'assets/style.css',
                    'assets/data-status.css',
                    'assets/modal-enhanced.css',                    'assets/modern-charts.css',
                    'assets/enhanced-sidebar.css',
                    'assets/recommendations.css',
                    'assets/geography.css',
                    'assets/geography-enhanced.css'
                ],
                external_scripts=[
                    'assets/dashboard.js'
                ],
                suppress_callback_exceptions=True)

app.title = "Dashboard"

# --- Sidebar Layout ---
sidebar = html.Div([
    # Logo/Branding Section
    html.Div([
        html.H2([
            html.Span("🎯", className="emoji-icon"),
            "Analytics Hub"
        ], className="sidebar-brand"),
        html.Hr(className="sidebar-divider"),
        # Status indicator
        html.Div([
            html.Span([
                html.I(className="fas fa-circle", style={"color": "#10b981", "fontSize": "0.8rem", "marginRight": "0.5rem"}),
                "System Online"
            ], style={"color": "#94a3b8", "fontSize": "0.875rem", "textAlign": "center"})
        ], style={"textAlign": "center", "marginBottom": "1rem"})
    ], className="sidebar-header"),
    
    # Navigation Menu
    html.Nav([
        html.Ul([
            html.Li([
                html.A([
                    html.Span("📊", className="nav-emoji"),
                    html.Span("Overview", className="nav-text"),
                    html.Span("", className="nav-indicator")
                ], href="#", className="nav-link active", id="overview-link")
            ], className="nav-item"),            
            html.Li([
                html.A([
                    html.Span("🏨", className="nav-emoji"),
                    html.Span("Hotels", className="nav-text"),
                    html.Span("", className="nav-indicator")
                ], href="#", className="nav-link", id="hotels-link")
            ], className="nav-item"),            html.Li([
                html.A([
                    html.Span("💡", className="nav-emoji"),
                    html.Span("Recommendations", className="nav-text"),
                    html.Span("", className="nav-indicator")
                ], href="#", className="nav-link", id="recommendations-link")
            ], className="nav-item"),
            
            html.Li([
                html.A([
                    html.Span("🌍", className="nav-emoji"),
                    html.Span("Geography", className="nav-text"),
                    html.Span("", className="nav-indicator")
                ], href="#", className="nav-link", id="geography-link")
            ], className="nav-item"),
        ], className="nav-menu")
    ], className="sidebar-nav"),
    
    # Sidebar Footer with enhanced info
    html.Div([
        html.Hr(className="sidebar-divider"),
        html.Div([
            html.P([
                html.I(className="fas fa-code", style={"marginRight": "0.5rem"}),
                "© 2025 Dashboard"
            ], className="sidebar-footer-text"),
            html.P([
                html.I(className="fas fa-chart-line", style={"marginRight": "0.5rem", "color": "#3b82f6"}),
                "v2.1.0"
            ], style={"color": "#64748b", "fontSize": "0.75rem", "textAlign": "center", "margin": "0.5rem 0 0 0"})
        ])
    ], className="sidebar-footer")
    
], className="sidebar")

# --- Filters Section (Global for all pages) ---
filters_section = html.Div([
    # Enhanced Page Title Section
    html.Div([
        html.Div([
            html.H1([
                html.Span("🚀", className="title-emoji"),
                "Tourism Analytics Dashboard"
            ], className="page-title enhanced-title")
        ], className="title-content")
    ], className="enhanced-content-header"),
    
    # Filters Header
    html.Div([
        html.H5([
            html.Span("🔍", className="filter-emoji"),
            "Smart Filters"
        ], className="filters-title"),
        
        # Data status indicator
        html.Div([
            html.Span(id="data-status-indicator", className="data-status-badge"),
        ], className="data-status-container")
    ], className="filters-header"),
    
    # Filters Content - Single Row
    html.Div([
        dbc.Row([
            # Location Filter
            dbc.Col([
                html.Label([
                    html.Span("📍", className="label-emoji"),
                    "Location"
                ], className="filter-label"),
                dcc.Dropdown(
                    id="city-dropdown",
                    options=[{'label': 'All Cities', 'value': 'all'}, {'label': 'Loading...', 'value': 'loading'}],
                    value="all",
                    className="enhanced-filter-dropdown",
                    placeholder="Select city"
                )
            ], lg=3, md=6, sm=12, className="filter-col"),
            
            # Time Period Filter
            dbc.Col([
                html.Label([
                    html.Span("⏰", className="label-emoji"),
                    "Time Period"
                ], className="filter-label"),
                dcc.RadioItems(
                    id="date-range-option",
                    options=[
                        {"label": "All Time", "value": "all_time"},
                        {"label": "Custom", "value": "custom"}
                    ],
                    value="all_time",
                    className="enhanced-radio-inline",
                    inline=True
                )
            ], lg=2, md=6, sm=12, className="filter-col"),
            
            # Start Date Filter
            dbc.Col([
                html.Label([
                    html.Span("📅", className="label-emoji"),
                    "Start Date"
                ], className="filter-label"),
                dcc.DatePickerSingle(
                    id="start-date-picker",
                    placeholder="Start date",
                    className="enhanced-date-input",
                    display_format="MMM DD, YYYY"
                )
            ], lg=2, md=6, sm=12, className="filter-col date-picker-disabled", id="start-date-col"),
            
            # End Date Filter
            dbc.Col([
                html.Label([
                    html.Span("📆", className="label-emoji"),
                    "End Date"
                ], className="filter-label"),
                dcc.DatePickerSingle(
                    id="end-date-picker",
                    placeholder="End date",
                    className="enhanced-date-input",
                    display_format="MMM DD, YYYY"
                )
            ], lg=2, md=6, sm=12, className="filter-col date-picker-disabled", id="end-date-col"),
            
            dbc.Col([
                html.Div([
                    dbc.Button([
                        html.I(className="fas fa-undo me-1"),
                        "Reset"
                    ], color="outline-secondary", size="sm", className="action-btn reset-btn me-2", id="reset-filters-btn"),
                    dbc.Button([
                        html.I(className="fas fa-sync me-1", id="refresh-icon"),
                        "Refresh Data"
                    ], color="primary", size="sm", className="action-btn refresh-btn", id="refresh-data-btn")
                ], className="filter-actions-inline")
            ], lg=3, md=12, sm=12, className="filter-col actions-col"),
            
        ], className="filters-row", align="end")
    ], className="filters-content")
], className="filters-section")

# --- Main Content Layout ---
main_content = html.Div([
    # Global Filters Section
    filters_section,
    # Content Body - Overview Page Content
    html.Div([
        html.Div(id="page-content")
    ], className="content-body")
    
], className="main-content")

# --- App Layout ---
app.layout = html.Div([
    dcc.Store(id='current-page', data='overview'),  # Store for current page
    dcc.Store(id='app-data', data={}),  # Store for shared data to avoid redundant loading
    dcc.Store(id='refresh-trigger', data=0),  # Store to trigger data refresh
    sidebar,
    main_content,
    
    # Toast notification for refresh feedback
    dbc.Toast(
        id="refresh-toast",
        header="Data Refresh",
        is_open=False,
        dismissable=True,
        duration=4000,
        icon="success",
        style={"position": "fixed", "top": 10, "right": 10, "width": 350, "z-index": 1999}
    )
], className="app-container")

# --- Callbacks ---

# Callback to load data once and store it
@app.callback(
    Output('app-data', 'data'),
    [Input('city-dropdown', 'id'),  # Trigger on page load
     Input('refresh-trigger', 'data')],  # Trigger on refresh button click
    prevent_initial_call=False
)
def load_app_data(_, refresh_count):
    """Load data once and store it for all pages to use with optimized performance"""
    try:
        start_time = time.time()
        
        # Check if this is a refresh trigger
        if refresh_count > 0:
            print(f"🔄 Data refresh triggered (count: {refresh_count})")
            # Clear performance cache on explicit refresh
            performance_cache.clear_all()
        
        # Use optimized data loading
        cities = get_cities_fast()
        basic_stats = get_basic_stats_fast()
        
        data_dict = {
            "status": "success",
            "last_updated": datetime.now().isoformat(),
            "record_count": basic_stats['total_reviews'],
            "cities": cities,
            "total_hotels": basic_stats['total_hotels'],
            "total_cities": basic_stats['total_cities'],
            "satisfaction_rate": basic_stats['satisfaction_rate']
        }
        
        elapsed = time.time() - start_time
        print(f"⚡ App data loaded in {elapsed:.2f} seconds")
        
        return data_dict
        
    except Exception as e:
        print(f"❌ Error loading app data: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"Error loading data: {str(e)}"}

# Callback to populate city dropdown
@app.callback(
    Output('city-dropdown', 'options'),
    [Input('app-data', 'data')],
    prevent_initial_call=False
)
def populate_city_dropdown(app_data):
    """Populate city dropdown from loaded data"""
    try:
        if not app_data or app_data.get('status') != 'success':
            return [{'label': 'All Cities', 'value': 'all'}, {'label': 'No Data Available', 'value': 'error'}]
        
        cities = app_data.get('cities', [])
        options = [{'label': 'All Cities', 'value': 'all'}]
        options.extend([{'label': city, 'value': city} for city in cities])
        
        return options
    except Exception as e:
        print(f"Error populating city dropdown: {e}")
        return [{'label': 'All Cities', 'value': 'all'}, {'label': 'Error Loading Cities', 'value': 'error'}]

# Navigation callback to handle page switching
@app.callback(
    [Output('current-page', 'data'),
     Output('overview-link', 'className'),
     Output('hotels-link', 'className'),
     Output('recommendations-link', 'className'),
     Output('geography-link', 'className')],
    [Input('overview-link', 'n_clicks'),
     Input('hotels-link', 'n_clicks'),
     Input('recommendations-link', 'n_clicks'),
     Input('geography-link', 'n_clicks')],
    prevent_initial_call=True
)
def handle_navigation(overview_clicks, hotels_clicks, recommendations_clicks, geography_clicks):
    """Handle navigation between pages"""
    ctx_triggered = ctx.triggered
    if not ctx_triggered:
        return 'overview', 'nav-link active', 'nav-link', 'nav-link', 'nav-link'
    
    button_id = ctx_triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'overview-link':
        return 'overview', 'nav-link active', 'nav-link', 'nav-link', 'nav-link'
    elif button_id == 'hotels-link':
        return 'hotels', 'nav-link', 'nav-link active', 'nav-link', 'nav-link'
    elif button_id == 'recommendations-link':
        return 'recommendations', 'nav-link', 'nav-link', 'nav-link active', 'nav-link'
    elif button_id == 'geography-link':
        return 'geography', 'nav-link', 'nav-link', 'nav-link', 'nav-link active'
    
    return 'overview', 'nav-link active', 'nav-link', 'nav-link', 'nav-link'

# Callback to show/hide date pickers based on date range option
@app.callback(
    [Output('start-date-col', 'style'),
     Output('end-date-col', 'style'),
     Output('start-date-col', 'className'),
     Output('end-date-col', 'className')],
    [Input('date-range-option', 'value')]
)
def toggle_date_pickers(date_range_option):
    if date_range_option == 'all_time':
        # Show date pickers with disabled state when "All Time" is selected
        disabled_style = {'display': 'block', 'opacity': '0.5', 'pointer-events': 'none'}
        disabled_class = 'filter-col date-picker-disabled'
        return disabled_style, disabled_style, disabled_class, disabled_class
    else:
        # Enable date pickers when "Custom" is selected
        enabled_style = {'display': 'block', 'opacity': '1', 'pointer-events': 'auto'}
        enabled_class = 'filter-col'
        return enabled_style, enabled_style, enabled_class, enabled_class

# Callback to reset filters
@app.callback(
    [Output('city-dropdown', 'value'),
     Output('start-date-picker', 'date'),
     Output('end-date-picker', 'date'),
     Output('date-range-option', 'value')],
    [Input('reset-filters-btn', 'n_clicks')],
    prevent_initial_call=True
)
def reset_filters(reset_clicks):
    if reset_clicks:
        return "all", None, None, "all_time"
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update

# Callback to update page content based on filters and current page
@app.callback(
    Output('page-content', 'children'),
    [Input('current-page', 'data'),
     Input('city-dropdown', 'value'),
     Input('start-date-picker', 'date'),
     Input('end-date-picker', 'date'),
     Input('date-range-option', 'value'),
     Input('reset-filters-btn', 'n_clicks'),
     Input('app-data', 'data'),
     Input('refresh-trigger', 'data')],  # Add refresh trigger dependency
    prevent_initial_call=False
)
def update_page_content(current_page, city_filter, start_date, end_date, date_range_option, reset_clicks, app_data, refresh_count):
    """Update page content based on current page and filters with optimized performance"""
    import time
    start_time = time.time()
    
    # Debug print for monitoring
    print(f"🔄 Page switching: {current_page}, city: {city_filter}")
    
    # Reset filters if reset button was clicked
    ctx_triggered = ctx.triggered
    if ctx_triggered and ctx_triggered[0]['prop_id'] == 'reset-filters-btn.n_clicks':
        city_filter = "all"
        start_date = None
        end_date = None
    
    # Only use date filters if custom date range is selected
    if date_range_option == 'all_time':
        start_date = None
        end_date = None      # Return appropriate page content based on current page with error handling
    try:
        if current_page == 'hotels':
            content = create_hotels_content(city_filter, start_date, end_date)
        elif current_page == 'recommendations':
            content = create_recommendations_content(city_filter, start_date, end_date)
        elif current_page == 'geography':
            content = geography_layout()
        else:  # Default to overview - most common case first
            content = create_overview_content(city_filter, start_date, end_date)
        
        elapsed = time.time() - start_time
        print(f"✅ Page content updated in {elapsed:.3f}s")
        return content
        
    except Exception as e:
        print(f"❌ Error updating page content: {e}")
        # Return error fallback
        return html.Div([
            html.Div([
                html.I(className="fas fa-exclamation-triangle fa-3x text-warning mb-3"),
                html.H4("Page Loading Error", className="text-warning"),
                html.P(f"Error: {str(e)}", className="text-muted"),
                html.P("Please try refreshing the page.", className="text-muted")
            ], className="text-center p-5")
        ])

# Callback for word frequency chart click events and modal handling
@app.callback(
    [Output('word-analysis-modal', 'is_open'),
     Output('word-analysis-modal', 'children')],
    [Input('word-freq-chart', 'clickData'),
     Input('word-analysis-close', 'n_clicks')],
    [State('word-analysis-modal', 'is_open'),
     State('city-dropdown', 'value')],
    prevent_initial_call=True
)
def toggle_word_modal(clickData, close_clicks, is_open, city_filter):
    """Handle word frequency chart clicks and modal toggling"""
    ctx_triggered = ctx.triggered
    
    # Close modal if close button clicked
    if ctx_triggered and ctx_triggered[0]['prop_id'] == 'word-analysis-close.n_clicks':
        return False, []
    
    # Open modal if chart clicked
    if clickData and clickData.get('points'):
        word = clickData['points'][0].get('x')
        if word:
            print(f"🔍 Word clicked: {word}")
            modal_content = create_modal_content(word, city_filter)
            return True, modal_content
    
    return False, []

# Callback for nationality pie chart click events and modal handling
@app.callback(
    [Output('nationality-analysis-modal', 'is_open'),
     Output('nationality-analysis-modal', 'children')],
    [Input('nationality-chart', 'clickData'),
     Input('nationality-analysis-close', 'n_clicks')],
    [State('nationality-analysis-modal', 'is_open'),
     State('city-dropdown', 'value')],
    prevent_initial_call=True
)
def toggle_nationality_modal(clickData, close_clicks, is_open, city_filter):
    """Handle nationality chart clicks and modal toggling"""
    ctx_triggered = ctx.triggered
    
    # Close modal if close button clicked
    if ctx_triggered and ctx_triggered[0]['prop_id'] == 'nationality-analysis-close.n_clicks':
        return False, []
    
    # Open modal if chart clicked
    if clickData and clickData.get('points'):
        nationality = clickData['points'][0].get('label')
        if nationality:
            print(f"🌍 Nationality clicked: {nationality}")
            modal_content = create_nationality_modal_content(nationality, city_filter)
            return True, modal_content
    
    return False, []

# Callback to handle data refresh
@app.callback(
    [Output('refresh-trigger', 'data'),
     Output('refresh-toast', 'is_open'),
     Output('refresh-toast', 'children'),
     Output('refresh-icon', 'className')],
    [Input('refresh-data-btn', 'n_clicks')],
    [State('refresh-trigger', 'data')],
    prevent_initial_call=True
)
def handle_data_refresh(refresh_clicks, current_trigger):
    """Handle data refresh button clicks"""
    if refresh_clicks:
        print("🔄 Manual data refresh triggered")
        
        # Clear cache to force fresh data load
        clear_cache_metadata()
        
        return (
            current_trigger + 1,  # Increment trigger
            True,  # Show toast
            "Data refreshed successfully! Latest information is now available.",
            "fas fa-sync fa-spin me-1"  # Spinning icon for visual feedback
        )
    
    return dash.no_update, False, "", "fas fa-sync me-1"

# Callback to stop the refresh icon spinning
@app.callback(
    Output('refresh-icon', 'className', allow_duplicate=True),
    [Input('app-data', 'data')],
    prevent_initial_call=True
)
def stop_refresh_spin(app_data):
    """Stop the refresh icon spinning when data is loaded"""
    return "fas fa-sync me-1"

# Callback to update data status indicator
@app.callback(
    Output('data-status-indicator', 'children'),
    [Input('app-data', 'data')],
    prevent_initial_call=False
)
def update_data_status(app_data):
    """Update the data status indicator"""
    if not app_data:
        return html.Span([
            html.I(className="fas fa-circle", style={"color": "#fbbf24", "marginRight": "0.5rem"}),
            "Loading..."
        ], style={"color": "#92400e", "fontSize": "0.875rem"})
    
    status = app_data.get('status', 'unknown')
    record_count = app_data.get('record_count', 0)
    
    if status == 'success':
        return html.Span([
            html.I(className="fas fa-circle", style={"color": "#10b981", "marginRight": "0.5rem"}),
            f"{record_count:,} records loaded"
        ], style={"color": "#065f46", "fontSize": "0.875rem"})
    else:
        return html.Span([
            html.I(className="fas fa-circle", style={"color": "#ef4444", "marginRight": "0.5rem"}),
            "Data Error"
        ], style={"color": "#991b1b", "fontSize": "0.875rem"})

# --- App Server ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
