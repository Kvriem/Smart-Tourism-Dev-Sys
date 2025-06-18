import dash
from dash import dcc, html, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
from datetime import datetime
import threading
# Import page components with better formatting
import             html.Li([
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
            ], className="nav-item"),otstrap_components as dbc
from datetime import datetime
import threading
# Import page components with better formatting
from pages.overview_page import (
    create_overview_content,
    get_cities_from_data,
    analyze_word_details,
    create_modal_content,
    analyze_nationality_details,
    create_nationality_modal_content
)
from pages.Hotels_page import create_hotels_content
from pages.trends_page import create_trends_content
from pages.recommendations_page import create_recommendations_content
from pages.geography_page import layout as geography_layout
from cache_utils import clear_cache_metadata

# --- App Setup ---
app = dash.Dash(__name__,                external_stylesheets=[
                    dbc.themes.BOOTSTRAP,
                    'assets/app.css',
                    'assets/overview.css',
                    'assets/style.css',
                    'assets/data-status.css',
                    'assets/modal-enhanced.css',
                    'assets/modern-charts.css',
                    'assets/trends.css',
                    'assets/enhanced-sidebar.css',
                    'assets/recommendations.css',
                    'assets/geography.css'
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
            ], className="nav-item"),
              html.Li([
                html.A([
                    html.Span("📈", className="nav-emoji"),
                    html.Span("Trends", className="nav-text"),
                    html.Span("", className="nav-indicator")
                ], href="#", className="nav-link", id="trends-link")
            ], className="nav-item"),
            
            html.Li([
                html.A([
                    html.Span("�", className="nav-emoji"),
                    html.Span("Recommendations", className="nav-text"),
                    html.Span("", className="nav-indicator")
                ], href="#", className="nav-link", id="recommendations-link")
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
filters_section = html.Div([    # Enhanced Page Title Section
    html.Div([
        html.Div([
            html.H1([
                html.Span("🚀", className="title-emoji"),
                "Tourism Analytics Dashboard"
            ], className="page-title enhanced-title")
        ], className="title-content")
    ], className="enhanced-content-header"),
      # Filters Header
    html.Div([        html.H5([
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
            dbc.Col([                html.Label([
                    html.Span("📍", className="label-emoji"),
                    "Location"
                ], className="filter-label"),                dcc.Dropdown(
                    id="city-dropdown",
                    options=[{'label': 'All Cities', 'value': 'all'}, {'label': 'Loading...', 'value': 'loading'}],
                    value="all",
                    className="enhanced-filter-dropdown",
                    placeholder="Select city"
                )
            ], lg=3, md=6, sm=12, className="filter-col"),            # Time Period Filter
            dbc.Col([                html.Label([
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
            dbc.Col([                html.Label([
                    html.Span("📅", className="label-emoji"),
                    "Start Date"
                ], className="filter-label"),
                dcc.DatePickerSingle(
                    id="start-date-picker",
                    placeholder="Start date",
                    className="enhanced-date-input",
                    display_format="MMM DD, YYYY"
                )            ], lg=2, md=6, sm=12, className="filter-col date-picker-disabled", id="start-date-col"),
              # End Date Filter
            dbc.Col([                html.Label([
                    html.Span("📆", className="label-emoji"),
                    "End Date"
                ], className="filter-label"),                dcc.DatePickerSingle(
                    id="end-date-picker",
                    placeholder="End date",
                    className="enhanced-date-input",
                    display_format="MMM DD, YYYY"
                )            ], lg=2, md=6, sm=12, className="filter-col date-picker-disabled", id="end-date-col"),
            dbc.Col([
                html.Div([
                    dbc.Button([
                        html.I(className="fas fa-undo me-1"),
                        "Reset"
                    ], color="outline-secondary", size="sm", className="action-btn reset-btn me-2", id="reset-filters-btn"),                    dbc.Button([
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
    filters_section,    # Content Body - Overview Page Content
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
        from pages.overview_page import load_data
        import json
        import time
        
        start_time = time.time()
        
        # Check if this is a refresh trigger
        ctx_triggered = ctx.triggered
        force_reload = False
        
        if ctx_triggered and ctx_triggered[0]['prop_id'] == 'refresh-trigger.data':
            print("🔄 Data refresh triggered - forcing reload...")
            force_reload = True
        
        # This will trigger data loading with the enhanced caching logic
        print("📊 Loading shared data for the app...")
        
        # For refresh triggers, we use the database directly for immediate response
        if force_reload:
            from database_config import load_data_from_database
            from pages.overview_page import clear_data_cache
            
            # Clear in-memory cache first
            clear_data_cache()
            
            # Load fresh data in background to avoid blocking
            import threading
            def background_refresh():
                try:
                    load_data_from_database(force_reload=True)
                    elapsed = time.time() - start_time
                    print(f"✅ Background refresh completed in {elapsed:.2f}s")
                except Exception as e:
                    print(f"❌ Background refresh error: {e}")
            
            thread = threading.Thread(target=background_refresh)
            thread.daemon = True
            thread.start()
        
        # Return metadata without storing large datasets in dcc.Store
        elapsed = time.time() - start_time
        return {
            'timestamp': datetime.now().isoformat(), 
            'refresh_count': refresh_count,
            'load_time': elapsed,
            'force_reload': force_reload
        }
    except Exception as e:
        print(f"Error pre-loading app data: {e}")
        return {'error': str(e), 'timestamp': datetime.now().isoformat()}

# Callback to populate city dropdown dynamically
@app.callback(
    Output('city-dropdown', 'options'),
    Input('city-dropdown', 'id'),
    prevent_initial_call=False
)
def populate_city_dropdown(_):
    """Populate city dropdown with data from database"""
    try:
        cities = get_cities_from_data()
        return cities
    except Exception as e:
        print(f"Error populating city dropdown: {e}")
        return [{'label': 'All Cities', 'value': 'all'}, {'label': 'Error Loading Cities', 'value': 'error'}]

# Navigation callback to handle page switching
@app.callback(
    [Output('current-page', 'data'),
     Output('overview-link', 'className'),
     Output('hotels-link', 'className'),
     Output('trends-link', 'className'),
     Output('recommendations-link', 'className'),
     Output('geography-link', 'className')],
    [Input('overview-link', 'n_clicks'),
     Input('hotels-link', 'n_clicks'),
     Input('trends-link', 'n_clicks'),
     Input('recommendations-link', 'n_clicks'),
     Input('geography-link', 'n_clicks')],
    prevent_initial_call=True
)
def handle_navigation(overview_clicks, hotels_clicks, trends_clicks, recommendations_clicks, geography_clicks):
    """Handle navigation between pages"""
    ctx_triggered = ctx.triggered
    if not ctx_triggered:
        return 'overview', 'nav-link active', 'nav-link', 'nav-link', 'nav-link', 'nav-link'
    
    button_id = ctx_triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'overview-link':
        return 'overview', 'nav-link active', 'nav-link', 'nav-link', 'nav-link', 'nav-link'
    elif button_id == 'hotels-link':
        return 'hotels', 'nav-link', 'nav-link active', 'nav-link', 'nav-link', 'nav-link'
    elif button_id == 'trends-link':
        return 'trends', 'nav-link', 'nav-link', 'nav-link active', 'nav-link', 'nav-link'
    elif button_id == 'recommendations-link':
        return 'recommendations', 'nav-link', 'nav-link', 'nav-link', 'nav-link active', 'nav-link'
    elif button_id == 'geography-link':
        return 'geography', 'nav-link', 'nav-link', 'nav-link', 'nav-link', 'nav-link active'
    
    return 'overview', 'nav-link active', 'nav-link', 'nav-link', 'nav-link', 'nav-link'

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
        # Show date pickers normally when "Custom Date" is selected
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
     Input('refresh-trigger', 'data')],  # Add refresh trigger dependency    prevent_initial_call=False
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
        end_date = None    # Return appropriate page content based on current page with error handling
    try:
        if current_page == 'hotels':
            content = create_hotels_content(city_filter, start_date, end_date)
        elif current_page == 'trends':
            content = create_trends_content(city_filter, start_date, end_date)
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
     Output('modal-word-title', 'children'),
     Output('modal-loading', 'style'),
     Output('modal-content', 'style'),
     Output('modal-content', 'children')],
    [Input('word-frequency-chart', 'clickData'),
     Input('close-modal', 'n_clicks')],
    [State('word-analysis-modal', 'is_open'),
     State('city-dropdown', 'value'),
     State('start-date-picker', 'date'),
     State('end-date-picker', 'date'),
     State('date-range-option', 'value'),
     State('app-data', 'data')],  # Add app-data state to avoid reloading data
    prevent_initial_call=True
)
def handle_word_click_and_modal(clickData, close_clicks, is_open, city_filter, start_date, end_date, date_range_option, app_data):
    """Handle word clicks and modal open/close"""
    # Determine which input triggered the callback
    ctx_triggered = ctx.triggered
    if not ctx_triggered:
        return False, "", {"display": "none"}, {"display": "none"}, ""
    
    trigger_id = ctx_triggered[0]['prop_id'].split('.')[0]
    
    # Close modal if close button clicked
    if trigger_id == 'close-modal':
        return False, "", {"display": "none"}, {"display": "none"}, ""
      # Handle word click
    if trigger_id == 'word-frequency-chart' and clickData:
        # Extract clicked word and type
        point = clickData['points'][0]
        clicked_word = point['y']  # Word is on y-axis
        word_type = point['customdata'][0]  # Extract word type from customdata array# Prepare modal title (simplified without word type)
        word_type_icon = "fa-thumbs-up text-success" if word_type == 'positive' else "fa-thumbs-down text-danger"
        
        modal_title = html.Div([
            html.I(className=f"fas {word_type_icon} me-2"),
            f"Analysis: '{clicked_word}'"
        ])
        
        # Show loading and hide content initially
        loading_style = {"display": "block", "text-align": "center", "padding": "2rem"}
        content_style = {"display": "none"}
        
        # Load data and perform analysis
        try:
            from pages.overview_page import load_data
            df = load_data()  # This will use cached data efficiently
            
            # Apply date filters if not "all time"
            if date_range_option != 'all_time':
                analysis_start_date = start_date
                analysis_end_date = end_date
            else:
                analysis_start_date = None
                analysis_end_date = None
            
            # Analyze word details
            word_data = analyze_word_details(
                df, clicked_word, word_type, 
                city_filter, analysis_start_date, analysis_end_date
            )
              # Create modal content
            modal_content = create_modal_content(word_data)
            
            # Hide loading and show content
            loading_style = {"display": "none"}
            content_style = {"display": "block"}
            
            return True, modal_title, loading_style, content_style, modal_content
            
        except Exception as e:
            print(f"Error analyzing word: {e}")
            error_content = html.Div([
                html.I(className="fas fa-exclamation-triangle text-warning fa-3x mb-3"),
                html.H5("Analysis Error", className="text-warning"),
                html.P(f"Unable to analyze '{clicked_word}'. Please try again.", className="text-muted")
            ], className="modal-error-content")
            
            loading_style = {"display": "none"}
            content_style = {"display": "block"}
            
            return True, modal_title, loading_style, content_style, error_content
    
    # Default: don't change modal state
    return is_open, "", {"display": "none"}, {"display": "none"}, ""

# Callback for nationality chart click events and modal handling
@app.callback(
    [Output('nationality-analysis-modal', 'is_open'),
     Output('modal-nationality-title', 'children'),
     Output('modal-nationality-loading', 'style'),
     Output('modal-nationality-content', 'style'),
     Output('modal-nationality-content', 'children')],    [Input('nationality-chart', 'clickData'),
     Input('close-nationality-modal', 'n_clicks')],
    [State('nationality-analysis-modal', 'is_open'),
     State('city-dropdown', 'value'),
     State('start-date-picker', 'date'),
     State('end-date-picker', 'date'),
     State('date-range-option', 'value'),
     State('app-data', 'data')],  # Add app-data state to avoid reloading data
    prevent_initial_call=True
)
def handle_nationality_click_and_modal(clickData, close_clicks, is_open, city_filter, start_date, end_date, date_range_option, app_data):
    """Handle nationality chart clicks and modal open/close"""
    # Determine which input triggered the callback
    ctx_triggered = ctx.triggered
    if not ctx_triggered:
        return False, "", {"display": "none"}, {"display": "none"}, ""
    
    trigger_id = ctx_triggered[0]['prop_id'].split('.')[0]
    
    # Close modal if close button clicked
    if trigger_id == 'close-nationality-modal':
        return False, "", {"display": "none"}, {"display": "none"}, ""
    
    # Handle nationality click
    if trigger_id == 'nationality-chart' and clickData:
        # Extract clicked nationality
        point = clickData['points'][0]
        clicked_nationality = point['y']  # Nationality is on y-axis
          # Prepare modal title
        modal_title = html.Div([
            html.I(className="fas fa-globe text-primary me-2"),
            f"Nationality Analysis: {clicked_nationality}"
        ])
        
        # Show loading and hide content initially
        loading_style = {"display": "block", "text-align": "center", "padding": "2rem"}
        content_style = {"display": "none"}
        
        # Load data and perform analysis
        try:
            from pages.overview_page import load_data, analyze_nationality_details, create_nationality_modal_content
            
            # Show loading indicator while we process the data
            # This gives immediate feedback to the user
            
            # Load data from cache - this should be very fast
            df = load_data()  # This will use cached data efficiently
            
            # Apply date filters if not "all time"
            if date_range_option != 'all_time':
                analysis_start_date = start_date
                analysis_end_date = end_date
            else:
                analysis_start_date = None
                analysis_end_date = None
            
            # Analyze nationality details
            nationality_data = analyze_nationality_details(
                df, clicked_nationality, 
                city_filter, analysis_start_date, analysis_end_date
            )
              # Create modal content
            modal_content = create_nationality_modal_content(nationality_data)
            
            # Hide loading and show content
            loading_style = {"display": "none"}
            content_style = {"display": "block"}
            
            return True, modal_title, loading_style, content_style, modal_content
            
        except Exception as e:
            print(f"Error analyzing nationality: {e}")
            error_content = html.Div([
                html.I(className="fas fa-exclamation-triangle text-warning fa-3x mb-3"),
                html.H5("Analysis Error", className="text-warning"),
                html.P(f"Unable to analyze '{clicked_nationality}'. Please try again.", className="text-muted")
            ], className="modal-error-content")
            
            loading_style = {"display": "none"}
            content_style = {"display": "block"}
            
            return True, modal_title, loading_style, content_style, error_content
    
    # Default: don't change modal state
    return is_open, "", {"display": "none"}, {"display": "none"}, ""

# Callback to handle data refresh button
@app.callback(
    [Output('refresh-trigger', 'data'),
     Output('refresh-toast', 'is_open'),
     Output('refresh-toast', 'children'),
     Output('refresh-toast', 'icon')],
    [Input('refresh-data-btn', 'n_clicks')],
    [State('refresh-trigger', 'data')],
    prevent_initial_call=True
)
def refresh_data(n_clicks, current_count):
    """Clear the cache and trigger a data refresh with improved performance"""
    if n_clicks:
        try:
            # Clear the cache metadata to force a reload
            from cache_utils import clear_cache_metadata
            from pages.overview_page import clear_data_cache
            import time
            
            start_time = time.time()
            print("🔄 Starting data refresh...")
            
            # Step 1: Clear all caches (both in-memory and disk cache)
            cache_result = clear_cache_metadata()
            memory_result = clear_data_cache()
            
            # Step 2: Immediately trigger a database reload in the background
            # This helps make the refresh more responsive
            import threading
            def background_load():
                try:
                    from database_config import load_data_from_database
                    print("🔄 Starting background data refresh...")
                    # Force reload from database
                    df = load_data_from_database(force_reload=True)
                    elapsed = time.time() - start_time
                    if df is not None and not df.empty:
                        print(f"✅ Background data refresh completed in {elapsed:.2f}s - {len(df)} records loaded")
                    else:
                        print(f"⚠️ Background data refresh completed in {elapsed:.2f}s - no data returned")
                except Exception as e:
                    print(f"❌ Error in background refresh: {e}")
            
            # Start a background thread to load the data
            thread = threading.Thread(target=background_load)
            thread.daemon = True  # Allow the thread to be killed when app exits
            thread.start()
            
            # Return success message
            return current_count + 1, True, "Data refresh initiated. New data will be loaded from database.", "success"
        except Exception as e:
            # Return error message
            print(f"Error refreshing data: {e}")
            import traceback
            traceback.print_exc()
            return current_count, True, f"Error refreshing data: {str(e)}", "danger"
    
    # No clicks, no update
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update

# Callback to update data status indicator
@app.callback(
    Output('data-status-indicator', 'children'),
    [Input('app-data', 'data'),
     Input('refresh-trigger', 'data')],
    prevent_initial_call=False
)
def update_data_status_indicator(app_data, refresh_count):
    """Update the data status indicator"""
    if app_data and 'timestamp' in app_data:
        try:
            timestamp = datetime.fromisoformat(app_data['timestamp'])
            now = datetime.now()
            time_diff = (now - timestamp).total_seconds()
            
            # Format the time difference
            if time_diff < 60:
                time_text = f"just now"
            elif time_diff < 3600:
                minutes = int(time_diff // 60)
                time_text = f"{minutes}m ago"
            else:
                hours = int(time_diff // 3600)
                time_text = f"{hours}h ago"
            
            # Display the data status
            return html.Span([
                html.I(className="fas fa-database me-1"),
                f"Data loaded {time_text}"
            ], className="data-status")
        except Exception as e:
            return html.Span([
                html.I(className="fas fa-exclamation-triangle me-1"),
                "Data status unknown"
            ], className="data-status error")
    
    return html.Span([
        html.I(className="fas fa-spinner fa-spin me-1"),
        "Loading data..."
    ], className="data-status loading")

# --- Callback to handle refresh button icon state
@app.callback(
    [Output('refresh-icon', 'className'),
     Output('refresh-data-btn', 'disabled')],
    [Input('refresh-data-btn', 'n_clicks'),
     Input('app-data', 'data')],
    [State('refresh-icon', 'className')],
    prevent_initial_call=True
)
def update_refresh_button_state(n_clicks, app_data, current_class):
    """Update the refresh button icon state without animation"""
    ctx_triggered = ctx.triggered
    if not ctx_triggered:
        return "fas fa-sync me-1", False
    
    trigger_id = ctx_triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'refresh-data-btn':
        # Button was clicked, disable but don't show spinner
        return "fas fa-sync me-1", True
    elif trigger_id == 'app-data':
        # Data was refreshed, restore normal icon and enable
        return "fas fa-sync me-1", False
    
    # Default case
    return current_class, False

# --- Run Server ---
if __name__ == '__main__':
    print("🚀 Starting Dashboard on http://localhost:8050")
    app.run(debug=True, host='127.0.0.1', port=8050)