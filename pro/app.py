import dash
from dash import dcc, html, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
from pages.overview_page import create_overview_content, get_cities_from_data, analyze_word_details, create_modal_content, analyze_nationality_details, create_nationality_modal_content
from pages.Hotels_page import create_hotels_content

# --- App Setup ---
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)

app.title = "Dashboard"

# --- Sidebar Layout ---
sidebar = html.Div([
    # Logo/Branding Section
    html.Div([
        html.H2([
            html.Span("🎯", style={"fontSize": "2rem", "marginRight": "0.5rem"}),
            "Analytics Hub"
        ], className="sidebar-brand"),
        html.Hr(className="sidebar-divider")
    ], className="sidebar-header"),
    
    # Navigation Menu
    html.Nav([
        html.Ul([
            html.Li([
                html.A([
                    html.Span("📊", style={"fontSize": "1.2rem", "marginRight": "0.75rem"}),
                    html.Span("Overview", className="nav-text")
                ], href="#", className="nav-link active", id="overview-link")
            ], className="nav-item"),
            
            html.Li([
                html.A([
                    html.Span("🏨", style={"fontSize": "1.2rem", "marginRight": "0.75rem"}),
                    html.Span("Hotels", className="nav-text")
                ], href="#", className="nav-link", id="hotels-link")
            ], className="nav-item"),
        ], className="nav-menu")
    ], className="sidebar-nav"),
    
    # Sidebar Footer
    html.Div([
        html.Hr(className="sidebar-divider"),
        html.P("© 2025 Dashboard", className="sidebar-footer-text")
    ], className="sidebar-footer")
    
], className="sidebar")

# --- Filters Section (Global for all pages) ---
filters_section = html.Div([
    # Enhanced Page Title Section
    html.Div([
        html.Div([
            html.H1([
                html.Span("🚀", style={"fontSize": "3rem", "marginRight": "1rem"}),
                "Tourism Analytics Dashboard"
            ], className="page-title enhanced-title"),
            html.P("✨ Discover insights, trends, and performance metrics in your comprehensive tourism analytics platform", className="page-subtitle enhanced-subtitle")
        ], className="title-content")
    ], className="enhanced-content-header"),
    
    # Filters Header
    html.Div([
        html.H5([
            html.Span("🔍", style={"fontSize": "1.2rem", "marginRight": "0.5rem"}),
            "Smart Filters"
        ], className="filters-title"),
    ], className="filters-header"),
    
    # Filters Content - Single Row
    html.Div([
        dbc.Row([
            # Location Filter
            dbc.Col([
                html.Label([
                    html.Span("📍", style={"fontSize": "1.1rem", "marginRight": "0.5rem"}),
                    "Location"
                ], className="filter-label"),
                dcc.Dropdown(
                    id="city-dropdown",
                    options=get_cities_from_data(),
                    value="all",
                    className="enhanced-filter-dropdown",
                    placeholder="Select city"
                )
            ], lg=3, md=6, sm=12, className="filter-col"),            # Time Period Filter
            dbc.Col([
                html.Label([
                    html.Span("⏰", style={"fontSize": "1.1rem", "marginRight": "0.5rem"}),
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
                    html.Span("📅", style={"fontSize": "1.1rem", "marginRight": "0.5rem"}),
                    "Start Date"
                ], className="filter-label"),
                dcc.DatePickerSingle(
                    id="start-date-picker",
                    placeholder="Start date",
                    className="enhanced-date-input",
                    display_format="MMM DD, YYYY"
                )
            ], lg=2, md=6, sm=12, className="filter-col date-picker-disabled", id="start-date-col", 
               style={"display": "block", "opacity": "0.5", "pointer-events": "none"}),
              # End Date Filter
            dbc.Col([
                html.Label([
                    html.Span("📆", style={"fontSize": "1.1rem", "marginRight": "0.5rem"}),
                    "End Date"
                ], className="filter-label"),
                dcc.DatePickerSingle(
                    id="end-date-picker",
                    placeholder="End date",
                    className="enhanced-date-input",
                    display_format="MMM DD, YYYY"
                )
            ], lg=2, md=6, sm=12, className="filter-col date-picker-disabled", id="end-date-col", 
               style={"display": "block", "opacity": "0.5", "pointer-events": "none"}),            # Action Buttons
            dbc.Col([
                html.Div([
                    dbc.Button([
                        html.I(className="fas fa-undo me-1"),
                        "Reset"
                    ], color="outline-secondary", size="sm", className="action-btn reset-btn", id="reset-filters-btn")
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
    sidebar,
    main_content
], className="app-container")

# --- Callbacks ---

# Navigation callback to handle page switching
@app.callback(
    [Output('current-page', 'data'),
     Output('overview-link', 'className'),
     Output('hotels-link', 'className')],
    [Input('overview-link', 'n_clicks'),
     Input('hotels-link', 'n_clicks')],
    prevent_initial_call=True
)
def handle_navigation(overview_clicks, hotels_clicks):
    """Handle navigation between pages"""
    ctx_triggered = ctx.triggered
    if not ctx_triggered:
        return 'overview', 'nav-link active', 'nav-link'
    
    button_id = ctx_triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'overview-link':
        return 'overview', 'nav-link active', 'nav-link'
    elif button_id == 'hotels-link':
        return 'hotels', 'nav-link', 'nav-link active'
    
    return 'overview', 'nav-link active', 'nav-link'

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
     Input('reset-filters-btn', 'n_clicks')]
)
def update_page_content(current_page, city_filter, start_date, end_date, date_range_option, reset_clicks):
    """Update page content based on current page and filters"""
    # Debug print to see what values are being passed
    print(f"Debug - current_page: {current_page}, city_filter: {city_filter}, start_date: {start_date}, end_date: {end_date}, date_range_option: {date_range_option}")    # Reset filters if reset button was clicked
    ctx_triggered = ctx.triggered
    if ctx_triggered and ctx_triggered[0]['prop_id'] == 'reset-filters-btn.n_clicks':
        city_filter = "all"
        start_date = None
        end_date = None
    
    # Only use date filters if custom date range is selected
    if date_range_option == 'all_time':
        start_date = None
        end_date = None
    
    # Return appropriate page content based on current page
    if current_page == 'hotels':
        return create_hotels_content(city_filter, start_date, end_date)
    else:  # Default to overview
        return create_overview_content(city_filter, start_date, end_date)

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
     State('date-range-option', 'value')],
    prevent_initial_call=True
)
def handle_word_click_and_modal(clickData, close_clicks, is_open, city_filter, start_date, end_date, date_range_option):
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
        word_type = point['customdata']  # positive or negative        # Prepare modal title (simplified without word type)
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
            df = load_data()
            
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
            ], className="text-center", style={"padding": "2rem"})
            
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
     Output('modal-nationality-content', 'children')],
    [Input('nationality-chart', 'clickData'),
     Input('close-nationality-modal', 'n_clicks')],
    [State('nationality-analysis-modal', 'is_open'),
     State('city-dropdown', 'value'),
     State('start-date-picker', 'date'),
     State('end-date-picker', 'date'),
     State('date-range-option', 'value')],
    prevent_initial_call=True
)
def handle_nationality_click_and_modal(clickData, close_clicks, is_open, city_filter, start_date, end_date, date_range_option):
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
            df = load_data()
            
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
            ], className="text-center", style={"padding": "2rem"})
            
            loading_style = {"display": "none"}
            content_style = {"display": "block"}
            
            return True, modal_title, loading_style, content_style, error_content
    
    # Default: don't change modal state
    return is_open, "", {"display": "none"}, {"display": "none"}, ""

# --- Run Server ---
if __name__ == '__main__':
    app.run(debug=True)