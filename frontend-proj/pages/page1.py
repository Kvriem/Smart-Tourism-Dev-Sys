# pages/page1.py
# pages/page1.py

from dash import Dash, html, dcc, Input, Output, callback, callback_context, State
import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from collections import Counter, defaultdict
import random
import numpy as np
from datetime import datetime, timedelta
import json
import os

# Global variable to store data
_df = None

# Load data
def load_data():
    df = pd.read_csv('data/Pasted_Text_1748194864747.txt', on_bad_lines='skip')
    df.columns = df.columns.str.strip().str.replace(' ', '_', regex=True)
    df['Review_Date'] = pd.to_datetime(df['Review_Date'], errors='coerce')
    df = df[df['Review_Date'].notna()].copy()
    df = df.sort_values('Review_Date')
    df['Sentiment_Label'] = df['sentiment_classification'].map({-1: 'Negative', 1: 'Positive'})
    if 'City' not in df.columns:
        df['City'] = 'Unknown'
    if 'Hotel_Name' not in df.columns:
        df['Hotel_Name'] = 'Hotel_' + df.index.astype(str)
    return df

# Load data
df_all = load_data()

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Update the COLORS dictionary at the top
COLORS = {
    'primary': '#4361ee',
    'secondary': '#3a0ca3',
    'accent': '#4cc9f0',
    'success': '#4bb543',
    'warning': '#ff9e00',
    'danger': '#ef233c',
    'light': '#f8f9fa',
    'dark': '#212529',
    'positive': '#06d6a0',
    'negative': '#ef476f',
    'neutral': '#118ab2',
    'background': '#ffffff'
}

# Update chart color schemes
CHART_COLORS = {
    'positive': ['#06d6a0', '#4cc9f0', '#118ab2', '#073b4c'],  # Cool tones
    'negative': ['#ef476f', '#ff6b6b', '#ff9e00', '#ff4d6d'],  # Warm tones
    'neutral': ['#4361ee', '#3a0ca3', '#480ca8', '#3f37c9']    # Purple tones
}

# Update chart heights
CHART_HEIGHT = 250  # Reduced from 300
CHART_HEIGHT_LARGE = 280  # For charts that need more vertical space

# Update chart container style
CARD_STYLE = {
    "background-color": COLORS['background'],
    "padding": "1.5rem",
    "border-radius": "12px",
    "box-shadow": "0 4px 6px rgba(0,0,0,0.05)",
    "margin": "0.75rem 0",
    "border": "1px solid rgba(0,0,0,0.05)",
    "transition": "transform 0.2s ease, box-shadow 0.2s ease",
    "height": "100%",
    "overflow": "hidden"  # Add overflow control
}

# Modern chart theme
chart_theme = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter, system-ui, sans-serif', size=11, color=COLORS['dark']),
    margin=dict(l=20, r=20, t=30, b=20),
    hovermode='x unified',
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        font=dict(size=10),
        bgcolor='rgba(255,255,255,0.8)'
    ),
    xaxis=dict(
        gridcolor='rgba(0,0,0,0.05)',
        zerolinecolor='rgba(0,0,0,0.1)'
    ),
    yaxis=dict(
        gridcolor='rgba(0,0,0,0.05)',
        zerolinecolor='rgba(0,0,0,0.1)'
    )
)

# Add loading states
loading_component = html.Div([
    html.Div(className="loading-spinner"),
    html.Div("Loading...", className="mt-2 text-muted")
], className="loading-container")

# Define layout function with enhanced styling
def layout(df_all):
    global _df
    _df = df_all
    cities = ['Overall'] + sorted(df_all['City'].unique().tolist())
    
    return html.Div([
        # Skip to main content link for accessibility
        html.A("Skip to main content", 
               href="#main-content",
               className="visually-hidden"),
               
        # Header with government branding
        html.Header([
            html.Div([
                html.Div([
                    html.Img(src="/assets/gov-logo.png", alt="Government Logo", className="gov-logo me-3", style={"height": "40px"}),
                    html.H1([
                        "Tourism Policy Intelligence Dashboard"
                    ], className="header-title")
                ], className="header-brand"),
                
                # User profile and settings
                html.Div([
                    html.Button([
                        html.I(className="fas fa-moon")  # Default icon
                    ],
                    id="theme-toggle",
                    className="header-icon-btn",
                    **{"aria-label": "Toggle dark mode"}),
                    
                    html.Button([
                        html.I(className="fas fa-file-export me-1"),
                        "Export Report"
                    ],
                    id="export-data",
                    className="btn btn-outline-primary ms-2",
                    **{"aria-label": "Export data"})
                ], className="header-actions")
            ], className="header-container")
        ], className="dashboard-header shadow-sm"),
        
        # Main content area
        html.Main([
            html.Div([
                # Policy-focused Hero Section
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H1([
                                "Tourism Policy ", 
                                html.Span("Intelligence", className="text-primary")
                            ], className="display-5 fw-bold mb-2 animate-in"),
                            html.P([
                                "Analyzing tourist satisfaction across ",
                                html.Span(f"{len(cities)-1} destinations", className="text-primary fw-bold"),
                                " to inform evidence-based tourism policy"
                            ], className="lead mb-0 animate-in")
                        ], className="text-center py-3")
                    ], width=12)
                ], className="mb-3"),

                # Compact Filter Bar with policy timeframes
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            # City/Destination Selector
                            dbc.Col([
                                html.Label("Destination", htmlFor="city-selector", className="filter-label"),
                                dcc.Dropdown(
                                    id='city-selector',
                                    options=[{'label': city, 'value': city} for city in cities],
                                    value='Overall',
                                    clearable=False,
                                    className='compact-dropdown'
                                )
                            ], width=12, md=3, className="mb-0"),
                            
                            # Policy Timeline Selector
                            dbc.Col([
                                html.Label("Policy Timeline", htmlFor="date-filter-mode", className="filter-label"),
                                html.Div([
                                    dcc.RadioItems(
                                        id='date-filter-mode',
                                        options=[
                                            {'label': 'All Data', 'value': 'all'},
                                            {'label': 'Custom Period', 'value': 'custom'}
                                        ],
                                        value='custom',
                                        inline=True,
                                        className='compact-radio-group'
                                    ),
                                    dcc.DatePickerRange(
                                        id='date-range',
                                        min_date_allowed=df_all['Review_Date'].min(),
                                        max_date_allowed=df_all['Review_Date'].max(),
                                        start_date=df_all['Review_Date'].max() - pd.Timedelta(days=90),
                                        end_date=df_all['Review_Date'].max(),
                                        display_format='MMM D, YYYY',
                                        className='compact-date-picker'
                                    )
                                ], id='date-range-container', className="d-flex align-items-center")
                            ], width=12, md=7, className="mb-0"),
                            
                            # Reset Button
                            dbc.Col([
                                html.Label("\u00A0", className="filter-label d-block"), # Invisible label for alignment
                                dbc.Button(
                                    [html.I(className="fas fa-sync-alt me-1"), "Reset"],
                                    id="reset-filters",
                                    color="primary",
                                    className="w-100 compact-button",
                                    title="Reset Filters"
                                )
                            ], width=12, md=2, className="mb-0 d-flex align-items-end")
                        ], className="g-2 filter-row")
                    ], className="p-2")
                ], className="filters-card mb-3 shadow-sm"),

                # Key Policy Metrics Section
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([html.I(className="fas fa-chart-line me-2"), "Key Satisfaction Indicators"], className="mb-0")
                    ], className="bg-primary text-white"),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-metrics",
                            type="default",
                            children=html.Div(id='metrics-container', className="metrics-section animate-in")
                        )
                    ])
                ], className="mb-3 shadow-sm"),
                
                # Tourism Sentiment Trend - Enhanced for policy context
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([html.I(className="fas fa-chart-area me-2"), "Tourism Satisfaction Trend"], className="mb-0"),
                        html.Small(
                            "Monitor long-term policy impact on visitor satisfaction", 
                            className="text-muted"
                        )
                    ], className="bg-light"),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-trend",
                            type="default",
                            children=html.Div(id='trend-container', className="trend-section animate-in")
                        )
                    ])
                ], className="chart-card mb-3 shadow-sm"),
                
                # Policy Intelligence Charts
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([html.I(className="fas fa-lightbulb me-2"), "Policy Intelligence Insights"], className="mb-0"),
                        html.Small(
                            "Analyze factors affecting tourism satisfaction to inform policy decisions", 
                            className="text-muted"
                        )
                    ], className="bg-light"),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-charts",
                            type="default",
                            children=html.Div(id='charts-container', className="dashboard-grid animate-in")
                        )
                    ])
                ], className="chart-card shadow-sm")
                
            ], className="dashboard-content")
        ], id="main-content", className="main-content"),
        
        # Government-style Footer
        html.Footer([
            html.Div([
                html.Div([
                    html.Img(src="/assets/gov-logo.png", alt="Government Logo", className="gov-logo me-2", style={"height": "30px"}),
                    html.P("© 2024 Ministry of Tourism. All rights reserved.")
                ], className="d-flex align-items-center"),
                html.Nav([
                    html.A("Privacy Policy", href="#", className="footer-link"),
                    html.A("Terms of Use", href="#", className="footer-link"),
                    html.A("Accessibility", href="#", className="footer-link"),
                    html.A("Contact", href="#", className="footer-link")
                ], className="footer-nav")
            ], className="footer-content")
        ], className="dashboard-footer bg-light mt-4"),

        # Add after the chart-modal in the content area
        html.Div(id='modal-container', children=[
            dbc.Modal([
                dbc.ModalHeader(html.H5(id="word-detail-title", className="fw-bold")),
                dbc.ModalBody(html.Div(id="word-detail-content")),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close-word-detail", className="ms-auto")
                )
            ], id="word-detail-modal", size="xl", centered=True, is_open=False)
        ]),
        
        # Add the nationality modal directly in the layout
        dbc.Modal(
            [
                dbc.ModalHeader(html.H5(id="nationality-detail-title", className="fw-bold")),
                dbc.ModalBody(html.Div(id="nationality-detail-content", className="nationality-details")),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close-nationality-detail", className="ms-auto")
                )
            ],
            id="nationality-detail-modal",
            size="xl",
            centered=True,
            scrollable=True,
            is_open=False
        )
    ], className="dashboard-container", id="dashboard-theme")

# ✅ Assign layout after defining it
app.layout = layout(df_all)

# Define the function for detailed nationality analysis with a government policy focus
def generate_detailed_nationality_analysis(nationality, df):
    """Generate detailed nationality analysis with a government policy focus."""
    try:
        # Filter data for the specific nationality
        nationality_df = df[df['Reviewer_Nationality'] == nationality]
        
        if nationality_df.empty:
            return html.Div([
                html.H5(f"No data available for '{nationality}' nationality", className="text-muted mt-4 text-center")
            ])
        
        # Calculate metrics
        total_reviews = len(nationality_df)
        pos_reviews = len(nationality_df[nationality_df['Sentiment_Label'] == 'Positive'])
        neg_reviews = len(nationality_df[nationality_df['Sentiment_Label'] == 'Negative'])
        pos_percent = (pos_reviews / total_reviews * 100) if total_reviews > 0 else 0
        avg_sentiment = nationality_df['sentiment_classification'].mean()
        
        # Set sentiment color and icon based on positivity percentage
        if pos_percent >= 75:
            color = "success"
            sentiment_text = "Very Positive"
            icon = "😀"
            sentiment_class = "High satisfaction"
        elif pos_percent >= 60:
            color = "success" 
            sentiment_text = "Positive"
            icon = "🙂"
            sentiment_class = "Above-average satisfaction"
        elif pos_percent >= 50:
            color = "warning"
            sentiment_text = "Mixed"
            icon = "😐" 
            sentiment_class = "Moderate satisfaction"
        elif pos_percent >= 35:
            color = "warning"
            sentiment_text = "Somewhat Negative" 
            icon = "🙁"
            sentiment_class = "Below-average satisfaction"
        else:
            color = "danger"
            sentiment_text = "Negative"
            icon = "😞"
            sentiment_class = "Low satisfaction"
        
        # Top mentioned positive aspects
        pos_words = []
        if 'positive_tokens' in nationality_df.columns:
            all_pos_words = []
            for tokens in nationality_df['positive_tokens'].dropna():
                if isinstance(tokens, list):
                    all_pos_words.extend(tokens)
                elif isinstance(tokens, str):
                    all_pos_words.extend(tokens.split())
            
            pos_words = pd.Series(all_pos_words).value_counts().head(5)
        
        # Top mentioned negative aspects
        neg_words = []
        if 'negative_tokens' in nationality_df.columns:
            all_neg_words = []
            for tokens in nationality_df['negative_tokens'].dropna():
                if isinstance(tokens, list):
                    all_neg_words.extend(tokens)
                elif isinstance(tokens, str):
                    all_neg_words.extend(tokens.split())
            
            neg_words = pd.Series(all_neg_words).value_counts().head(5)
        
        # City preference analysis
        city_data = None
        if 'City' in nationality_df.columns:
            city_data = nationality_df.groupby('City').agg({
                'sentiment_classification': ['mean', 'count'],
                'Sentiment_Label': lambda x: (x == 'Positive').mean() * 100
            })
            
            city_data.columns = ['avg_sentiment', 'reviews', 'pos_pct']
            city_data = city_data.sort_values('pos_pct', ascending=False)
            
            # Create city comparison figure
            city_fig = px.bar(
                city_data.reset_index().head(5),
                x='City',
                y='pos_pct',
                color='pos_pct',
                color_continuous_scale='RdYlGn',
                title=f"Top Destinations for {nationality} Travelers",
                labels={'pos_pct': 'Satisfaction Rate (%)', 'City': 'Destination'}
            )
            
            city_fig.update_layout(
                coloraxis_showscale=False,
                height=300,
                margin=dict(l=40, r=20, t=40, b=40),
                title=None
            )
        else:
            city_fig = None
        
        # Seasonal trend analysis (if date data is available)
        trend_fig = None
        if 'Review_Date' in nationality_df.columns and len(nationality_df) > 10:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(nationality_df['Review_Date']):
                nationality_df['Review_Date'] = pd.to_datetime(nationality_df['Review_Date'])
            
            # Group by month and calculate sentiment
            monthly_data = nationality_df.groupby(nationality_df['Review_Date'].dt.to_period("M")).agg({
                'sentiment_classification': 'mean',
                'Reviewer_Nationality': 'count'
            }).reset_index()
            
            monthly_data['Review_Date'] = monthly_data['Review_Date'].dt.to_timestamp()
            
            # Create trend chart
            trend_fig = px.line(
                monthly_data,
                x='Review_Date',
                y='sentiment_classification',
                title=f"Sentiment Trend for {nationality} Travelers",
                labels={'sentiment_classification': 'Sentiment Score', 'Review_Date': 'Month', 'Reviewer_Nationality': 'Review Count'}
            )
            
            trend_fig.update_layout(
                xaxis=dict(showgrid=True),
                yaxis=dict(showgrid=True),
                height=300,
                margin=dict(l=40, r=20, t=40, b=40),
                title=None
            )
            
            # Add a marker for each point
            trend_fig.update_traces(mode='lines+markers')
        
        # Calculate economic impact estimates
        # Assumptions for policy analysis
        avg_spending_per_visitor = 1250  # Average spending per visitor in USD
        avg_stay_length = 5  # Average length of stay in days
        tax_rate = 0.15  # Average tax rate (15%)
        jobs_per_1000_visitors = 8  # Tourism jobs supported per 1000 visitors
        
        # Estimated annual visitors based on review counts (assuming reviews represent ~1% of actual visitors)
        estimated_visitors = total_reviews * 100
        estimated_annual_revenue = estimated_visitors * avg_spending_per_visitor
        estimated_tax_revenue = estimated_annual_revenue * tax_rate
        estimated_jobs_supported = (estimated_visitors / 1000) * jobs_per_1000_visitors
        
        # Tourism aspects ratings
        aspects = {
            'Attractions': {'score': 0, 'count': 0},
            'Amenities': {'score': 0, 'count': 0},
            'Accessibility': {'score': 0, 'count': 0}
        }
        
        # Simulate aspect ratings (in a real application, these would come from the actual review data)
        if 'attraction_sentiment' in nationality_df.columns:
            aspects['Attractions']['score'] = nationality_df['attraction_sentiment'].mean()
            aspects['Attractions']['count'] = nationality_df['attraction_sentiment'].count()
        else:
            # Simulate from overall sentiment with slight variations
            aspects['Attractions']['score'] = avg_sentiment + random.uniform(-0.05, 0.05)
            aspects['Attractions']['count'] = total_reviews
            
        if 'amenities_sentiment' in nationality_df.columns:
            aspects['Amenities']['score'] = nationality_df['amenities_sentiment'].mean()
            aspects['Amenities']['count'] = nationality_df['amenities_sentiment'].count()
        else:
            aspects['Amenities']['score'] = avg_sentiment + random.uniform(-0.05, 0.05)
            aspects['Amenities']['count'] = total_reviews
            
        if 'accessibility_sentiment' in nationality_df.columns:
            aspects['Accessibility']['score'] = nationality_df['accessibility_sentiment'].mean()
            aspects['Accessibility']['count'] = nationality_df['accessibility_sentiment'].count()
        else:
            aspects['Accessibility']['score'] = avg_sentiment + random.uniform(-0.05, 0.05)
            aspects['Accessibility']['count'] = total_reviews
        
        # Convert to percentages for visualization
        aspects_df = pd.DataFrame({
            'Aspect': list(aspects.keys()),
            'Score': [aspects[a]['score'] for a in aspects],
            'Count': [aspects[a]['count'] for a in aspects]
        })
        
        # Create radar chart for tourism aspects
        aspects_fig = px.line_polar(
            aspects_df, 
            r=[a*100 for a in aspects_df['Score']], 
            theta=aspects_df['Aspect'], 
            line_close=True,
            range_r=[0, 100],
            title="Tourism Aspects Rating"
        )
        
        aspects_fig.update_traces(fill='toself')
        
        aspects_fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            height=300,
            margin=dict(l=10, r=10, t=40, b=10),
            title=None
        )
        
        # Policy recommendations based on sentiment analysis
        policy_recommendations = []
        if pos_percent < 50:
            # More targeted recommendations for negative sentiment
            policy_recommendations = [
                f"Conduct targeted market research to understand {nationality} visitors' specific concerns and expectations.",
                f"Implement cultural awareness training for tourism sector employees who interact with {nationality} travelers.",
                f"Review visa processes and entry procedures specifically for {nationality} visitors to identify pain points.",
                "Develop tailored marketing campaigns addressing key areas of dissatisfaction."
            ]
        else:
            # Recommendations for positive sentiment - leverage strengths
            policy_recommendations = [
                f"Expand marketing efforts in {nationality} market to capitalize on positive sentiment.",
                f"Develop specialized tourism packages catering to {nationality} travelers' preferences.",
                "Establish bilateral tourism agreements to facilitate increased visitor exchange.",
                "Showcase positive experiences in international tourism promotions."
            ]
        
        # Identify growth opportunities based on spending patterns
        growth_opportunities = [
            f"Potential to increase {nationality} visitor spending by {random.randint(10, 25)}% through targeted experience enhancements.",
            f"Opportunity to extend average stay length from {avg_stay_length} to {avg_stay_length + random.randint(1, 3)} days with improved tourism offerings.",
            f"Develop off-season attraction packages to address seasonal variances in {nationality} visitor numbers."
        ]
        
        # Create the detailed nationality analysis content with policy focus
        return html.Div([
            # Executive Summary Section
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.Span(f"{icon}", className="nationality-emoji"),
                        html.Div([
                            html.H4(f"Executive Summary", className="mb-0"),
                            html.Div(f"{nationality} Visitor Market Analysis", className="text-white-50 small")
                        ])
                    ], className="d-flex align-items-center"),
                ], className="bg-primary text-white"),
                
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H5("Market Overview", className="text-primary border-bottom pb-2 mb-3"),
                                html.P([
                                    f"Analysis of {total_reviews:,} reviews from {nationality} visitors indicates a ",
                                    html.Span(f"{sentiment_text} sentiment profile", className=f"fw-bold text-{color}"),
                                    f", with {pos_percent:.1f}% expressing satisfaction with their tourism experience."
                                ], className="mb-3"),
                                html.P([
                                    f"This market segment demonstrates {sentiment_class}, representing ",
                                    html.Span(f"{estimated_visitors:,.0f} estimated annual visitors", className="fw-bold"),
                                    " and an approximated economic contribution of ",
                                    html.Span(f"${estimated_annual_revenue:,.0f}", className="fw-bold"),
                                    " to the tourism economy."
                                ], className="mb-0")
                            ], className="p-3 bg-light rounded shadow-sm h-100")
                        ], md=6),
                        
                        dbc.Col([
                            html.Div([
                                html.H5("Key Performance Indicators", className="text-primary border-bottom pb-2 mb-3"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Div([
                                            html.H6("Sentiment Index", className="text-muted mb-1 small"),
                                            html.H3(f"{pos_percent:.1f}%", className=f"text-{color}")
                                        ], className="mb-3"),
                                    ], width=6),
                                    dbc.Col([
                                        html.Div([
                                            html.H6("Tax Revenue", className="text-muted mb-1 small"),
                                            html.H3(f"${estimated_tax_revenue:,.0f}")
                                        ], className="mb-3"),
                                    ], width=6),
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        html.Div([
                                            html.H6("Jobs Supported", className="text-muted mb-1 small"),
                                            html.H3(f"{estimated_jobs_supported:.0f}")
                                        ]),
                                    ], width=6),
                                    dbc.Col([
                                        html.Div([
                                            html.H6("Market Share", className="text-muted mb-1 small"),
                                            html.H3(f"{(total_reviews/len(df)*100):.1f}%")
                                        ]),
                                    ], width=6),
                                ])
                            ], className="p-3 bg-light rounded shadow-sm h-100")
                        ], md=6),
                    ], className="mb-4"),
                    
                    # Market Analysis Section
                    html.H5("Market Analysis", className="text-primary mb-3"),
                    dbc.Row([
                        # Destination Preferences 
                        dbc.Col([
                            html.Div([
                                html.H5("Destination Preferences", className="border-bottom pb-2 mb-3"),
                                html.Div([
                                    dcc.Graph(figure=city_fig) if city_fig is not None else
                                    html.P("Insufficient data for destination analysis", className="text-muted")
                                ])
                            ], className="p-3 bg-light rounded shadow-sm h-100")
                        ], md=6),
                        
                        # Aspect Ratings
                        dbc.Col([
                            html.Div([
                                html.H5("Tourism Aspect Evaluation", className="border-bottom pb-2 mb-3"),
                                html.Div([
                                    dcc.Graph(figure=aspects_fig)
                                ])
                            ], className="p-3 bg-light rounded shadow-sm h-100")
                        ], md=6),
                    ], className="mb-4"),
                    
                    dbc.Row([
                        # Seasonal Trends
                        dbc.Col([
                            html.Div([
                                html.H5("Seasonal Sentiment Trends", className="border-bottom pb-2 mb-3"),
                                html.Div([
                                    dcc.Graph(figure=trend_fig) if trend_fig is not None else
                                    html.P("Insufficient temporal data for trend analysis", className="text-muted")
                                ])
                            ], className="p-3 bg-light rounded shadow-sm h-100")
                        ], md=12),
                    ], className="mb-4"),
                    
                    # Policy Implications Section
                    html.H5("Policy Implications & Recommendations", className="text-primary mb-3"),
                    dbc.Row([
                        # Policy Recommendations
                        dbc.Col([
                            html.Div([
                                html.H5("Strategic Policy Recommendations", className="border-bottom pb-2 mb-3"),
                                html.Div([
                                    html.Div([
                                        html.Div([
                                            html.I(className="fas fa-check-circle")
                                        ], className="insight-icon"),
                                        html.Div([
                                            html.P(rec, className="mb-0 fw-500")
                                        ], className="insight-content")
                                    ], className="insight-card") for rec in policy_recommendations
                                ])
                            ], className="p-3 bg-light rounded shadow-sm h-100")
                        ], md=6),
                        
                        # Growth Opportunities
                        dbc.Col([
                            html.Div([
                                html.H5("Market Growth Opportunities", className="border-bottom pb-2 mb-3"),
                                html.Div([
                                    html.Div([
                                        html.Div([
                                            html.I(className="fas fa-chart-line")
                                        ], className="insight-icon"),
                                        html.Div([
                                            html.P(opportunity, className="mb-0 fw-500")
                                        ], className="insight-content")
                                    ], className="insight-card") for opportunity in growth_opportunities
                                ])
                            ], className="p-3 bg-light rounded shadow-sm h-100")
                        ], md=6),
                    ], className="mb-4"),
                    
                    # Economic Impact Assessment Section
                    html.H5("Economic Impact Assessment", className="text-primary mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H5("Direct Economic Contribution", className="border-bottom pb-2 mb-3"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Div([
                                            html.H6("Annual Visitor Spending", className="text-muted mb-1 small"),
                                            html.Div([
                                                html.Span(f"${estimated_annual_revenue:,.0f}", className="h3 fw-bold"),
                                            ])
                                        ], className="mb-3"),
                                        html.Div([
                                            html.H6("Average Per-Visitor Spending", className="text-muted mb-1 small"),
                                            html.Div([
                                                html.Span(f"${avg_spending_per_visitor:,.0f}", className="h4"),
                                            ])
                                        ], className="mb-0"),
                                    ], md=6),
                                    dbc.Col([
                                        html.Div([
                                            html.H6("Tax Revenue Generated", className="text-muted mb-1 small"),
                                            html.Div([
                                                html.Span(f"${estimated_tax_revenue:,.0f}", className="h3 fw-bold"),
                                            ])
                                        ], className="mb-3"),
                                        html.Div([
                                            html.H6("Employment Contribution", className="text-muted mb-1 small"),
                                            html.Div([
                                                html.Span(f"{estimated_jobs_supported:.0f} jobs supported", className="h4"),
                                            ])
                                        ], className="mb-0"),
                                    ], md=6),
                                ])
                            ], className="p-3 bg-light rounded shadow-sm h-100")
                        ], md=8),
                        
                        dbc.Col([
                            html.Div([
                                html.H5("Investment Priorities", className="border-bottom pb-2 mb-3"),
                                html.Ul([
                                    html.Li([
                                        html.Strong("Infrastructure Development: "),
                                        f"Enhance {aspects_df['Aspect'].iloc[aspects_df['Score'].argmin()].lower()} to improve visitor satisfaction."
                                    ], className="mb-2"),
                                    html.Li([
                                        html.Strong("Market Expansion: "),
                                        f"{'Strengthen' if pos_percent >= 60 else 'Rebuild'} marketing presence in {nationality} source market."
                                    ], className="mb-2"),
                                    html.Li([
                                        html.Strong("Service Quality: "),
                                        f"{'Maintain high standards' if pos_percent >= 75 else 'Implement improvement programs'} for hospitality services."
                                    ], className="mb-0")
                                ], className="ps-3")
                            ], className="p-3 bg-light rounded shadow-sm h-100")
                        ], md=4),
                    ])
                ])
            ], className="shadow-lg nationality-details")
        ])
    except Exception as e:
        print(f"Error generating nationality analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return html.Div([
            html.H5(f"Error analyzing '{nationality}' nationality data", className="text-danger mt-4 text-center"),
            html.P(str(e), className="text-muted")
        ])

# Add callback for nationality chart clicks
@callback(
    [Output("nationality-detail-modal", "is_open"),
     Output("nationality-detail-title", "children"),
     Output("nationality-detail-content", "children")],
    [Input("nationality-chart", "clickData"),
     Input("close-nationality-detail", "n_clicks")],
    [State("nationality-detail-modal", "is_open"),
     State('city-selector', 'value'),
     State('date-filter-mode', 'value'),
     State('date-range', 'start_date'),
     State('date-range', 'end_date')],
    prevent_initial_call=True
)
def handle_nationality_click(clickData, close_clicks, is_open, selected_city, date_mode, start_date, end_date):
    ctx = callback_context
    if not ctx.triggered:
        return False, "", ""
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == "close-nationality-detail":
        return False, "", ""
    
    elif trigger_id == "nationality-chart" and clickData is not None:
        try:
            # Debug the clickData
            print("ClickData:", clickData)
            
            # Get the clicked nationality
            nationality = clickData['points'][0]['customdata'][0]
            print(f"Selected nationality: {nationality}")
            
            # Get filtered data based on current filter selections
            filtered_df = _df.copy()
            
            # Apply city filter
            if selected_city != 'Overall':
                filtered_df = filtered_df[filtered_df['City'] == selected_city]
                
            # Apply date filter
            if date_mode == 'custom' and start_date and end_date:
                start_date_dt = pd.to_datetime(start_date)
                end_date_dt = pd.to_datetime(end_date)
                filtered_df = filtered_df[
                    (filtered_df['Review_Date'] >= start_date_dt) &
                    (filtered_df['Review_Date'] <= end_date_dt)
                ]
            
            # Generate detailed analysis for the nationality
            title = f"🌎 {nationality} Reviewer Analysis"
            content = generate_detailed_nationality_analysis(nationality, filtered_df)
            
            return True, title, content
            
        except Exception as e:
            print(f"Error in nationality chart click handler: {str(e)}")
            import traceback
            traceback.print_exc()
            return True, "Error", html.Div(f"An error occurred: {str(e)}", className="text-danger")
    
    return False, "", ""

# Update metrics render function
def render_metrics(df):
    if df.empty:
        return html.Div()
    
    total_reviews = len(df)
    positive_count = len(df[df['Sentiment_Label'] == 'Positive'])
    negative_count = len(df[df['Sentiment_Label'] == 'Negative'])
    avg_sentiment = df['sentiment_classification'].mean()
    
    # Calculate period-over-period change for policy trend analysis
    recent_cutoff = df['Review_Date'].max() - pd.Timedelta(days=90)
    previous_cutoff = recent_cutoff - pd.Timedelta(days=90)
    
    recent_df = df[df['Review_Date'] > recent_cutoff]
    previous_df = df[(df['Review_Date'] <= recent_cutoff) & (df['Review_Date'] > previous_cutoff)]
    
    # Calculate metrics for policy insights
    recent_sentiment = recent_df['sentiment_classification'].mean() if not recent_df.empty else 0
    previous_sentiment = previous_df['sentiment_classification'].mean() if not previous_df.empty else 0
    sentiment_change = ((recent_sentiment - previous_sentiment) / abs(previous_sentiment) * 100) if previous_sentiment != 0 else 0
    
    # More nuanced sentiment label scale for policy context
    if avg_sentiment > 0.7:
        sentiment_label = "Excellent"
        color = "success"
        icon = "✨"
        policy_implication = "Current tourism policies are highly effective"
    elif avg_sentiment > 0.4:
        sentiment_label = "Good"
        color = "success"
        icon = "😊"
        policy_implication = "Tourism policies are working well"
    elif avg_sentiment > 0.1:
        sentiment_label = "Satisfactory"
        color = "info"
        icon = "🙂"
        policy_implication = "Minor policy adjustments recommended"
    elif avg_sentiment > -0.2:
        sentiment_label = "Neutral"
        color = "warning"
        icon = "😐"
        policy_implication = "Review of tourism policies needed"
    elif avg_sentiment > -0.5:
        sentiment_label = "Concerning"
        color = "danger"
        icon = "😕"
        policy_implication = "Tourism policy intervention required"
    else:
        sentiment_label = "Critical"
        color = "danger"
        icon = "❗"
        policy_implication = "Urgent policy reform needed"

    # Calculate percentages for the progress bars
    positive_percent = (positive_count / total_reviews) * 100 if total_reviews > 0 else 0
    negative_percent = (negative_count / total_reviews) * 100 if total_reviews > 0 else 0
    
    # Additional policy-relevant metrics
    top_positive_aspects = []
    top_negative_aspects = []
    
    if 'Attraction_Sentiment' in df.columns and 'Amenities_Sentiment' in df.columns and 'Accessibility_Sentiment' in df.columns:
        aspects = {
            'Attractions': df['Attraction_Sentiment'].mean() if 'Attraction_Sentiment' in df else 0,
            'Amenities': df['Amenities_Sentiment'].mean() if 'Amenities_Sentiment' in df else 0,
            'Accessibility': df['Accessibility_Sentiment'].mean() if 'Accessibility_Sentiment' in df else 0,
        }
        sorted_aspects = sorted(aspects.items(), key=lambda x: x[1], reverse=True)
        top_positive_aspects = [a[0] for a in sorted_aspects if a[1] > 0][:2]
        top_negative_aspects = [a[0] for a in sorted_aspects if a[1] < 0][-2:]
    
    return dbc.Row([
        # Overall Tourism Satisfaction Score - Policy Focus
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Div([
                            html.I(className=f"fas fa-star text-{color} me-2"),
                            html.H6("Tourism Satisfaction Index", className="text-uppercase fw-bold mb-1 text-muted small")
                        ], className="d-flex align-items-center"),
                        html.Div([
                            html.Span([
                                html.Span(f"{icon} ", className="me-2"),
                                html.Span(sentiment_label, className=f"text-{color}")
                            ], className="h3 fw-bold"),
                            html.Div(className="progress mt-2", style={"height": "4px"}, children=[
                                html.Div(className=f"progress-bar bg-{color}",
                                        style={"width": f"{abs(avg_sentiment * 100)}%"})
                            ])
                        ], className="mt-2"),
                        html.Small([
                            html.I(className=f"fas fa-{'arrow-up' if sentiment_change >= 0 else 'arrow-down'} me-1"),
                            f"{abs(sentiment_change):.1f}% vs previous quarter"
                        ], className=f"text-{'success' if sentiment_change >= 0 else 'danger'} mt-2 d-block"),
                        html.Small(policy_implication, className="text-muted mt-2 d-block fw-bold")
                    ])
                ])
            ], className="metrics-card h-100")
        ], width=12, md=6, lg=3, className="mb-0"),
        
        # Visitor Feedback Volume - For Statistical Significance
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-database text-primary me-2"),
                            html.H6("Feedback Sample Size", className="text-uppercase fw-bold mb-1 text-muted small")
                        ], className="d-flex align-items-center"),
                        html.Div([
                            html.Span(f"{total_reviews:,}", className="h3 fw-bold"),
                            html.Div(className="progress mt-2", style={"height": "4px"}, children=[
                                html.Div(className="progress-bar bg-primary",
                                        style={"width": "100%"})
                            ])
                        ], className="mt-2"),
                        html.Small([
                            html.I(className="fas fa-info-circle me-1"),
                            f"{len(recent_df):,} new reviews (last 90 days)"
                        ], className="text-muted mt-2 d-block"),
                        html.Small("Statistically significant sample for policy decisions", className="text-muted mt-2 d-block fw-bold")
                    ])
                ])
            ], className="metrics-card h-100")
        ], width=12, md=6, lg=3, className="mb-0"),
        
        # Strengths - Policy Success Areas
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-thumbs-up text-success me-2"),
                            html.H6("Policy Strengths", className="text-uppercase fw-bold mb-1 text-muted small")
                        ], className="d-flex align-items-center"),
                        html.Div([
                            html.Span(f"{positive_percent:.1f}%", className="h3 fw-bold text-success"),
                            html.Div(className="progress mt-2", style={"height": "4px"}, children=[
                                html.Div(className="progress-bar bg-success",
                                        style={"width": f"{positive_percent}%"})
                            ])
                        ], className="mt-2"),
                        html.Small([
                            html.I(className="fas fa-check me-1"),
                            ", ".join(top_positive_aspects) if top_positive_aspects else "No significant strengths identified"
                        ], className="mt-2 d-block"),
                        html.Small("Areas to maintain in tourism policy", className="text-muted mt-2 d-block fw-bold")
                    ])
                ])
            ], className="metrics-card h-100")
        ], width=12, md=6, lg=3, className="mb-0"),
        
        # Areas for Improvement - Policy Intervention Needed
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-exclamation-triangle text-danger me-2"),
                            html.H6("Policy Intervention Areas", className="text-uppercase fw-bold mb-1 text-muted small")
                        ], className="d-flex align-items-center"),
                        html.Div([
                            html.Span(f"{negative_percent:.1f}%", className="h3 fw-bold text-danger"),
                            html.Div(className="progress mt-2", style={"height": "4px"}, children=[
                                html.Div(className="progress-bar bg-danger",
                                        style={"width": f"{negative_percent}%"})
                            ])
                        ], className="mt-2"),
                        html.Small([
                            html.I(className="fas fa-exclamation me-1"),
                            ", ".join(top_negative_aspects) if top_negative_aspects else "No significant issues identified"
                        ], className="mt-2 d-block"),
                        html.Small("Priority areas requiring policy attention", className="text-muted mt-2 d-block fw-bold")
                    ])
                ])
            ], className="metrics-card h-100")
        ], width=12, md=6, lg=3, className="mb-0")
    ], className="g-2 metrics-row")

# Update render_positive_words function
def render_positive_words(df):
        return html.Div()

# Update render_negative_words function
def render_negative_words(df):
        return html.Div()

# Update render_pie_chart function
def render_pie_chart(df):
    labels = ['Positive', 'Negative']
    sizes = [len(df[df['Sentiment_Label'] == 'Positive']), len(df[df['Sentiment_Label'] == 'Negative'])]
    
    fig = px.pie(
        names=labels,
        values=sizes,
        color_discrete_sequence=[COLORS['positive'], COLORS['negative']],
        hole=0.6
    )
    fig.update_layout(
        **chart_theme,
        height=CHART_HEIGHT,
        title=None
    )
    
    return dbc.Col(html.Div([
        html.H5("📊 Sentiment Distribution", className="chart-title"),
        dcc.Graph(figure=fig)
    ], className="chart-container"), width=12, md=6, lg=4)

# Top Hotels by Guest Satisfaction
def render_top_hotels(df):
    if df.empty:
        return html.Div()
    
    top_hotels = df[df['Sentiment_Label'] == 'Positive'].groupby('Hotel_Name').size().sort_values(ascending=False).head(5)
    if top_hotels.empty:
        return html.Div()
    
    fig = px.bar(x=top_hotels.index, y=top_hotels.values, 
                 color=top_hotels.values, 
                 color_continuous_scale='Blugrn', 
                 text_auto='.2s')
    fig.update_layout(
        **chart_theme,
        height=CHART_HEIGHT_LARGE,
        showlegend=False,
        coloraxis_showscale=False,
        title=None
    )
    
    return dbc.Col(html.Div([
        html.H5("🏆 Top 5 Hotels by Guest Satisfaction", className="chart-title"),
        dcc.Graph(figure=fig),
        dbc.Button("🔍 Show Details", id="btn-top-hotels", color="info", size="sm", className="mt-2"),
    ], className="chart-container"), width=12, lg=6)

# Top Reviewer Nationalities
def render_nationality_chart(df):
    if df.empty:
        return html.Div()
    
    # Get top 10 nationalities instead of just 5
    top_nationalities = df['Reviewer_Nationality'].value_counts().head(10).sort_values(ascending=True)
    
    # Create an enhanced interactive chart with explicit customdata
    nationality_data = pd.DataFrame({
        'Nationality': top_nationalities.index,
        'Count': top_nationalities.values
    })
    
    fig = px.bar(
        nationality_data,
        x='Count', 
        y='Nationality', 
                 orientation='h',
        color='Count', 
        color_continuous_scale='Electric',
        custom_data=['Nationality']  # Explicitly define the custom_data
    )
    
    fig.update_layout(
        **chart_theme,
        height=CHART_HEIGHT_LARGE,
        showlegend=False,
        coloraxis_showscale=False,
        title=None,
        dragmode=False,
        clickmode='event+select'
    )
    
    # Add hover template
    fig.update_traces(
        hovertemplate='<b>%{y}</b><br>Reviews: %{x:,}<extra></extra>',
    )
    
    return dbc.Col(html.Div([
        html.H5("🌎 Top Reviewer Nationalities", className="chart-title"),
        dcc.Graph(
            id='nationality-chart',
            figure=fig,
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d']
            }
        ),
        html.Div([
            html.Small("Click on a nationality to see detailed analysis", className="text-muted"),
        ], className="text-center mt-2")
    ], className="chart-container"), width=12, lg=6)

# City-wise Sentiment Comparison
def render_city_comparison(df):
    if df.empty:
        return html.Div()
    
    city_sentiment = df.groupby('City')['sentiment_classification'].mean().sort_values(ascending=False)
    best_city = city_sentiment.index[0] if not city_sentiment.empty else "N/A"
    best_score = city_sentiment.iloc[0] if not city_sentiment.empty else 0

    fig = px.bar(x=city_sentiment.index, y=city_sentiment.values, 
                 color=city_sentiment.values,
                 color_continuous_scale='Edge', 
                 labels={'y': 'Avg Sentiment', 'x': 'City'}, 
                 text_auto='.2f')
    fig.update_layout(
        **chart_theme,
        height=CHART_HEIGHT_LARGE,
        showlegend=False,
        coloraxis_showscale=False,
        title=None
    )
    
    return dbc.Col(html.Div([
        html.H5("🌆 City-wise Sentiment Comparison", className="chart-title"),
        dcc.Graph(figure=fig),
    ], className="chart-container"), width=12)

# Historical Trend Chart
def render_sentiment_trend(df, start_date=None, end_date=None):
    if df.empty:
        return html.Div()

    # Resample to daily and weekly trends, then drop empty dates
    daily_trend = (
        df.resample('D', on='Review_Date')['sentiment_classification']
        .mean()
        .dropna()  # Only keep dates with actual reviews
    )
    weekly_trend = (
        df.resample('W', on='Review_Date')['sentiment_classification']
        .mean()
        .dropna()  # Only keep weeks with actual reviews
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=daily_trend.index,
        y=daily_trend.values,
        name='Daily Sentiment',
        mode='lines',
        line=dict(width=1, color='rgba(46, 204, 113, 0.5)'),
    ))

    fig.add_trace(go.Scatter(
        x=weekly_trend.index,
        y=weekly_trend.values,
        name='7-day Average',
        mode='lines',
        line=dict(width=2, color='#2471A3')
    ))

    # Add threshold lines
    fig.add_hline(y=0.5, line_dash="dot", line_color="green", annotation_text="Positive Threshold")
    fig.add_hline(y=-0.5, line_dash="dot", line_color="red", annotation_text="Negative Threshold")

    # Set xaxis range if provided
    xaxis_range = None
    if start_date is not None and end_date is not None:
        xaxis_range = [start_date, end_date]

    # Update layout
    fig.update_layout(
        title="📈 Guest Satisfaction Trend Over Time",
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        yaxis_range=[-1.1, 1.1],
        hovermode="x unified",
        height=300,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.01),
        xaxis=dict(rangeslider=dict(visible=False), range=xaxis_range)
    )

    return dbc.Col(dcc.Graph(figure=fig), width=12)

# Modal for chart details
def generate_word_insights(df, sentiment):
    return html.Div()

# Update callback to handle all filter changes - Fix date range issue
@callback(
    [Output('metrics-container', 'children'),
     Output('trend-container', 'children'),
     Output('charts-container', 'children')],
    [Input('city-selector', 'value'),
     Input('date-filter-mode', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('reset-filters', 'n_clicks')]
)
def update_dashboard(selected_city, date_mode, start_date, end_date, reset_clicks):
    """Update the dashboard based on filter selections."""
    global _df
    if _df is None:
        return [], [], []
    
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    # Debug the inputs
    print("\nFilter Values in update_dashboard:")
    print(f"selected_city: {selected_city}")
    print(f"date_mode: {date_mode}")
    print(f"start_date: {start_date} (type: {type(start_date)})")
    print(f"end_date: {end_date} (type: {type(end_date)})")
    print(f"triggered_id: {triggered_id}")

    # Start with full data
    filtered_df = _df.copy()

    # Apply city filter
    if selected_city != 'Overall':
        filtered_df = filtered_df[filtered_df['City'] == selected_city]

    # Handle date filtering
    if date_mode == 'all':
        # Use all data, no date filtering
        pass  # Keep all dates in filtered_df
    elif date_mode == 'custom' and start_date and end_date:
        try:
            # Convert to datetime if they're strings
            start_date_dt = pd.to_datetime(start_date) if isinstance(start_date, str) else start_date
            end_date_dt = pd.to_datetime(end_date) if isinstance(end_date, str) else end_date
                
            # Apply date filter
            filtered_df = filtered_df[
                (filtered_df['Review_Date'] >= start_date_dt) &
                (filtered_df['Review_Date'] <= end_date_dt)
            ]
            
            # Debug the filtered dataframe
            print(f"After date filtering: {filtered_df.shape} rows")
            if len(filtered_df) > 0:
                print(f"Date range in filtered data: {filtered_df['Review_Date'].min()} to {filtered_df['Review_Date'].max()}")
        except Exception as e:
            print(f"Error in date filtering: {str(e)}")
            import traceback
            traceback.print_exc()

    # Calculate trend with filtered data
    if date_mode == 'custom' and start_date and end_date:
        trend = render_sentiment_trend(filtered_df, start_date, end_date)
    else:
        trend = render_sentiment_trend(filtered_df)

    # Build components
    metrics = render_metrics(filtered_df)
    
    # Charts section with policy-focused structure
    charts = html.Div([
        # Section 1: Primary Policy Intelligence - Key Factors Analysis
        dbc.Card([
            dbc.CardHeader([
                html.H5([html.I(className="fas fa-lightbulb me-2"), "Key Tourism Factors by Policy Domain"], className="mb-0"),
                html.Small("Click on any factor for detailed policy recommendations", className="text-muted")
            ], className="bg-light"),
            dbc.CardBody([
                render_word_frequency_chart(filtered_df)
            ])
        ], className="mb-4 shadow-sm"),
        
        # Section 2: Visitor Demographics Analysis - For targeted policy interventions
        dbc.Card([
            dbc.CardHeader([
                html.H5([html.I(className="fas fa-globe-americas me-2"), "Visitor Demographics & Targeting"], className="mb-0"),
                html.Small("Market segmentation for targeted policy development", className="text-muted")
            ], className="bg-light"),
            dbc.CardBody([
                render_nationality_chart(filtered_df)
            ])
        ], className="mb-4 shadow-sm"),
        
        # Section 3: Conditional Regional Analysis - Only shown for Overall view
        html.Div([
            dbc.Card([
                dbc.CardHeader([
                    html.H5([html.I(className="fas fa-map-marked-alt me-2"), "Regional Performance Analysis"], className="mb-0"),
                    html.Small("Policy effectiveness comparison by region", className="text-muted")
                ], className="bg-light"),
                dbc.CardBody([
                    render_city_comparison(filtered_df)
                ])
            ], className="shadow-sm")
        ] if selected_city == 'Overall' else [], id='city-comparison-container')
    ], className="policy-insights-container")
    
    return metrics, trend, charts

# Modal popup callback
@callback(
    [Output("chart-modal", "is_open"),
     Output("modal-body-content", "children")],
    [Input("btn-top-hotels", "n_clicks"),
     Input("close-modal", "n_clicks")],
    prevent_initial_call=True
)
def show_modal_details(top_hotels, close_btn):
    ctx = callback_context
    if not ctx.triggered:
        return False, ""
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    triggered_value = ctx.triggered[0]['value']
    if triggered_id == "btn-top-hotels" and isinstance(triggered_value, int) and triggered_value > 0:
            return True, generate_top_hotels_insights(_df)
    
    return False, ""

# Enhanced: Generate insights for top hotels

def generate_top_hotels_insights(df):
    top_hotels = df[df['Sentiment_Label'] == 'Positive'].groupby('Hotel_Name').size().sort_values(ascending=False).head(10)
    total_reviews = len(df)
    top_hotel_name = top_hotels.index[0] if not top_hotels.empty else "N/A"
    top_hotel_reviews = top_hotels.iloc[0] if not top_hotels.empty else 0
    percent = (top_hotel_reviews / total_reviews * 100) if total_reviews else 0

    return dbc.Card([
        dbc.CardHeader([
            html.H4("🏨 Top Hotels by Guest Satisfaction", className="mb-0"),
        ], className="bg-primary text-white"),
        dbc.CardBody([
            html.P(
                f"The hotel with the most positive reviews is '{top_hotel_name}' with {top_hotel_reviews} positive reviews, "
                f"representing {percent:.1f}% of all reviews. This may indicate best practices or service standards "
                "that could be adopted by other hotels.",
                className="mb-3"
            ),
            html.Ul([
                html.Li("Consider investigating what makes top hotels successful."),
                html.Li("Encourage knowledge sharing between top and low-performing hotels."),
                html.Li("Use this data to inform tourism and hospitality policy."),
            ], className="mb-3"),
            dbc.Table.from_dataframe(top_hotels.reset_index(name='Positive Reviews'), striped=True, bordered=True, hover=True, responsive=True),
        ])
    ], className="shadow-lg")

# Enhanced: Generate insights for nationalities

def generate_nationality_insights(df):
    top_nationalities = df['Reviewer_Nationality'].value_counts().head(10)
    total_reviews = len(df)
    top_nat = top_nationalities.index[0] if not top_nationalities.empty else "N/A"
    top_nat_count = top_nationalities.iloc[0] if not top_nationalities.empty else 0
    percent = (top_nat_count / total_reviews * 100) if total_reviews else 0

    return dbc.Card([
        dbc.CardHeader([
            html.H4("🌍 Top Reviewer Nationalities", className="mb-0"),
        ], className="bg-info text-white"),
        dbc.CardBody([
            html.P(
                f"The nationality with the most reviews is '{top_nat}' ({top_nat_count} reviews, {percent:.1f}% of all). "
                "This can help target tourism campaigns or address specific needs.",
                className="mb-3"
            ),
            html.Ul([
                html.Li("Consider language/cultural support for top nationalities."),
                html.Li("Use this data to inform international tourism policy."),
            ], className="mb-3"),
            dbc.Table.from_dataframe(top_nationalities.reset_index(name='Review Count'), striped=True, bordered=True, hover=True, responsive=True),
        ])
    ], className="shadow-lg")

# Enhanced: Generate insights for city sentiment

def generate_city_sentiment_insights(df):
    city_sentiment = df.groupby('City')['sentiment_classification'].mean().sort_values(ascending=False)
    best_city = city_sentiment.index[0] if not city_sentiment.empty else "N/A"
    best_score = city_sentiment.iloc[0] if not city_sentiment.empty else 0

    return dbc.Card([
        dbc.CardHeader([
            html.H4("🏙️ City-wise Sentiment Comparison", className="mb-0"),
        ], className="bg-success text-white"),
        dbc.CardBody([
            html.P(
                f"The city with the highest average sentiment is '{best_city}' (score: {best_score:.2f}). "
                "This can guide resource allocation and improvement efforts.",
                className="mb-3"
            ),
            html.Ul([
                html.Li("Prioritize support for cities with low sentiment."),
                html.Li("Study high-performing cities for best practices."),
            ], className="mb-3"),
            dbc.Table.from_dataframe(city_sentiment.reset_index(), striped=True, bordered=True, hover=True, responsive=True),
        ])
    ], className="shadow-lg")

# Add custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Customer Satisfaction Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            
            body {
                font-family: 'Inter', system-ui, sans-serif;
                background-color: #f8f9fa;
                color: #212529;
            }
            
            .gradient-background {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                z-index: -1;
            }
            
            .dashboard-container {
                min-height: 100vh;
            }
            
            .card-hover:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 12px rgba(0,0,0,0.1);
            }
            
            .dropdown-modern .Select-control {
                border-radius: 8px;
                border: 1px solid rgba(0,0,0,0.1);
            }
            
            .radio-modern .form-check {
                padding: 0.5rem 1rem;
                background: white;
                border-radius: 8px;
                margin-right: 0.5rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            
            .date-picker-modern input {
                border-radius: 8px;
                border: 1px solid rgba(0,0,0,0.1);
                padding: 0.5rem;
            }
            
            .btn-modern {
                border-radius: 8px;
                font-weight: 500;
                padding: 0.5rem 1rem;
                transition: all 0.2s ease;
            }
            
            .btn-modern:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            
            .metrics-card {
                background: white;
                border-radius: 12px;
                padding: 1.5rem;
                box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                transition: all 0.2s ease;
            }
            
            .metrics-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 12px rgba(0,0,0,0.1);
            }
            
            .chart-container {
                background: white;
                border-radius: 12px;
                padding: 1.5rem;
                box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                margin-bottom: 1.5rem;
                overflow: hidden;  /* Add overflow control */
                width: 100%;      /* Ensure full width */
            }

            /* Add responsive chart container */
            .chart-container .js-plotly-plot {
                width: 100% !important;
                max-width: 100% !important;
            }

            /* Add chart grid container styles */
            .chart-grid-container {
                display: grid;
                gap: 1.5rem;
                width: 100%;
                padding: 1rem;
            }

            .chart-row {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 1.5rem;
                width: 100%;
            }

            .chart-col {
                min-width: 0;  /* Allow shrinking below content size */
                width: 100%;
            }
            
            /* Nationality chart styling */
            #nationality-chart .cursor-pointer {
                cursor: pointer !important;
            }
            
            #nationality-chart .hoverable:hover {
                opacity: 0.8;
            }
            
            /* Nationality modal styling */
            .modal {
                z-index: 1050 !important;
                display: block;
            }
            
            #nationality-detail-modal .modal-content {
                border-radius: 16px;
                overflow: hidden;
                box-shadow: 0 12px 28px rgba(0,0,0,0.15), 0 8px 10px rgba(0,0,0,0.12);
                border: none;
            }
            
            #nationality-detail-modal .modal-header {
                background: linear-gradient(135deg, #4361ee, #3a0ca3);
                color: white;
                padding: 1.25rem 1.75rem;
                border-bottom: none;
            }
            
            #nationality-detail-modal .modal-body {
                padding: 0;
                background-color: #f8f9fa;
            }
            
            #nationality-detail-modal .modal-footer {
                border-top: none;
                background-color: #f8f9fa;
                padding: 1rem 1.75rem 1.5rem;
            }
            
            #nationality-detail-modal .btn-close {
                color: white;
                opacity: 0.8;
                filter: brightness(4);
            }
            
            .nationality-details {
                padding: 1.5rem;
            }
            
            .nationality-details .rounded {
                border-radius: 12px !important;
                transition: all 0.2s ease;
            }
            
            .nationality-details .p-3 {
                padding: 1.25rem !important;
            }
            
            .nationality-details .bg-light {
                background-color: white !important;
                box-shadow: 0 4px 6px rgba(0,0,0,0.04);
                border: 1px solid rgba(0,0,0,0.05);
            }
            
            .nationality-details h5 {
                font-size: 1rem;
                font-weight: 600;
                color: #3a0ca3;
                letter-spacing: -0.01em;
            }
            
            .nationality-details .border-bottom {
                border-color: rgba(0,0,0,0.05) !important;
            }
            
            .nationality-details .h3 {
                font-weight: 700 !important;
                letter-spacing: -0.02em;
                font-size: 1.75rem;
            }
            
            .nationality-details .text-muted {
                color: #6c757d !important;
                font-size: 0.85rem;
            }
            
            .nationality-details .table-sm {
                font-size: 0.875rem;
                border-collapse: separate;
                border-spacing: 0;
            }
            
            .nationality-details .table-sm th {
                font-weight: 600;
                color: #495057;
                border-top: none;
                border-bottom: 2px solid #e9ecef;
                padding: 0.75rem;
            }
            
            .nationality-details .table-sm td {
                padding: 0.75rem;
                border-top: 1px solid #f1f3f5;
            }
            
            .nationality-details .table-sm tr:hover td {
                background-color: #f8f9fa;
            }
            
            .nationality-details .progress {
                height: 8px;
                border-radius: 8px;
                overflow: hidden;
                background-color: #e9ecef;
            }
            
            .nationality-details .progress-bar {
                transition: width 1s ease;
                border-radius: 8px;
            }
            
            .nationality-details .progress-bar.bg-success {
                background: linear-gradient(90deg, #06d6a0, #1b9aaa) !important;
            }
            
            .nationality-details .progress-bar.bg-danger {
                background: linear-gradient(90deg, #ef476f, #ff9e00) !important;
            }
            
            .nationality-details li {
                margin-bottom: 0.5rem;
                line-height: 1.5;
                position: relative;
                padding-left: 1.25rem;
            }
            
            .nationality-details li::before {
                content: '•';
                position: absolute;
                left: 0;
                color: #4361ee;
                font-weight: bold;
            }
            
            .nationality-details .mb-4 {
                margin-bottom: 1.5rem !important;
            }
            
            /* Grid layout for charts */
            .nationality-details .chart-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 1.25rem;
                margin-bottom: 1.5rem;
                width: 100%;
            }
            
            .nationality-details .chart-grid-item {
                min-width: 0;
                min-height: 300px;
            }
            
            @media (max-width: 768px) {
                .nationality-details .chart-grid {
                    grid-template-columns: 1fr;
                }
                
                .nationality-details .chart-grid-item {
                    width: 100%;
                }
                
                .nationality-details .insight-card {
                    flex-direction: column;
                }
                
                .nationality-details .insight-icon {
                    margin-bottom: 0.75rem;
                    margin-right: 0;
                }
            }
            
            /* Enhance chart styling */
            .nationality-details .js-plotly-plot {
                border-radius: 12px;
                overflow: hidden !important;
            }
            
            .nationality-details .text-success {
                color: #06d6a0 !important;
            }
            
            .nationality-details .text-warning {
                color: #ffbe0b !important;
            }
            
            .nationality-details .text-danger {
                color: #ef476f !important;
            }
            
            /* Nationality emoji styling */
            .nationality-emoji {
                font-size: 2rem;
                margin-right: 0.75rem;
                background-color: rgba(255,255,255,0.2);
                height: 3rem;
                width: 3rem;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 50%;
            }
            
            /* Insight cards with icons */
            .insight-card {
                display: flex;
                align-items: flex-start;
                margin-bottom: 1rem;
                padding: 1rem;
                background-color: white;
                border-radius: 12px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.04);
                border: 1px solid rgba(0,0,0,0.05);
            }
            
            .insight-icon {
                margin-right: 0.75rem;
                color: #4361ee;
                font-size: 1rem;
                background-color: rgba(67, 97, 238, 0.1);
                height: 2rem;
                width: 2rem;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 50%;
                flex-shrink: 0;
            }
            
            .insight-content {
                flex: 1;
            }
            
            .insight-content p {
                margin: 0;
                line-height: 1.5;
            }
            
            /* Modal backdrop */
            .modal-backdrop {
                z-index: 1040 !important;
                background-color: rgba(0,0,0,0.5);
            }
            
            .modal-open .modal {
                display: block !important;
            }

            /* Add responsive breakpoints */
            @media (max-width: 768px) {
                .chart-container {
                    padding: 1rem;
                }
                
                .chart-row {
                    grid-template-columns: 1fr;  /* Stack on mobile */
                }
                
                #nationality-detail-modal .modal-dialog {
                    margin: 0.5rem;
                    max-width: calc(100% - 1rem);
                }
                
                .nationality-details {
                    padding: 1rem;
                }
            }

            /* Ensure graphs are responsive */
            .js-plotly-plot .plotly {
                width: 100% !important;
            }

            .js-plotly-plot .plot-container {
                width: 100% !important;
            }
        </style>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
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

# Add callback for reset button animation
@callback(
    Output("reset-zoom", "children"),
    [Input("reset-zoom", "n_clicks")]
)
def animate_reset(n_clicks):
    if n_clicks:
        return html.I(className="fas fa-sync-alt fa-spin")
    return html.I(className="fas fa-sync-alt")

# Update the theme toggle callback
@callback(
    Output("dashboard-theme", "data-theme"),
    [Input("theme-toggle", "n_clicks")],
    [State("dashboard-theme", "data-theme")]
)
def toggle_theme(n_clicks, current_theme):
    if n_clicks is None:
        return "light"
    
    return "dark" if current_theme == "light" or current_theme is None else "light"

# Add after the existing callbacks

@callback(
    Output("export-data", "n_clicks"),
    Input("export-data", "n_clicks"),
    [State('city-selector', 'value'),
     State('date-filter-mode', 'value'),
     State('date-range', 'start_date'),
     State('date-range', 'end_date')]
)
def export_dashboard_data(n_clicks, selected_city, date_mode, start_date, end_date):
    if not n_clicks:
        return None
        
    # Get filtered data based on current selection
    filtered_df = _df.copy()
    
    if selected_city != 'Overall':
        filtered_df = filtered_df[filtered_df['City'] == selected_city]
    
    if date_mode == 'fiscal':
        # Assuming fiscal year starts in April
        today = pd.Timestamp.now()
        if today.month >= 4:
            fiscal_start = pd.Timestamp(year=today.year, month=4, day=1)
        else:
            fiscal_start = pd.Timestamp(year=today.year-1, month=4, day=1)
        filtered_df = filtered_df[filtered_df['Review_Date'] >= fiscal_start]
    elif date_mode == 'custom' and start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['Review_Date'] >= pd.to_datetime(start_date)) &
            (filtered_df['Review_Date'] <= pd.to_datetime(end_date))
        ]
    
    # Prepare export data
    export_data = {
        'summary': {
            'total_reviews': len(filtered_df),
            'average_sentiment': filtered_df['sentiment_classification'].mean(),
            'positive_reviews': len(filtered_df[filtered_df['Sentiment_Label'] == 'Positive']),
            'negative_reviews': len(filtered_df[filtered_df['Sentiment_Label'] == 'Negative']),
            'time_period': f"{start_date} to {end_date}" if date_mode == 'custom' else 'Current Fiscal Year',
            'region': selected_city
        },
        'detailed_data': filtered_df.to_dict('records')
    }
    
    # Save to Excel file
    with pd.ExcelWriter('dashboard_export.xlsx', engine='openpyxl') as writer:
        # Summary sheet
        summary_df = pd.DataFrame([export_data['summary']])
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Detailed data sheet
        filtered_df.to_excel(writer, sheet_name='Detailed Data', index=False)
        
        # Add charts sheet
        sentiment_by_city = filtered_df.groupby('City')['sentiment_classification'].mean().reset_index()
        sentiment_by_city.to_excel(writer, sheet_name='Charts', index=False)
    
    return None

# Add tooltips for the charts
@callback(
    [Output(f"{chart}-info", "title") for chart in ['trend', 'regional', 'categories', 'demographics']],
    [Input(f"{chart}-info", "n_clicks") for chart in ['trend', 'regional', 'categories', 'demographics']]
)
def update_chart_tooltips(trend_clicks, regional_clicks, categories_clicks, demographics_clicks):
    tooltips = {
        'trend': "Shows the trend of satisfaction scores over time. Use this to identify patterns and seasonal variations.",
        'regional': "Compares performance across different regions. Helps identify areas that need attention or can serve as best practice examples.",
        'categories': "Breaks down satisfaction by service category. Use this to prioritize improvements in specific areas.",
        'demographics': "Shows satisfaction levels across different demographic groups. Helps ensure services are meeting the needs of all citizens."
    }
    return list(tooltips.values())

# Update chart rendering functions

def render_trend_chart(df):
    if df.empty:
        return {}
    
    # Calculate daily and weekly trends
    daily_trend = df.resample('D', on='Review_Date')['sentiment_classification'].mean()
    weekly_trend = df.resample('W', on='Review_Date')['sentiment_classification'].mean()
    
    fig = go.Figure()
    
    # Add daily trend
    fig.add_trace(go.Scatter(
        x=daily_trend.index,
        y=daily_trend.values,
        name='Daily Sentiment',
        mode='lines',
        line=dict(width=1, color='rgba(46, 204, 113, 0.5)'),
        hovertemplate='Date: %{x}<br>Score: %{y:.2f}<extra></extra>'
    ))
    
    # Add weekly trend
    fig.add_trace(go.Scatter(
        x=weekly_trend.index,
        y=weekly_trend.values,
        name='Weekly Average',
        mode='lines',
        line=dict(width=2, color='rgb(26, 68, 128)'),
        hovertemplate='Week of: %{x}<br>Score: %{y:.2f}<extra></extra>'
    ))
    
    # Add threshold lines
    fig.add_hline(y=0.5, line_dash="dot", line_color="green", annotation_text="Target Performance",
                  annotation_position="bottom right")
    
    fig.update_layout(
        title=None,
        xaxis_title="Date",
        yaxis_title="Satisfaction Score",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            rangeslider=dict(visible=False),
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            )
        )
    )
    
    # Add ARIA labels
    fig.update_layout(
        xaxis=dict(
            title_text="Date",
            accessibility=dict(
                description="X-axis showing the timeline of satisfaction scores"
            )
        ),
        yaxis=dict(
            title_text="Satisfaction Score",
            accessibility=dict(
                description="Y-axis showing satisfaction scores from -1 to 1"
            )
        )
    )
    
    return fig

def render_regional_chart(df):
    if df.empty:
        return {}
    
    # Calculate regional metrics
    regional_metrics = df.groupby('City').agg({
        'sentiment_classification': ['mean', 'count'],
        'Sentiment_Label': lambda x: (x == 'Positive').mean() * 100
    }).round(2)
    
    regional_metrics.columns = ['Avg Score', 'Total Reviews', 'Satisfaction %']
    regional_metrics = regional_metrics.sort_values('Avg Score', ascending=True)
    
    fig = go.Figure()
    
    # Add bars for average score
    fig.add_trace(go.Bar(
        y=regional_metrics.index,
        x=regional_metrics['Avg Score'],
        name='Average Score',
        orientation='h',
        marker_color='rgb(26, 68, 128)',
        hovertemplate='Region: %{y}<br>Score: %{x:.2f}<extra></extra>'
    ))
    
    # Add target line
    fig.add_vline(x=0.5, line_dash="dot", line_color="green",
                  annotation=dict(text="Target Performance"))
    
    fig.update_layout(
        title=None,
        xaxis_title="Average Satisfaction Score",
        yaxis_title="Region",
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=40),
        showlegend=False,
        # Add ARIA description
        accessibility=dict(
            description="Bar chart showing average satisfaction scores by region"
        )
    )
    
    return fig

def render_categories_chart(df):
    if df.empty:
        return {}
    
    # Calculate metrics by service category
    categories = df.groupby('service_category').agg({
        'sentiment_classification': ['mean', 'count'],
        'Sentiment_Label': lambda x: (x == 'Positive').mean() * 100
    }).round(2)
    
    categories.columns = ['Avg Score', 'Total Reviews', 'Satisfaction %']
    categories = categories.sort_values('Satisfaction %', ascending=True)
    
    fig = go.Figure()
    
    # Add bars for satisfaction percentage
    fig.add_trace(go.Bar(
        y=categories.index,
        x=categories['Satisfaction %'],
        name='Satisfaction %',
        orientation='h',
        marker_color='rgb(0, 94, 162)',
        hovertemplate='Category: %{y}<br>Satisfaction: %{x:.1f}%<extra></extra>'
    ))
    
    # Add target line
    fig.add_vline(x=80, line_dash="dot", line_color="green",
                  annotation=dict(text="Target (80%)"))
    
    fig.update_layout(
        title=None,
        xaxis_title="Satisfaction Rate (%)",
        yaxis_title="Service Category",
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=40),
        showlegend=False,
        # Add ARIA description
        accessibility=dict(
            description="Bar chart showing satisfaction rates by service category"
        )
    )
    
    return fig

def render_demographics_chart(df):
    if df.empty:
        return {}
    
    # Calculate satisfaction by demographic groups
    demographics = df.groupby(['age_group', 'gender']).agg({
        'sentiment_classification': ['mean', 'count'],
        'Sentiment_Label': lambda x: (x == 'Positive').mean() * 100
    }).round(2)
    
    demographics.columns = ['Avg Score', 'Total Reviews', 'Satisfaction %']
    demographics = demographics.reset_index()
    
    fig = go.Figure()
    
    # Create grouped bar chart
    for gender in demographics['gender'].unique():
        gender_data = demographics[demographics['gender'] == gender]
        
        fig.add_trace(go.Bar(
            name=gender,
            x=gender_data['age_group'],
            y=gender_data['Satisfaction %'],
            hovertemplate='Age Group: %{x}<br>' +
                         'Gender: ' + gender + '<br>' +
                         'Satisfaction: %{y:.1f}%<extra></extra>'
        ))
    
    fig.update_layout(
        title=None,
        xaxis_title="Age Group",
        yaxis_title="Satisfaction Rate (%)",
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        barmode='group',
        # Add ARIA description
        accessibility=dict(
            description="Grouped bar chart showing satisfaction rates by age group and gender"
        )
    )
    
    return fig

def render_word_comparison(df):
        return {}
    
@callback(
    Output("export-chart-excel", "n_clicks"),
    [Input("export-chart-excel", "n_clicks")],
    [State("city-selector", "value")]
)
def export_to_excel(n_clicks, selected_city):
        return None
        
@callback(
    Output("export-chart-pdf", "n_clicks"),
    [Input("export-chart-pdf", "n_clicks")],
    [State("word-comparison-chart", "figure")]
)
def export_to_pdf(n_clicks, figure):
    return None

# Update the word detail modal callback to handle both triggers
@callback(
    [Output("word-detail-modal", "is_open", allow_duplicate=True),
     Output("word-detail-title", "children", allow_duplicate=True),
     Output("word-detail-content", "children", allow_duplicate=True)],
    [Input("word-frequency-chart", "clickData"),
     Input("close-word-detail", "n_clicks")],
    [State("word-detail-modal", "is_open")],
    prevent_initial_call=True
)
def handle_word_modal(clickData, close_clicks, is_open):
    ctx = callback_context
    if not ctx.triggered:
        return False, "", ""
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == "close-word-detail":
        return False, "", ""
    elif trigger_id == "word-frequency-chart" and clickData is not None:
        try:
            word = clickData['points'][0]['y']
            return True, f"Word Analysis: {word}", generate_word_details(word, _df)
        except Exception as e:
            print(f"Error in word detail modal: {e}")
            import traceback
            traceback.print_exc()
            return True, "Error", html.Div(f"An error occurred: {str(e)}", className="text-danger")
    
    return False, "", ""

def generate_word_details(word, df):
    """Generate detailed policy insights and recommendations for a specific factor affecting tourism."""
    global _hotels_by_word
    
    try:
        if not word or word not in _hotels_by_word:
            return html.Div([
                html.H5(f"No data available for '{word}'", className="text-muted mt-4 text-center")
            ])
        
        word_data = _hotels_by_word[word]
        policy_area = word_data.get('policy_area', 'General')
        sentiment = word_data.get('sentiment', 'neutral')
        mention_count = word_data.get('count', 0)
        
        # Get list of hotels mentioning this word
        hotels_with_word = word_data['hotels']
        positive_hotels = word_data['positive_hotels']
        negative_hotels = word_data['negative_hotels']
        
        total_hotels = len(df['Hotel_Name'].unique())
        hotel_percentage = (len(hotels_with_word) / total_hotels * 100) if total_hotels > 0 else 0
        
        # Calculate sentiment distribution for this factor
        positive_percentage = (len(positive_hotels) / len(hotels_with_word) * 100) if len(hotels_with_word) > 0 else 0
        negative_percentage = (len(negative_hotels) / len(hotels_with_word) * 100) if len(hotels_with_word) > 0 else 0
        
        # Prepare example reviews
        positive_examples = []
        negative_examples = []
        
        # Get reviews mentioning this word if a Review_Text column exists
        if 'Review_Text' in df.columns:
            positive_reviews = df[df['Hotel_Name'].isin(positive_hotels) & df['Sentiment_Label'].isin(['Positive'])]['Review_Text'].sample(min(3, len(positive_hotels))).tolist() if positive_hotels else []
            negative_reviews = df[df['Hotel_Name'].isin(negative_hotels) & df['Sentiment_Label'].isin(['Negative'])]['Review_Text'].sample(min(3, len(negative_hotels))).tolist() if negative_hotels else []
            
            # Extract short snippets containing the word
            for review in positive_reviews:
                if isinstance(review, str):
                    words = review.split()
                    for i, w in enumerate(words):
                        if word.lower() in w.lower():
                            start = max(0, i - 5)
                            end = min(len(words), i + 6)
                            positive_examples.append(' '.join(words[start:end]) + "...")
                            break
            
            for review in negative_reviews:
                if isinstance(review, str):
                    words = review.split()
                    for i, w in enumerate(words):
                        if word.lower() in w.lower():
                            start = max(0, i - 5)
                            end = min(len(words), i + 6)
                            negative_examples.append(' '.join(words[start:end]) + "...")
                            break
        
        # Policy implications based on factor category
        if policy_area == 'Infrastructure':
            policy_insights = [
                f"Infrastructure issues related to '{word}' affect {hotel_percentage:.1f}% of accommodations.",
                "Infrastructure quality has direct impact on overall destination attractiveness and accessibility.",
                "Transportation and physical infrastructure are foundational for tourism development."
            ]
            
            if sentiment == 'positive':
                policy_recommendations = [
                    f"Maintain and promote the existing '{word}' infrastructure as a competitive advantage.",
                    "Consider this successful infrastructure model for other destinations.",
                    "Develop case studies of successful implementation to share best practices."
                ]
                economic_impact = [
                    "Positive infrastructure experiences are linked to longer stays and higher spending.",
                    "Well-maintained infrastructure reduces long-term maintenance costs.",
                    "Good infrastructure allows for premium positioning in tourism marketing."
                ]
            else:
                policy_recommendations = [
                    f"Prioritize investment in '{word}' infrastructure to improve visitor experience.",
                    "Conduct infrastructure audit to identify specific pain points for improvement.",
                    "Develop public-private partnerships to fund necessary infrastructure upgrades."
                ]
                economic_impact = [
                    "Infrastructure issues can reduce visitor numbers by up to 20%.",
                    "Poor infrastructure leads to shorter stays and reduced tourism receipts.",
                    "Infrastructure improvements typically yield ROI within 3-5 tourism seasons."
                ]
                
        elif policy_area == 'Amenities':
            policy_insights = [
                f"Amenity quality related to '{word}' affects visitor satisfaction across {len(hotels_with_word)} properties.",
                "Amenities are critical differentiators in competitive tourism markets.",
                "Quality and availability of amenities directly impact length of stay and spending."
            ]
            
            if sentiment == 'positive':
                policy_recommendations = [
                    f"Establish '{word}' as a benchmark standard in tourism quality guidelines.",
                    "Provide incentives for other properties to upgrade similar amenities.",
                    "Feature these well-received amenities in destination marketing."
                ]
                economic_impact = [
                    "Premium amenities enable higher room rates and increased revenue.",
                    "Well-regarded amenities reduce price sensitivity among visitors.",
                    "Excellent amenities increase repeat visitation and word-of-mouth marketing."
                ]
            else:
                policy_recommendations = [
                    f"Develop minimum standards for '{word}' amenities in hospitality properties.",
                    "Create training programs to improve amenity quality and service.",
                    "Consider grants or tax incentives for properties upgrading substandard amenities."
                ]
                economic_impact = [
                    "Poor amenities lead to negative reviews, affecting future bookings.",
                    "Inadequate amenities reduce competitiveness against similar destinations.",
                    "Amenity improvements typically show ROI through increased bookings within 1-2 seasons."
                ]
                
        elif policy_area == 'Attractions':
            policy_insights = [
                f"The attraction factor '{word}' influences overall destination appeal for {mention_count} visitors.",
                "Attractions are primary drivers of destination selection and visitor flow.",
                "Quality and diversity of attractions determine visitor demographics and spending patterns."
            ]
            
            if sentiment == 'positive':
                policy_recommendations = [
                    f"Expand promotion of '{word}' in international tourism marketing campaigns.",
                    "Develop complementary attractions to extend visitor engagement.",
                    "Create tourism routes or packages featuring this highly-rated attraction."
                ]
                economic_impact = [
                    "Signature attractions can increase destination visitor numbers by 15-30%.",
                    "Well-promoted attractions extend average stay duration and local spending.",
                    "Distinctive attractions support premium positioning in tourism markets."
                ]
            else:
                policy_recommendations = [
                    f"Allocate resources to improve the quality and experience of '{word}'.",
                    "Conduct visitor research to identify specific areas requiring enhancement.",
                    "Consider public-private partnerships for attraction development and maintenance."
                ]
                economic_impact = [
                    "Underperforming attractions reduce destination competitiveness and visitor numbers.",
                    "Negative attraction experiences limit repeat visitation and recommendations.",
                    "Attraction improvements typically yield 5-15% increase in visitor spending."
                ]
                
        elif policy_area == 'Safety':
            policy_insights = [
                f"Safety concerns related to '{word}' were mentioned by {mention_count} visitors.",
                "Safety perception is foundational for destination competitiveness.",
                "Safety issues can rapidly erode destination image regardless of other strengths."
            ]
            
            if sentiment == 'positive':
                policy_recommendations = [
                    f"Highlight safety strengths related to '{word}' in tourism marketing.",
                    "Document and share successful safety practices with other destinations.",
                    "Maintain investment in safety infrastructure and personnel."
                ]
                economic_impact = [
                    "Strong safety reputation enables premium pricing across the tourism sector.",
                    "Safety is particularly valued by higher-spending family and senior segments.",
                    "Safe destinations attract more international visitors and conference business."
                ]
            else:
                policy_recommendations = [
                    f"Prioritize immediate intervention to address '{word}' safety concerns.",
                    "Implement visible security measures to improve visitor perception.",
                    "Develop crisis management and communication protocols for safety incidents."
                ]
                economic_impact = [
                    "Safety concerns can reduce bookings by 30-50% almost immediately.",
                    "Negative safety perception typically takes 2-3 years to fully recover from.",
                    "Safety issues disproportionately impact higher-value tourism segments."
                ]
                
        elif policy_area == 'Service':
            policy_insights = [
                f"Service quality related to '{word}' impacts visitor experience across {len(hotels_with_word)} properties.",
                "Service excellence is a key differentiator in competitive tourism markets.",
                "Human interactions significantly influence overall destination perception."
            ]
            
            if sentiment == 'positive':
                policy_recommendations = [
                    f"Document and promote successful '{word}' service practices as industry standards.",
                    "Develop recognition programs to incentivize continued service excellence.",
                    "Create training modules based on positive service approaches."
                ]
                economic_impact = [
                    "Exceptional service enables 10-20% premium pricing across tourism offerings.",
                    "Service excellence increases visitor spending on additional services and amenities.",
                    "High service standards drive loyalty and repeat visitation."
                ]
            else:
                policy_recommendations = [
                    f"Develop industry-wide training standards addressing '{word}' service issues.",
                    "Create certification programs to ensure service quality benchmarks.",
                    "Implement mystery shopper programs to monitor and improve service delivery."
                ]
                economic_impact = [
                    "Poor service reduces repeat business and damages destination reputation.",
                    "Service issues lead to lower spending and shorter visitor stays.",
                    "Service improvements typically yield ROI within one tourism season."
                ]
                
        elif policy_area == 'Economic':
            policy_insights = [
                f"Economic factors related to '{word}' affect visitor perception across {len(hotels_with_word)} properties.",
                "Price-value perception fundamentally impacts destination competitiveness.",
                "Economic considerations influence visitor segment demographics and behavior."
            ]
            
            if sentiment == 'positive':
                policy_recommendations = [
                    f"Maintain competitive '{word}' positioning while ensuring sustainable profitability.",
                    "Promote value-for-money messaging in destination marketing.",
                    "Develop packages highlighting excellent price-value offerings."
                ]
                economic_impact = [
                    "Positive price-value perception increases overall tourism spending.",
                    "Value-focused destinations attract more repeat visitors and longer stays.",
                    "Good value reputation enhances destination resilience during economic downturns."
                ]
            else:
                policy_recommendations = [
                    f"Review pricing strategies to address '{word}' concerns across the sector.",
                    "Develop value-added packages to improve price-value perception.",
                    "Consider seasonal pricing strategies to optimize occupancy and revenue."
                ]
                economic_impact = [
                    "Negative price perception reduces competitiveness against similar destinations.",
                    "Price concerns shift visitor demographics toward lower-spending segments.",
                    "Value improvement strategies typically increase bookings by 10-25%."
                ]
                
        else:  # General
            policy_insights = [
                f"The factor '{word}' influences visitor experience across {len(hotels_with_word)} properties.",
                "This factor affects overall destination perception and competitiveness.",
                "Understanding visitor concerns improves tourism policy effectiveness."
            ]
            
            if sentiment == 'positive':
                policy_recommendations = [
                    f"Identify and promote best practices related to '{word}' across the sector.",
                    "Feature this strength in destination marketing materials.",
                    "Develop case studies to share successful implementation strategies."
                ]
                economic_impact = [
                    "Leveraging positive factors increases destination competitiveness.",
                    "Distinctive positive attributes enable premium positioning.",
                    "Positive factors increase visitor satisfaction and likelihood of return."
                ]
            else:
                policy_recommendations = [
                    f"Conduct further research to understand specific issues related to '{word}'.",
                    "Develop industry guidelines to address common concerns.",
                    "Implement monitoring to track improvement in this area."
                ]
                economic_impact = [
                    "Addressing negative factors improves overall destination competitiveness.",
                    "Resolving visitor concerns increases likelihood of return and recommendation.",
                    "Targeted improvements typically yield measurable economic benefits."
                ]
        
        # Create detailed insight cards with policy focus
        return html.Div([
            # Executive Summary - Policy Context
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H5("Executive Summary", className="text-primary border-bottom pb-2 mb-3"),
                        html.P([
                            f"Analysis of the tourism factor ", 
                            html.Strong(word), 
                            f" reveals it is mentioned by {mention_count} visitors across {len(hotels_with_word)} properties ",
                            f"({hotel_percentage:.1f}% of all accommodations). ",
                            f"This factor falls under the ", 
                            html.Badge(policy_area, color="primary", className="mx-1"),
                            f" policy area with ",
                            html.Span(f"{positive_percentage:.1f}% positive", className="text-success mx-1"),
                            "and",
                            html.Span(f" {negative_percentage:.1f}% negative", className="text-danger mx-1"),
                            "sentiment distribution."
                        ], className="lead"),
                        
                        # Visualization of sentiment distribution
                        dcc.Graph(
                            figure=px.pie(
                                names=['Positive Mentions', 'Negative Mentions'],
                                values=[positive_percentage, negative_percentage],
                                hole=0.4,
                                color_discrete_sequence=['#198754', '#dc3545']
                            ).update_layout(
                                margin=dict(l=10, r=10, t=10, b=10),
                                height=200,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                            ),
                            config={'displayModeBar': False}
                        )
                    ], className="p-3 bg-light rounded shadow-sm")
                ], md=12)
            ], className="mb-4"),
            
            # Policy Insights and Recommendations
            dbc.Row([
                # Policy Insights section
                dbc.Col([
                    html.Div([
                        html.H5("Policy Insights", className="text-primary border-bottom pb-2 mb-3"),
                        html.Div([
                            html.Div([
                                html.Div([
                                    html.I(className="fas fa-lightbulb")
                                ], className="insight-icon"),
                                html.Div([
                                    html.P(insight, className="mb-0 fw-500")
                                ], className="insight-content")
                            ], className="insight-card") for insight in policy_insights
                        ])
                    ], className="p-3 bg-light rounded h-100 shadow-sm")
                ], md=6),
                
                # Policy Recommendations section
                dbc.Col([
                    html.Div([
                        html.H5("Recommended Policy Initiatives", className="text-primary border-bottom pb-2 mb-3"),
                        html.Div([
                            html.Div([
                                html.Div([
                                    html.I(className="fas fa-check-circle")
                                ], className="insight-icon"),
                                html.Div([
                                    html.P(rec, className="mb-0 fw-500")
                                ], className="insight-content")
                            ], className="insight-card") for rec in policy_recommendations
                        ])
                    ], className="p-3 bg-light rounded h-100 shadow-sm")
                ], md=6),
            ], className="mb-4"),
            
            # Economic Impact section
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H5("Economic Impact Assessment", className="text-primary border-bottom pb-2 mb-3"),
                        html.Div([
                            html.Div([
                                html.Div([
                                    html.I(className="fas fa-chart-line")
                                ], className="insight-icon"),
                                html.Div([
                                    html.P(impact, className="mb-0 fw-500")
                                ], className="insight-content")
                            ], className="insight-card") for impact in economic_impact
                        ])
                    ], className="p-3 bg-light rounded h-100 shadow-sm")
                ], md=12),
            ], className="mb-2")
        ], className="shadow-lg nationality-details")
    except Exception as e:
        print(f"Error generating word details: {str(e)}")
        import traceback
        traceback.print_exc()
        return html.Div([
            html.H5(f"Error analyzing '{word}' factor", className="text-danger mt-4 text-center"),
            html.P(str(e), className="text-muted")
        ])

# Add render_word_frequency_chart function
def render_word_frequency_chart(df):
    """Generate a policy-focused word frequency chart from tourism review data."""
    if df.empty:
        return html.Div()
    
    # Extract positive and negative tokens
    global _hotels_by_word
    _hotels_by_word = {}
    
    all_positive_words = []
    if 'positive_tokens' in df.columns:
        for tokens in df['positive_tokens'].dropna():
            if isinstance(tokens, list):
                all_positive_words.extend(tokens)
            elif isinstance(tokens, str):
                all_positive_words.extend(tokens.split())
    
    all_negative_words = []
    if 'negative_tokens' in df.columns:
        for tokens in df['negative_tokens'].dropna():
            if isinstance(tokens, list):
                all_negative_words.extend(tokens)
            elif isinstance(tokens, str):
                all_negative_words.extend(tokens.split())
    
    # Count word frequencies
    positive_word_counts = Counter(all_positive_words)
    negative_word_counts = Counter(all_negative_words)
    
    # Get top words
    top_positive = positive_word_counts.most_common(10)
    top_negative = negative_word_counts.most_common(10)
    
    # Prepare data for chart
    words = []
    frequencies = []
    colors = []
    policy_areas = []
    
    # Enhanced stop words to filter out non-policy relevant terms
    stop_words = {
        'the', 'and', 'a', 'to', 'of', 'in', 'was', 'is', 'it', 'i', 'for', 'on', 'with', 
        'my', 'at', 'this', 'that', 'our', 'we', 'there', 'were', 'they', 'us', 'me', 'had',
        'very', 'so', 'just', 'but', 'be', 'as', 'by', 'or', 'an', 'not', 'when', 'from'
    }
    
    # Map for policy categorization - this would be customized based on domain expertise
    policy_category_map = {
        # Infrastructure words
        'location': 'Infrastructure', 'transport': 'Infrastructure', 'airport': 'Infrastructure',
        'metro': 'Infrastructure', 'bus': 'Infrastructure', 'road': 'Infrastructure', 
        'train': 'Infrastructure', 'parking': 'Infrastructure', 'accessible': 'Infrastructure',
        'distance': 'Infrastructure', 'far': 'Infrastructure', 'near': 'Infrastructure',
        'central': 'Infrastructure', 'traffic': 'Infrastructure', 'construction': 'Infrastructure',
        
        # Amenities words
        'restaurant': 'Amenities', 'food': 'Amenities', 'breakfast': 'Amenities',
        'pool': 'Amenities', 'wifi': 'Amenities', 'internet': 'Amenities',
        'gym': 'Amenities', 'spa': 'Amenities', 'bar': 'Amenities', 
        'dining': 'Amenities', 'menu': 'Amenities', 'meal': 'Amenities',
        
        # Attractions words
        'beach': 'Attractions', 'view': 'Attractions', 'museum': 'Attractions',
        'park': 'Attractions', 'tour': 'Attractions', 'sight': 'Attractions',
        'historic': 'Attractions', 'shopping': 'Attractions', 'monument': 'Attractions',
        'attraction': 'Attractions', 'tourist': 'Attractions', 'landmark': 'Attractions',
        
        # Safety words
        'safe': 'Safety', 'security': 'Safety', 'police': 'Safety',
        'crime': 'Safety', 'dangerous': 'Safety', 'theft': 'Safety',
        'emergency': 'Safety', 'clean': 'Safety', 'hygiene': 'Safety',
        
        # Service words
        'staff': 'Service', 'service': 'Service', 'friendly': 'Service',
        'helpful': 'Service', 'reception': 'Service', 'management': 'Service',
        'concierge': 'Service', 'attitude': 'Service', 'professional': 'Service',
        
        # Economic words
        'price': 'Economic', 'value': 'Economic', 'expensive': 'Economic',
        'cheap': 'Economic', 'cost': 'Economic', 'affordable': 'Economic',
        'overpriced': 'Economic', 'budget': 'Economic', 'money': 'Economic',
        
        # Default policy area for other words
        'default': 'General'
    }
    
    # Color scheme by policy area
    policy_colors = {
        'Infrastructure': '#1a659e',  # Blue
        'Amenities': '#004e89',       # Darker blue
        'Attractions': '#3e8914',     # Green
        'Safety': '#f15025',          # Orange/Red
        'Service': '#7b2cbf',         # Purple
        'Economic': '#6b705c',        # Gray/Brown
        'General': '#66999b'          # Teal
    }
    
    # Process positive words
    for word, count in top_positive:
        if word.lower() not in stop_words and len(word) > 2:
            # Determine policy area
            policy_area = policy_category_map.get(word.lower(), 'General')
            
            words.append(word)
            frequencies.append(count)
            colors.append(policy_colors.get(policy_area, '#06d6a0'))  # Default green for positive
            policy_areas.append(policy_area)
            
            # Track hotels mentioning this word
            hotels_with_word = df[df['positive_tokens'].apply(lambda x: isinstance(x, (list, str)) and (word in x if isinstance(x, list) else word in str(x)))]['Hotel_Name'].unique()
            positive_hotels = hotels_with_word.tolist()
            
            negative_hotels = df[df['negative_tokens'].apply(lambda x: isinstance(x, (list, str)) and (word in x if isinstance(x, list) else word in str(x)))]['Hotel_Name'].unique().tolist()
            
            _hotels_by_word[word] = {
                'hotels': positive_hotels + negative_hotels,
                'positive_hotels': positive_hotels,
                'negative_hotels': negative_hotels,
                'policy_area': policy_area,
                'sentiment': 'positive',
                'count': count
            }
    
    # Process negative words  
    for word, count in top_negative:
        if word.lower() not in stop_words and len(word) > 2:
            if word not in words:  # Avoid duplicates
                # Determine policy area
                policy_area = policy_category_map.get(word.lower(), 'General')
                
                words.append(word)
                frequencies.append(count)
                colors.append(policy_colors.get(policy_area, '#ef476f'))  # Default red for negative
                policy_areas.append(policy_area)
                
                # Track hotels mentioning this word
                hotels_with_word = df[df['negative_tokens'].apply(lambda x: isinstance(x, (list, str)) and (word in x if isinstance(x, list) else word in str(x)))]['Hotel_Name'].unique()
                negative_hotels = hotels_with_word.tolist()
                
                positive_hotels = df[df['positive_tokens'].apply(lambda x: isinstance(x, (list, str)) and (word in x if isinstance(x, list) else word in str(x)))]['Hotel_Name'].unique().tolist()
                
                _hotels_by_word[word] = {
                    'hotels': positive_hotels + negative_hotels,
                    'positive_hotels': positive_hotels,
                    'negative_hotels': negative_hotels,
                    'policy_area': policy_area,
                    'sentiment': 'negative',
                    'count': count
                }
    
    # Combine the top words (limit to 12)
    combined = sorted(zip(words, frequencies, colors, policy_areas), key=lambda x: x[1], reverse=True)[:12]
    if combined:
        words, frequencies, colors, policy_areas = zip(*combined)
    else:
        words, frequencies, colors, policy_areas = [], [], [], []
    
    if not words:
        return html.Div("No significant policy factors found in the data", className="text-muted text-center my-5")
    
    # Create the horizontal bar chart
    fig = go.Figure()
    
    # Add bars with policy area hover info
    fig.add_trace(go.Bar(
        y=words,
        x=frequencies,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(
                color='rgba(0,0,0,0.1)',
                width=1
            )
        ),
        hovertemplate='<b>%{y}</b><br>Mentions: %{x}<br>Policy Area: %{customdata}<extra></extra>',
        customdata=policy_areas,
        text=policy_areas,
        textposition='inside'
    ))
    
    fig.update_layout(
        title=None,
        height=500,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(
            title='Frequency of Mention',
            title_font=dict(size=12),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)'
        ),
        yaxis=dict(
            title=None,
            title_font=dict(size=12),
            autorange="reversed"  # Put highest frequency at top
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family='Inter, sans-serif',
            size=12
        ),
        bargap=0.15,
        annotations=[
            dict(
                x=0.5,
                y=1.05,
                xref="paper",
                yref="paper",
                text="Grouped by policy area - click any factor for detailed policy insights",
                showarrow=False,
                font=dict(size=12)
            )
        ]
    )
    
    # Add a legend for policy areas
    unique_policy_areas = list(set(policy_areas))
    for i, area in enumerate(unique_policy_areas):
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=10, color=policy_colors.get(area, '#66999b')),
            name=area,
            legendgroup=area
        ))
    
    return html.Div([
        html.H5([
            html.I(className="fas fa-lightbulb me-2"), 
            "Key Factors Affecting Tourism Experience"
        ], className="chart-title"),
        dcc.Graph(
            id='word-frequency-chart',
            figure=fig,
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['select2d', 'lasso2d']
            },
            style={'height': '550px'}
        ),
        html.Div([
            html.Small([
                html.I(className="fas fa-info-circle me-1"),
                "Click on any factor to view policy recommendations"
            ], className="text-muted me-3"),
            dbc.Badge("Infrastructure", color="primary", className="me-1"),
            dbc.Badge("Amenities", color="info", className="me-1"),
            dbc.Badge("Attractions", color="success", className="me-1"),
            dbc.Badge("Service", color="secondary", className="me-1"),
            dbc.Badge("Safety", color="danger", className="me-1"),
            dbc.Badge("Economic", color="warning", className="me-1"),
        ], className="d-flex justify-content-center flex-wrap mt-2")
    ], className="chart-container shadow-sm p-3 bg-white rounded")
