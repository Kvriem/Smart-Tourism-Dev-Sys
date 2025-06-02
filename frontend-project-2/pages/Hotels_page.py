import dash_bootstrap_components as dbc
from dash import html, dcc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def load_hotels_data():
    """Load and return hotels data from CSV"""
    try:
        df = pd.read_csv('data/data.csv')
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def create_hotels_content(city_filter="all", start_date=None, end_date=None):
    """Create the hotels page content"""
    df = load_hotels_data()
    
    if df.empty:
        return html.Div([
            html.H2("🏨 Hotels Dashboard", className="page-title"),
            html.P("No data available", className="text-muted")        ])
    
    # Filter data based on city selection
    filtered_df = df.copy()
    if city_filter and city_filter != "all":
        filtered_df = filtered_df[filtered_df['City'] == city_filter]
    
    # Apply date filters if provided
    if start_date and end_date:
        filtered_df['Review Date'] = pd.to_datetime(filtered_df['Review Date'])
        filtered_df = filtered_df[
            (filtered_df['Review Date'] >= start_date) & 
            (filtered_df['Review Date'] <= end_date)
        ]
    
    # Calculate hotel metrics
    hotel_metrics = calculate_hotel_metrics(filtered_df)
    return html.Div([
        # Hotel KPI Cards
        create_hotel_kpi_section(hotel_metrics),
        
        # Hotel Charts Section
        create_hotel_charts_section(filtered_df),
        
        # Top Hotels Section
        create_top_hotels_section(filtered_df)
    ], className="hotels-content")

def calculate_hotel_metrics(df):
    """Calculate key hotel metrics"""
    if df.empty:
        return {
            'total_hotels': 0,
            'avg_rating': 0,
            'total_reviews': 0,
            'occupancy_rate': 0
        }
    
    total_hotels = df['Hotel Name'].nunique() if 'Hotel Name' in df.columns else 0
    
    # Calculate average rating from sentiment classification
    # Assuming sentiment classification: 1 = positive, -1 = negative, 0 = neutral
    if 'sentiment classification' in df.columns:
        # Convert sentiment to rating scale (1-5)
        # Positive sentiment = 4-5, Neutral = 3, Negative = 1-2
        sentiment_to_rating = {1: 4.5, 0: 3.0, -1: 2.0}
        df['calculated_rating'] = df['sentiment classification'].map(sentiment_to_rating)
        avg_rating = df['calculated_rating'].mean()
    else:
        avg_rating = 0
        
    total_reviews = len(df)
    
    # Mock occupancy rate calculation (you can replace with actual logic)
    occupancy_rate = 78.5  # Placeholder value
    
    return {
        'total_hotels': total_hotels,
        'avg_rating': round(avg_rating, 1),
        'total_reviews': total_reviews,
        'occupancy_rate': occupancy_rate
    }

def create_hotel_kpi_section(metrics):
    """Create hotel KPI cards section"""
    return html.Div([
        html.H2([
            html.I(className="fas fa-chart-line me-2"),
            "Hotel Performance Metrics"
        ], className="section-title"),
        
        dbc.Row([
            dbc.Col([
                create_hotel_kpi_card(
                    "Total Hotels", 
                    metrics['total_hotels'], 
                    "🏨", 
                    "Active properties", 
                    "primary"
                )
            ], lg=3, md=6, sm=12),
            
            dbc.Col([
                create_hotel_kpi_card(
                    "Average Rating", 
                    f"{metrics['avg_rating']}/5", 
                    "⭐", 
                    "Guest satisfaction", 
                    "success"
                )
            ], lg=3, md=6, sm=12),
            
            dbc.Col([
                create_hotel_kpi_card(
                    "Total Reviews", 
                    f"{metrics['total_reviews']:,}", 
                    "💬", 
                    "Guest feedback", 
                    "info"
                )
            ], lg=3, md=6, sm=12),
            
            dbc.Col([
                create_hotel_kpi_card(
                    "Occupancy Rate", 
                    f"{metrics['occupancy_rate']}%", 
                    "📊", 
                    "Current period", 
                    "warning"
                )
            ], lg=3, md=6, sm=12),
        ], className="hotels-kpi-row")
    ], className="hotels-kpi-section")

def create_hotel_kpi_card(title, value, emoji, subtitle, color_class):
    """Create an individual hotel KPI card"""
    return html.Div([
        html.Div([
            html.Div([
                html.Span(emoji, className="kpi-emoji"),
                html.H3(str(value), className=f"kpi-value text-{color_class}"),
                html.H6(title, className="kpi-title"),
                html.P(subtitle, className="kpi-subtitle")
            ])
        ], className="hotel-kpi-card")
    ])

def create_hotel_charts_section(df):
    """Create hotel charts section"""
    if df.empty:
        return html.Div([
            html.H3("No data available for charts", className="text-muted text-center")
        ])
    
    # Create rating distribution chart
    rating_chart = create_rating_distribution_chart(df)
    
    # Create hotels by city chart
    city_chart = create_hotels_by_city_chart(df)
    
    return html.Div([
        html.H2([
            html.I(className="fas fa-chart-bar me-2"),
            "Hotel Analytics"
        ], className="section-title"),
          dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4([
                        html.I(className="fas fa-star me-2"),
                        "Guest Sentiment Distribution"
                    ], className="chart-title"),
                    dcc.Graph(
                        figure=rating_chart,
                        className="hotel-chart",
                        config={'displayModeBar': False}
                    )
                ], className="chart-container")
            ], lg=6, md=12),
            
            dbc.Col([
                html.Div([
                    html.H4([
                        html.I(className="fas fa-map-marker-alt me-2"),
                        "Hotels by City"
                    ], className="chart-title"),
                    dcc.Graph(
                        figure=city_chart,
                        className="hotel-chart",
                        config={'displayModeBar': False}
                    )
                ], className="chart-container")
            ], lg=6, md=12),
        ])
    ], className="hotels-charts-section")

def create_rating_distribution_chart(df):
    """Create sentiment distribution chart"""
    if 'sentiment classification' not in df.columns:
        return go.Figure().add_annotation(
            text="Sentiment data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    sentiment_counts = df['sentiment classification'].value_counts()
    sentiment_labels = {1: 'Positive', -1: 'Negative', 0: 'Neutral'}
    
    # Create labels and colors
    labels = [sentiment_labels.get(k, str(k)) for k in sentiment_counts.index]
    colors = ['#28a745' if k == 1 else '#dc3545' if k == -1 else '#ffc107' for k in sentiment_counts.index]
    fig = px.bar(
        x=labels,
        y=sentiment_counts.values,
        labels={'x': 'Sentiment', 'y': 'Number of Reviews'},
        color=sentiment_counts.values,
        color_continuous_scale='RdYlGn'
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"),
        title_font_size=16,
        showlegend=False
    )
    
    return fig

def create_hotels_by_city_chart(df):
    """Create hotels by city chart"""
    if 'Hotel Name' not in df.columns or 'City' not in df.columns:
        return go.Figure().add_annotation(
            text="City data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    hotels_by_city = df.groupby('City')['Hotel Name'].nunique().sort_values(ascending=True)
    fig = px.bar(
        x=hotels_by_city.values,
        y=hotels_by_city.index,
        orientation='h',
        labels={'x': 'Number of Hotels', 'y': 'City'},
        color=hotels_by_city.values,
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"),
        title_font_size=16,
        showlegend=False,
        height=400
    )
    
    return fig

def create_top_hotels_section(df):
    """Create top hotels section"""
    if df.empty or 'Hotel Name' not in df.columns:
        return html.Div([
            html.H3("No hotel data available", className="text-muted text-center")
        ])
    
    # Calculate average sentiment and review count per hotel
    hotel_stats = df.groupby('Hotel Name').agg({
        'sentiment classification': 'mean',
        'Reviewer Name': 'count'  # Count reviews
    }).round(2)
    
    hotel_stats.columns = ['avg_sentiment', 'review_count']
    
    # Convert sentiment to rating scale for display
    sentiment_to_rating = {1: 4.5, 0: 3.0, -1: 2.0}
    hotel_stats['avg_rating'] = hotel_stats['avg_sentiment'].apply(
        lambda x: sentiment_to_rating.get(round(x), 3.0)
    )
    
    hotel_stats = hotel_stats.sort_values('avg_sentiment', ascending=False).head(10)
    
    return html.Div([
        html.H2([
            html.I(className="fas fa-trophy me-2"),
            "Top Rated Hotels"
        ], className="section-title"),
        
        html.Div([
            create_hotel_cards(hotel_stats)
        ], className="top-hotels-grid")
    ], className="top-hotels-section")

def create_hotel_cards(hotel_stats):
    """Create hotel cards for top hotels"""
    cards = []
    
    for idx, (hotel_name, stats) in enumerate(hotel_stats.iterrows()):
        # Determine badge color based on ranking
        if idx == 0:
            badge_class = "bg-warning"
            icon = "🥇"
        elif idx == 1:
            badge_class = "bg-secondary"
            icon = "🥈"
        elif idx == 2:
            badge_class = "bg-info"
            icon = "🥉"
        else:
            badge_class = "bg-light text-dark"
            icon = "🏨"
        
        card = dbc.Col([
            html.Div([
                html.Div([
                    html.Span(f"#{idx + 1}", className=f"badge {badge_class} position-absolute top-0 start-0 m-2"),
                    html.Span(icon, style={"fontSize": "2rem"}),
                ], className="card-header-custom text-center"),
                
                html.Div([
                    html.H5(hotel_name, className="card-title text-truncate"),
                    html.Div([
                        html.Span("⭐", style={"color": "#ffd700"}),
                        html.Span(f" {stats['avg_rating']:.1f}/5.0", className="fw-bold"),
                    ], className="rating-display"),
                    html.P(f"{int(stats['review_count'])} reviews", className="text-muted small")
                ], className="card-body")
            ], className="hotel-card h-100")
        ], lg=4, md=6, sm=12, className="mb-3")
        
        cards.append(card)
    
    return dbc.Row(cards)