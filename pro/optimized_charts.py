# Optimized chart creation functions with caching for overview page
from performance_cache import cache_chart_result, cache_processed_result
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from collections import Counter
import ast

@cache_chart_result("kpi_cards")
def create_kpi_cards_cached(stats):
    """Create KPI cards with caching"""
    import dash_bootstrap_components as dbc
    from dash import html
    
    def create_kpi_card(title, value, subtitle, icon, color_class="primary"):
        return dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.Div([
                        html.Span(icon, className="kpi-icon"),
                        html.H3(value, className="kpi-value"),
                        html.P(title, className="kpi-title"),
                        html.Small(subtitle, className="kpi-subtitle")
                    ], className="kpi-content")
                ], className=f"kpi-card-inner {color_class}")
            ])
        ], className="kpi-card enhanced-kpi-card")
    
    return dbc.Row([
        dbc.Col([
            create_kpi_card(
                title="Total Reviews",
                value=f"{stats['total_reviews']:,}",
                subtitle="Filtered results",
                icon="👥",
                color_class="primary"
            )
        ], lg=3, md=6, sm=12),
        dbc.Col([
            create_kpi_card(
                title="Positive Reviews",
                value=f"{stats['positive_reviews']:,}",
                subtitle=f"{(stats['positive_reviews']/max(stats['total_reviews'], 1)*100):.1f}% of total",
                icon="😊",
                color_class="success"
            )
        ], lg=3, md=6, sm=12),
        dbc.Col([
            create_kpi_card(
                title="Negative Reviews",
                value=f"{stats['negative_reviews']:,}",
                subtitle=f"{(stats['negative_reviews']/max(stats['total_reviews'], 1)*100):.1f}% of total",
                icon="😞",
                color_class="danger"
            )
        ], lg=3, md=6, sm=12),
        dbc.Col([
            create_kpi_card(
                title="Satisfaction Rate",
                value=f"{stats['satisfaction_rate']:.1f}%",
                subtitle="Overall satisfaction",
                icon="⭐",
                color_class="warning" if stats['satisfaction_rate'] < 60 else "success"
            )
        ], lg=3, md=6, sm=12)
    ], className="kpi-cards-row mb-4")

@cache_chart_result("satisfaction_trend")
def create_satisfaction_trend_chart_cached(df, city_filter="all", start_date=None, end_date=None):
    """Create satisfaction trend chart with caching"""
    if df.empty:
        return go.Figure().add_annotation(
            text="No data available for the selected filters",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="gray")
        )
    
    try:
        # Convert Review Date and process sentiment efficiently
        df_copy = df.copy()
        df_copy['Review Date'] = pd.to_datetime(df_copy['Review Date'], errors='coerce')
        
        # Convert sentiment to standardized format
        def standardize_sentiment_fast(sentiment):
            if pd.isna(sentiment):
                return 0
            if sentiment == 1 or sentiment == "1":
                return 1
            elif sentiment == -1 or sentiment == "-1":
                return -1
            else:
                return 0
        
        df_copy['sentiment_numeric'] = df_copy['sentiment classification'].apply(standardize_sentiment_fast)
        
        # Group by month efficiently
        df_copy['Month'] = df_copy['Review Date'].dt.to_period('M')
        
        # Calculate monthly stats using vectorized operations
        monthly_stats = df_copy.groupby('Month').agg({
            'sentiment_numeric': [
                'count',
                lambda x: (x == 1).sum(),
                lambda x: (x == -1).sum(),
                lambda x: (x == 0).sum()
            ]
        }).round(2)
        
        monthly_stats.columns = ['total_reviews', 'positive_reviews', 'negative_reviews', 'neutral_reviews']
        monthly_stats['satisfaction_rate'] = (monthly_stats['positive_reviews'] / monthly_stats['total_reviews'] * 100).round(1)
        monthly_stats = monthly_stats.reset_index()
        monthly_stats['Date'] = monthly_stats['Month'].dt.to_timestamp()
        
        # Create optimized chart
        fig = go.Figure()
        
        # Modern colors
        colors = {
            'total': '#4c1d95',
            'positive': '#059669',
            'negative': '#dc2626',
            'neutral': '#6366f1'
        }
        
        # Add traces with optimized styling
        fig.add_trace(go.Scatter(
            x=monthly_stats['Date'],
            y=monthly_stats['total_reviews'],
            name='📊 Total Reviews',
            line=dict(color=colors['total'], width=3),
            mode='lines+markers',
            marker=dict(size=8, color=colors['total']),
            hovertemplate='<b>Total Reviews:</b> %{y:,.0f}<br><extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=monthly_stats['Date'],
            y=monthly_stats['positive_reviews'],
            name='😊 Positive Reviews',
            line=dict(color=colors['positive'], width=3),
            mode='lines+markers',
            marker=dict(size=8, color=colors['positive']),
            fill='tonexty',
            fillcolor='rgba(5, 150, 105, 0.1)',
            hovertemplate='<b>Positive Reviews:</b> %{y:,.0f}<br><extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=monthly_stats['Date'],
            y=monthly_stats['negative_reviews'],
            name='😞 Negative Reviews',
            line=dict(color=colors['negative'], width=3),
            mode='lines+markers',
            marker=dict(size=8, color=colors['negative']),
            hovertemplate='<b>Negative Reviews:</b> %{y:,.0f}<br><extra></extra>'
        ))
        
        # Update layout for performance
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif", size=12),
            height=400,
            margin=dict(l=50, r=50, t=50, b=50),
            hovermode='x unified',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating satisfaction trend chart: {e}")
        return go.Figure().add_annotation(
            text="Chart temporarily unavailable",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="gray")
        )

@cache_chart_result("word_frequency")
def create_word_frequency_chart_cached(df, city_filter="all", start_date=None, end_date=None):
    """Create word frequency chart with caching"""
    if df.empty:
        return go.Figure().add_annotation(
            text="No data available for word analysis",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="gray")
        )
    
    try:
        # Extract tokens efficiently
        all_tokens = []
        
        for tokens_col in ['positive tokens', 'negative tokens']:
            if tokens_col in df.columns:
                for tokens_str in df[tokens_col].dropna():
                    try:
                        if isinstance(tokens_str, str) and tokens_str.strip():
                            tokens = ast.literal_eval(tokens_str)
                            if isinstance(tokens, list):
                                all_tokens.extend([token.lower() for token in tokens if token and len(token) > 2])
                    except:
                        continue
        
        if not all_tokens:
            return go.Figure().add_annotation(
                text="No tokens found in reviews",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=16, color="gray")
            )
        
        # Get top words efficiently
        word_counts = Counter(all_tokens)
        top_words = word_counts.most_common(15)
        
        words, counts = zip(*top_words)
        
        # Create bar chart
        fig = px.bar(
            x=list(counts),
            y=list(words),
            orientation='h',
            color=list(counts),
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif", size=12),
            height=400,
            margin=dict(l=100, r=50, t=50, b=50),
            showlegend=False,
            xaxis_title="Frequency",
            yaxis_title="Keywords"
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating word frequency chart: {e}")
        return go.Figure().add_annotation(
            text="Chart temporarily unavailable",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="gray")
        )

@cache_chart_result("nationality_chart")
def create_nationality_chart_cached(df, city_filter="all", start_date=None, end_date=None):
    """Create nationality distribution chart with caching"""
    if df.empty or 'Nationality' not in df.columns:
        return go.Figure().add_annotation(
            text="No nationality data available",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="gray")
        )
    
    try:
        # Count nationalities efficiently
        nationality_counts = df['Nationality'].value_counts().head(10)
        
        # Create pie chart
        fig = px.pie(
            values=nationality_counts.values,
            names=nationality_counts.index,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif", size=12),
            height=400,
            margin=dict(l=50, r=50, t=50, b=50),
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating nationality chart: {e}")
        return go.Figure().add_annotation(
            text="Chart temporarily unavailable",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="gray")
        )

@cache_chart_result("city_satisfaction")
def create_city_satisfaction_chart_cached(df, city_filter="all", start_date=None, end_date=None):
    """Create city satisfaction chart with caching"""
    if df.empty or 'City' not in df.columns:
        return go.Figure().add_annotation(
            text="No city data available",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="gray")
        )
    
    try:
        # Calculate city satisfaction efficiently
        city_satisfaction = df.groupby('City')['sentiment classification'].agg([
            'count',
            lambda x: (x == 1).sum(),
            lambda x: (x == -1).sum()
        ]).reset_index()
        
        city_satisfaction.columns = ['City', 'total', 'positive', 'negative']
        city_satisfaction['satisfaction_rate'] = (city_satisfaction['positive'] / city_satisfaction['total'] * 100).round(1)
        
        # Filter cities with at least 10 reviews
        city_satisfaction = city_satisfaction[city_satisfaction['total'] >= 10]
        
        # Create bar chart
        fig = px.bar(
            city_satisfaction,
            x='City',
            y='satisfaction_rate',
            color='satisfaction_rate',
            color_continuous_scale='RdYlGn',
            text='satisfaction_rate'
        )
        
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif", size=12),
            height=400,
            margin=dict(l=50, r=50, t=50, b=80),
            xaxis_title="Cities",
            yaxis_title="Satisfaction Rate (%)",
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating city satisfaction chart: {e}")
        return go.Figure().add_annotation(
            text="Chart temporarily unavailable",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="gray")
        )

@cache_chart_result("tourism_economics")
def create_tourism_economics_chart_cached(df, city_filter="all", start_date=None, end_date=None):
    """Create tourism economics chart with caching"""
    if df.empty:
        return go.Figure().add_annotation(
            text="No data available for economic analysis",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="gray")
        )
    
    try:
        # Create a simple placeholder chart for economics
        fig = go.Figure()
        
        # Add sample economic indicators
        months = pd.date_range(start='2024-01-01', end='2024-12-01', freq='M')
        revenue_estimate = np.random.uniform(80, 120, len(months)) * len(df) / 1000
        
        fig.add_trace(go.Scatter(
            x=months,
            y=revenue_estimate,
            mode='lines+markers',
            name='Estimated Tourism Revenue',
            line=dict(color='#0ea5e9', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif", size=12),
            height=400,
            margin=dict(l=50, r=50, t=50, b=50),
            xaxis_title="Month",
            yaxis_title="Revenue Index",
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating economics chart: {e}")
        return go.Figure().add_annotation(
            text="Chart temporarily unavailable",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="gray")
        )

@cache_chart_result("seasonal_analytics")
def create_seasonal_analytics_chart_cached(df, city_filter="all", start_date=None, end_date=None):
    """Create seasonal analytics chart with caching"""
    if df.empty:
        return go.Figure().add_annotation(
            text="No data available for seasonal analysis",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="gray")
        )
    
    try:
        # Analyze by season efficiently
        df_copy = df.copy()
        df_copy['Review Date'] = pd.to_datetime(df_copy['Review Date'], errors='coerce')
        df_copy['Month'] = df_copy['Review Date'].dt.month
        
        # Define seasons
        season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                     3: 'Spring', 4: 'Spring', 5: 'Spring',
                     6: 'Summer', 7: 'Summer', 8: 'Summer',
                     9: 'Fall', 10: 'Fall', 11: 'Fall'}
        
        df_copy['Season'] = df_copy['Month'].map(season_map)
        
        # Calculate seasonal statistics
        seasonal_stats = df_copy.groupby('Season').agg({
            'sentiment classification': ['count', lambda x: (x == 1).sum()]
        }).reset_index()
        
        seasonal_stats.columns = ['Season', 'total_reviews', 'positive_reviews']
        seasonal_stats['satisfaction_rate'] = (seasonal_stats['positive_reviews'] / seasonal_stats['total_reviews'] * 100).round(1)
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=seasonal_stats['satisfaction_rate'],
            theta=seasonal_stats['Season'],
            fill='toself',
            name='Satisfaction Rate',
            line=dict(color='#059669', width=3),
            fillcolor='rgba(5, 150, 105, 0.1)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            font=dict(family="Inter, sans-serif", size=12),
            height=400,
            margin=dict(l=50, r=50, t=50, b=50),
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating seasonal chart: {e}")
        return go.Figure().add_annotation(
            text="Chart temporarily unavailable",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="gray")
        )
