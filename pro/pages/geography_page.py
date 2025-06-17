import dash
from dash import dcc, html, Input, Output, State, callback, clientside_callback, ctx
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from collections import Counter
import ast
import re
from database_config import load_data_from_database

def standardize_sentiment(sentiment):
    """Standardize sentiment values to ensure consistency across all calculations"""
    if pd.isnull(sentiment):
        return 0
    
    # Handle string representations
    if isinstance(sentiment, str):
        sentiment = sentiment.strip().lower()
        if sentiment in ['positive', '1', '1.0']:
            return 1
        elif sentiment in ['negative', '-1', '-1.0']:
            return -1
        elif sentiment in ['neutral', '0', '0.0']:
            return 0
        else:
            # Try to parse as number
            try:
                return int(float(sentiment))
            except (ValueError, TypeError):
                return 0
    
    # Handle numeric values
    try:
        sentiment_num = float(sentiment)
        if sentiment_num > 0:
            return 1
        elif sentiment_num < 0:
            return -1
        else:
            return 0
    except (ValueError, TypeError):
        return 0

def parse_tokens(token_str):
    """Parse token arrays from string format"""
    if pd.isnull(token_str) or token_str == '':
        return []
    
    try:
        # Handle string representation of lists
        if isinstance(token_str, str):
            # Remove extra quotes and clean the string
            token_str = token_str.strip()
            if token_str.startswith('[') and token_str.endswith(']'):
                # Use ast.literal_eval for safe evaluation
                tokens = ast.literal_eval(token_str)
                return [str(token).strip() for token in tokens if token and str(token).strip()]
            else:
                # Split by comma if it's a comma-separated string
                return [token.strip() for token in token_str.split(',') if token.strip()]
        elif isinstance(token_str, list):
            return [str(token).strip() for token in token_str if token and str(token).strip()]
        else:
            return []
    except (ValueError, SyntaxError):
        # Fallback: split by common delimiters
        if isinstance(token_str, str):
            # Remove brackets and quotes
            clean_str = re.sub(r'[\[\]\'"]', '', token_str)
            return [token.strip() for token in clean_str.split(',') if token.strip()]
        return []

def get_tokens_as_list(token_str):
    """Convert string representation of tokens back to list"""
    if pd.isnull(token_str) or token_str == '' or token_str == '[]':
        return []
    
    try:
        # If it's already a string representation of a list
        if isinstance(token_str, str) and token_str.startswith('[') and token_str.endswith(']'):
            return ast.literal_eval(token_str)
        else:
            return []
    except (ValueError, SyntaxError):
        return []

def get_geography_data():
    """Load and prepare data for geography analysis"""
    df = load_data_from_database()
    
    if df is None or df.empty:
        print("⚠️ No data available for geography analysis")
        return pd.DataFrame()
    
    # Ensure we have the required columns
    required_columns = ['City', 'sentiment classification', 'positive tokens', 'negative tokens', 'Hotel Name', 'Travel Type']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"⚠️ Missing columns for geography analysis: {missing_columns}")
        return pd.DataFrame()
    
    # Standardize sentiment values
    df['sentiment_standardized'] = df['sentiment classification'].apply(standardize_sentiment)
      # Parse token columns - convert to string to avoid unhashable type error
    try:
        df['positive_tokens_parsed'] = df['positive tokens'].apply(lambda x: str(parse_tokens(x)) if parse_tokens(x) else '[]')
        df['negative_tokens_parsed'] = df['negative tokens'].apply(lambda x: str(parse_tokens(x)) if parse_tokens(x) else '[]')
    except Exception as e:
        print(f"Warning: Error parsing tokens: {e}")
        df['positive_tokens_parsed'] = '[]'
        df['negative_tokens_parsed'] = '[]'
    
    # Filter for main Egyptian tourist cities
    main_cities = ['Hurghada', 'Sharm El Sheikh', 'Luxor', 'Aswan']
    df_filtered = df[df['City'].isin(main_cities)].copy()
    
    if df_filtered.empty:
        print("⚠️ No data found for main Egyptian cities. Using all available cities.")
        df_filtered = df.copy()
    
    print(f"📊 Geography data loaded: {len(df_filtered)} records across {df_filtered['City'].nunique()} cities")
    return df_filtered

def create_sentiment_by_city_chart(df):
    """Create sentiment distribution chart by city"""
    if df.empty:
        return go.Figure().add_annotation(
            text="No data available for sentiment analysis",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="gray")
        )
    
    # Calculate sentiment distribution by city
    sentiment_data = []
    cities = df['City'].unique()
    
    for city in cities:
        city_data = df[df['City'] == city]
        total_reviews = len(city_data)
        
        if total_reviews > 0:
            positive = len(city_data[city_data['sentiment_standardized'] == 1])
            negative = len(city_data[city_data['sentiment_standardized'] == -1])
            neutral = len(city_data[city_data['sentiment_standardized'] == 0])
            
            sentiment_data.extend([
                {'City': city, 'Sentiment': 'Positive', 'Count': positive, 'Percentage': (positive/total_reviews)*100},
                {'City': city, 'Sentiment': 'Negative', 'Count': negative, 'Percentage': (negative/total_reviews)*100},
                {'City': city, 'Sentiment': 'Neutral', 'Count': neutral, 'Percentage': (neutral/total_reviews)*100}
            ])
    
    sentiment_df = pd.DataFrame(sentiment_data)
    
    if sentiment_df.empty:
        return go.Figure().add_annotation(
            text="No sentiment data available",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="gray")
        )      # Create enhanced stacked bar chart with improved styling
    fig = px.bar(
        sentiment_df, 
        x='City', 
        y='Percentage', 
        color='Sentiment',
        title="🏙️ Sentiment Distribution by City",
        color_discrete_map={
            'Positive': '#16a34a',    # Vibrant green
            'Negative': '#dc2626',    # Vibrant red
            'Neutral': '#6366f1'      # Vibrant indigo
        },
        labels={'Percentage': 'Percentage of Reviews (%)', 'City': 'Tourist Destinations'},
        text='Percentage',
        hover_data={'Count': True}
    )
      # Enhanced styling with better colors and effects
    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='inside',
        textfont_size=12,
        textfont_color='white',
        textfont_family="Inter, sans-serif",
        marker_line=dict(width=1, color='rgba(255,255,255,0.3)')
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", size=12),
        title=dict(
            font=dict(size=20, weight='bold', color='#1f2937'),
            x=0.5,
            xanchor='center',
            pad=dict(t=20)
        ),
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="center", 
            x=0.5,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1,
            font=dict(size=11)
        ),
        height=480,
        xaxis=dict(
            title='Cities',
            title_font=dict(size=14, weight='bold', color='#374151'),
            tickfont=dict(size=12, color='#6b7280'),
            showgrid=False
        ),
        yaxis=dict(
            title='Percentage of Reviews (%)',
            title_font=dict(size=14, weight='bold', color='#374151'),
            tickfont=dict(size=12, color='#6b7280'),
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True
        ),        margin=dict(l=70, r=60, t=120, b=80),
        hovermode='x unified'
    )
    
    return fig

def create_top_tokens_by_city_chart(df, token_type='positive', top_n=5):
    """Create top tokens by city chart"""
    if df.empty:
        return go.Figure().add_annotation(
            text="No data available for token analysis",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="gray")
        )
    
    token_column = f'{token_type}_tokens_parsed'
    
    # Get top tokens by city
    city_tokens = []
    cities = df['City'].unique()
    
    for city in cities:
        city_data = df[df['City'] == city]
        all_tokens = []
        for tokens in city_data[token_column]:
            token_list = get_tokens_as_list(tokens)
            if token_list:
                all_tokens.extend(token_list)
        
        if all_tokens:
            token_counts = Counter(all_tokens)
            top_tokens = token_counts.most_common(top_n)
            
            for token, count in top_tokens:                city_tokens.append({
                    'City': city,
                    'Token': token,
                    'Count': count
                })
    
    if not city_tokens:
        return go.Figure().add_annotation(
            text=f"No {token_type} tokens found",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="gray")
        )
    
    tokens_df = pd.DataFrame(city_tokens)
      # Create enhanced grouped bar chart with better styling and larger size
    color_map = {
        'positive': ['#10B981', '#34D399', '#6EE7B7', '#A7F3D0'],  # Green tones
        'negative': ['#EF4444', '#F87171', '#FCA5A5', '#FECACA']   # Red tones
    }
    
    # Use appropriate colors based on token type
    colors = color_map.get(token_type, px.colors.qualitative.Pastel)
    
    fig = px.bar(
        tokens_df,
        x='Token',
        y='Count',
        color='City',
        title=f"� Top {token_type.capitalize()} Keywords by City - Detailed Analysis",
        barmode='group',
        color_discrete_sequence=colors,
        text='Count',
        labels={'Count': 'Frequency', 'Token': f'{token_type.title()} Keywords'},
        hover_data={'Count': True}
    )
    
    # Enhanced styling with better visibility and larger text
    fig.update_traces(
        texttemplate='<b>%{text}</b>',
        textposition='outside',
        textfont_size=12,
        textfont_color='#1F2937',
        textfont_family='Inter, sans-serif',
        marker_line=dict(width=1.5, color='rgba(255,255,255,0.4)'),        hovertemplate='<b>%{fullData.name}</b><br>' +
                     'Keyword: <b>%{x}</b><br>' +
                     'Frequency: <b>%{y}</b><br>' +
                     '<extra></extra>'
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(248,250,252,0.8)',
        paper_bgcolor='rgba(255,255,255,0.95)',
        font=dict(family="Inter, sans-serif", size=13),
        title=dict(
            font=dict(size=22, weight='bold', color='#111827'),
            x=0.5,
            xanchor='center',
            pad=dict(t=20, b=20)
        ),
        height=550,  # Increased height for better visibility
        width=1200,  # Increased width for better spacing
        xaxis=dict(
            title=f'{token_type.title()} Keywords',
            title_font=dict(size=16, weight='bold', color='#374151'),
            tickfont=dict(size=13, color='#4B5563'),
            tickangle=-35,  # Better angle for readability
            tickmode='linear',
            showgrid=True,
            gridcolor='rgba(156,163,175,0.2)',
            zeroline=False
        ),
        yaxis=dict(
            title='Frequency Count',
            title_font=dict(size=16, weight='bold', color='#374151'),
            tickfont=dict(size=13, color='#4B5563'),
            gridcolor='rgba(156,163,175,0.3)',
            zeroline=True,
            zerolinecolor='rgba(107,114,128,0.3)',
            showgrid=True
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='rgba(156,163,175,0.3)',
            borderwidth=1,
            font=dict(size=12),
            title=dict(text='<b>Cities</b>', font=dict(size=14))
        ),        margin=dict(l=80, r=160, t=120, b=140),
        showlegend=True,
        # Better spacing between bars
        bargap=0.15,
        bargroupgap=0.1
    )
    
    return fig

def create_hotel_performance_by_city_chart(df):
    """Create hotel performance by city chart"""
    if df.empty:
        return go.Figure().add_annotation(
            text="No data available for hotel performance analysis",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="gray")
        )
    
    # Calculate average sentiment by hotel and city
    hotel_performance = df.groupby(['City', 'Hotel Name']).agg({
        'sentiment_standardized': 'mean',
        'Hotel Name': 'count'
    }).rename(columns={'Hotel Name': 'review_count'}).reset_index()
    
    # Filter hotels with at least 5 reviews
    hotel_performance = hotel_performance[hotel_performance['review_count'] >= 5]
    
    if hotel_performance.empty:
        return go.Figure().add_annotation(
            text="No hotels with sufficient reviews found",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="gray")
        )
    
    # Get top 3 hotels per city
    top_hotels = []
    for city in hotel_performance['City'].unique():
        city_hotels = hotel_performance[hotel_performance['City'] == city]
        city_top = city_hotels.nlargest(3, 'sentiment_standardized')
        top_hotels.append(city_top)
    
    if top_hotels:
        top_hotels_df = pd.concat(top_hotels, ignore_index=True)
        
        # Create enhanced scatter plot
        fig = px.scatter(
            top_hotels_df,
            x='sentiment_standardized',
            y='Hotel Name',
            color='City',
            size='review_count',
            title="🏨 Top Hotel Performance by City",
            labels={
                'sentiment_standardized': 'Average Sentiment Score', 
                'review_count': 'Number of Reviews',
                'Hotel Name': 'Hotels'
            },
            hover_data=['review_count'],
            size_max=20
        )
        
        # Enhanced styling
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif", size=12),
            title=dict(
                font=dict(size=18, weight='bold', color='#1f2937'),
                x=0.5,
                xanchor='center'
            ),
            height=550,
            xaxis=dict(
                title='Sentiment Score (Higher = Better)',
                title_font=dict(size=14, weight='bold'),
                tickfont=dict(size=12),
                gridcolor='rgba(0,0,0,0.1)',
                range=[-1.1, 1.1]
            ),
            yaxis=dict(
                title_font=dict(size=14, weight='bold'),
                tickfont=dict(size=11)
            ),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(0,0,0,0.1)',
                borderwidth=1
            ),
            margin=dict(l=50, r=150, t=80, b=50)
        )
        
        # Add reference lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                      annotation_text="Neutral Sentiment", annotation_position="bottom right")
        
        return fig
    
    return go.Figure().add_annotation(
        text="No hotel performance data available",
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=16, color="gray")
    )

def create_traveler_type_by_city_chart(df):
    """Create traveler type distribution by city chart"""
    if df.empty or 'Travel Type' not in df.columns:
        return go.Figure().add_annotation(
            text="No traveler type data available",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="gray")
        )
    
    # Calculate traveler type distribution by city
    traveler_data = df.groupby(['City', 'Travel Type']).size().reset_index(name='count')
    
    if traveler_data.empty:
        return go.Figure().add_annotation(
            text="No traveler type distribution data",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="gray")
        )
      # Calculate percentages
    traveler_data['percentage'] = traveler_data.groupby('City')['count'].transform(lambda x: (x / x.sum()) * 100)
    
    # Create enhanced stacked bar chart
    fig = px.bar(
        traveler_data,
        x='City',
        y='percentage',
        color='Travel Type',
        title="👥 Traveler Type Distribution by City",
        labels={'percentage': 'Percentage of Reviews (%)', 'City': 'Tourist Destinations'},
        color_discrete_sequence=px.colors.qualitative.Set2,
        text='percentage'
    )
    
    # Enhanced styling
    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='inside',
        textfont_size=10,
        textfont_color='white'
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", size=12),
        title=dict(
            font=dict(size=18, weight='bold', color='#1f2937'),
            x=0.5,
            xanchor='center'
        ),
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="center", 
            x=0.5,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        ),
        height=450,
        xaxis=dict(
            title_font=dict(size=14, weight='bold'),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title_font=dict(size=14, weight='bold'),
            tickfont=dict(size=12),
            gridcolor='rgba(0,0,0,0.1)'
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

def create_top_nationalities_by_city_chart(df):
    """Create top nationalities by city chart with modern UI"""
    if df.empty:
        return go.Figure().add_annotation(
            text="No data available for nationality analysis",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="gray")
        )
    
    # Check if nationality column exists
    nationality_col = None
    for col in ['Nationality', 'nationality', 'Country', 'country']:
        if col in df.columns:
            nationality_col = col
            break
    
    if nationality_col is None:
        return go.Figure().add_annotation(
            text="No nationality data available",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="gray")
        )
    
    # Get top nationalities by city
    city_nationalities = []
    cities = df['City'].unique()
    
    for city in cities:
        city_data = df[df['City'] == city]
        if not city_data.empty and nationality_col in city_data.columns:
            # Count nationalities for this city
            nationality_counts = city_data[nationality_col].value_counts().head(5)
            
            for nationality, count in nationality_counts.items():
                if pd.notna(nationality) and nationality.strip():
                    percentage = (count / len(city_data)) * 100
                    city_nationalities.append({
                        'City': city,
                        'Nationality': str(nationality).strip(),
                        'Count': count,
                        'Percentage': percentage
                    })
    
    if not city_nationalities:
        return go.Figure().add_annotation(
            text="No nationality data found for any city",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="gray")
        )
    
    nationalities_df = pd.DataFrame(city_nationalities)
    
    # Create modern sunburst chart for better visualization
    fig = px.sunburst(
        nationalities_df,
        path=['City', 'Nationality'],
        values='Count',
        title="🌍 Top Nationalities by City",
        color='Percentage',
        color_continuous_scale='Viridis',
        hover_data={'Percentage': ':.1f%'}
    )
    
    # Enhanced styling for modern UI
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", size=12),
        title=dict(
            font=dict(size=20, weight='bold', color='#1f2937'),
            x=0.5,
            xanchor='center',
            pad=dict(t=20)
        ),
        height=550,
        margin=dict(l=70, r=70, t=120, b=70),
        coloraxis_colorbar=dict(
            title="Percentage (%)",
            titlefont=dict(size=12),
            tickfont=dict(size=11)
        )
    )
    
    # Update traces for better interactivity
    fig.update_traces(
        textinfo="label+percent entry",
        hovertemplate="<b>%{label}</b><br>" +
                      "Count: %{value}<br>" +
                      "Percentage: %{percentEntry}<br>" +
                      "<extra></extra>",
        maxdepth=2,
        branchvalues="total"
    )
    
    return fig

def create_nationality_bar_chart(df):
    """Create alternative bar chart for nationalities if sunburst doesn't work well"""
    if df.empty:
        return go.Figure().add_annotation(
            text="No data available for nationality analysis",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="gray")
        )
    
    # Check if nationality column exists
    nationality_col = None
    for col in ['Nationality', 'nationality', 'Country', 'country']:
        if col in df.columns:
            nationality_col = col
            break
    
    if nationality_col is None:
        return go.Figure().add_annotation(
            text="No nationality data available",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="gray")
        )
    
    # Get top 3 nationalities by city for cleaner visualization
    city_nationalities = []
    cities = df['City'].unique()
    
    for city in cities:
        city_data = df[df['City'] == city]
        if not city_data.empty:
            nationality_counts = city_data[nationality_col].value_counts().head(3)
            
            for nationality, count in nationality_counts.items():
                if pd.notna(nationality) and nationality.strip():
                    percentage = (count / len(city_data)) * 100
                    city_nationalities.append({
                        'City': city,
                        'Nationality': str(nationality).strip(),
                        'Count': count,
                        'Percentage': percentage
                    })
    
    if not city_nationalities:
        return go.Figure().add_annotation(
            text="No nationality data found",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="gray")
        )
    
    nationalities_df = pd.DataFrame(city_nationalities)
    
    # Create modern grouped bar chart
    fig = px.bar(
        nationalities_df,
        x='City',
        y='Percentage',
        color='Nationality',
        title="🌍 Top Visitor Nationalities by City",
        color_discrete_sequence=px.colors.qualitative.Set3,
        labels={'Percentage': 'Percentage of Visitors (%)', 'City': 'Tourist Destinations'},
        text='Percentage',
        hover_data={'Count': True}
    )
      # Enhanced styling and interactivity
    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside',
        textfont_size=11,
        textfont_color='#374151',
        marker_line=dict(width=1, color='rgba(255,255,255,0.3)'),
        hovertemplate='<b>%{x}</b><br>' +
                     'Nationality: %{fullData.name}<br>' +
                     'Percentage: %{y:.1f}%<br>' +
                     'Count: %{customdata}<br>' +
                     '<i>Click to see detailed breakdown</i>' +
                     '<extra></extra>',
        customdata=nationalities_df['Count']
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", size=12),
        title=dict(
            font=dict(size=18, weight='bold', color='#1f2937'),
            x=0.5,
            xanchor='center',
            pad=dict(t=20)
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1,
            font=dict(size=11)
        ),
        height=480,
        xaxis=dict(
            title='Cities',
            title_font=dict(size=14, weight='bold', color='#374151'),
            tickfont=dict(size=12, color='#6b7280'),
            showgrid=False
        ),
        yaxis=dict(
            title='Percentage of Visitors (%)',
            title_font=dict(size=14, weight='bold', color='#374151'),
            tickfont=dict(size=12, color='#6b7280'),
            gridcolor='rgba(0,0,0,0.1)'
        ),        margin=dict(l=70, r=70, t=120, b=80),
        hovermode='x unified',
        barmode='group'
    )
    
    return fig

def create_nationality_satisfaction_chart(df):
    """Create nationality satisfaction chart showing top 3 nationalities per city"""
    if df.empty:
        return go.Figure().add_annotation(
            text="No data available for nationality satisfaction analysis",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="gray")
        )
    
    # Check if nationality column exists
    nationality_col = None
    for col in ['Nationality', 'nationality', 'Country', 'country']:
        if col in df.columns:
            nationality_col = col
            break
    
    if nationality_col is None:
        return go.Figure().add_annotation(
            text="No nationality data available",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="gray")
        )
    
    # Calculate satisfaction by nationality and city
    nationality_satisfaction = []
    cities = df['City'].unique()
    
    for city in cities:
        city_data = df[df['City'] == city]
        if city_data.empty:
            continue
            
        # Group by nationality and calculate satisfaction
        nationality_groups = city_data.groupby(nationality_col).agg({
            'sentiment_standardized': ['mean', 'count']
        }).round(2)
        
        nationality_groups.columns = ['avg_satisfaction', 'review_count']
        nationality_groups = nationality_groups.reset_index()
        
        # Filter nationalities with at least 2 reviews
        nationality_groups = nationality_groups[nationality_groups['review_count'] >= 2]
        
        if nationality_groups.empty:
            continue
            
        # Get top 3 nationalities by review count for this city
        top_nationalities = nationality_groups.nlargest(3, 'review_count')
        
        for _, row in top_nationalities.iterrows():
            nationality = str(row[nationality_col]).strip()
            if nationality and nationality.lower() not in ['nan', 'none', '']:
                # Convert sentiment to satisfaction percentage (0-100 scale)
                satisfaction_score = ((row['avg_satisfaction'] + 1) / 2) * 100
                
                nationality_satisfaction.append({
                    'City': city,
                    'Nationality': nationality,
                    'Satisfaction': satisfaction_score,
                    'Review_Count': int(row['review_count'])
                })
    
    if not nationality_satisfaction:
        return go.Figure().add_annotation(
            text="No sufficient nationality satisfaction data found",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="gray")
        )
    
    satisfaction_df = pd.DataFrame(nationality_satisfaction)
    
    # Create grouped bar chart
    fig = px.bar(
        satisfaction_df,
        x='City',
        y='Satisfaction',
        color='Nationality',
        title="😊 Top 3 Nationalities Satisfaction by City",
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={'Satisfaction': 'Satisfaction Score (%)', 'City': 'Cities'},
        text='Satisfaction'
    )
    
    # Enhanced styling
    fig.update_traces(
        texttemplate='%{text:.0f}%',
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Nationality: %{fullData.name}<br>Satisfaction: %{y:.1f}%<br>Reviews: %{customdata}<br><extra></extra>',
        customdata=satisfaction_df['Review_Count']
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=480,
        yaxis=dict(range=[0, 100]),
        barmode='group'
    )
    
    return fig

# Layout
def layout():
    return html.Div([
        dcc.Location(id='url', refresh=False),
        dbc.Container([
            # Enhanced Header with gradient background
            dbc.Row([
                dbc.Col([                    html.Div([
                        html.H1("🌍 Geography Analysis", className="text-center mb-3", 
                               style={'color': 'white', 'fontWeight': '700', 'fontSize': '2.5rem'}),
                        html.P(
                            "Explore city-specific insights across Egypt's major tourist destinations",
                            className="text-center mb-4",
                            style={'color': 'rgba(255,255,255,0.9)', 'fontSize': '1.2rem'}
                        )
                    ], className="geography-header")
                ])
            ]),
              # Enhanced City Overview Cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-map-marker-alt", 
                                      style={'fontSize': '2rem', 'color': '#667eea'}),
                                html.H4("Cities Analyzed", className="card-title mt-2"),
                                html.H3(id="cities-overview", className="card-value", 
                                        style={'color': '#1f2937', 'fontWeight': 'bold'})
                            ], className="text-center")
                        ])
                    ], className="stat-card", style={
                        'borderLeft': '4px solid #667eea',
                        'boxShadow': '0 4px 15px rgba(0,0,0,0.1)',
                        'transition': 'transform 0.3s ease'
                    })
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-comments", 
                                      style={'fontSize': '2rem', 'color': '#28a745'}),
                                html.H4("Total Reviews", className="card-title mt-2"),
                                html.H3(id="total-reviews", className="card-value",
                                        style={'color': '#1f2937', 'fontWeight': 'bold'})
                            ], className="text-center")
                        ])
                    ], className="stat-card", style={
                        'borderLeft': '4px solid #28a745',
                        'boxShadow': '0 4px 15px rgba(0,0,0,0.1)',
                        'transition': 'transform 0.3s ease'
                    })
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-hotel", 
                                      style={'fontSize': '2rem', 'color': '#fd7e14'}),
                                html.H4("Hotels Covered", className="card-title mt-2"),
                                html.H3(id="total-hotels", className="card-value",
                                        style={'color': '#1f2937', 'fontWeight': 'bold'})
                            ], className="text-center")
                        ])
                    ], className="stat-card", style={
                        'borderLeft': '4px solid #fd7e14',
                        'boxShadow': '0 4px 15px rgba(0,0,0,0.1)',
                        'transition': 'transform 0.3s ease'
                    })
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-smile", 
                                      style={'fontSize': '2rem', 'color': '#20c997'}),
                                html.H4("Overall Sentiment", className="card-title mt-2"),
                                html.H3(id="overall-sentiment", className="card-value",
                                        style={'color': '#1f2937', 'fontWeight': 'bold'})
                            ], className="text-center")
                        ])
                    ], className="stat-card", style={
                        'borderLeft': '4px solid #20c997',
                        'boxShadow': '0 4px 15px rgba(0,0,0,0.1)',
                        'transition': 'transform 0.3s ease'
                    })
                ], width=3)
            ], className="mb-4"),
              # Enhanced City Filter Section
            dbc.Row([
                dbc.Col([                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-filter me-2", style={'color': '#667eea'}),
                                "🎯 Filter by City"
                            ], className="mb-0", style={'color': '#1f2937'})
                        ], style={'backgroundColor': '#f8f9fa', 'border': 'none'}),dbc.CardBody([
                            html.P("Select specific cities to focus your analysis, or leave empty to view all destinations.", 
                                  className="text-muted mb-3"),
                            dcc.Dropdown(
                                id="city-filter-dropdown",
                                multi=True,
                                placeholder="🏙️ Choose cities to analyze...",
                                className="enhanced-dropdown",
                                style={
                                    'fontSize': '14px',
                                    'borderRadius': '8px'
                                },
                                clearable=True,
                                searchable=True
                            ),
                            html.Div([
                                dbc.Button(
                                    "Select All Cities", 
                                    id="select-all-cities",
                                    color="outline-primary", 
                                    size="sm",
                                    className="me-2 mt-2"
                                ),
                                dbc.Button(
                                    "Clear Selection", 
                                    id="clear-cities",
                                    color="outline-secondary", 
                                    size="sm",
                                    className="mt-2"
                                )
                            ])
                        ])
                    ], style={
                        'boxShadow': '0 4px 15px rgba(0,0,0,0.1)',
                        'border': 'none',
                        'borderRadius': '12px'
                    })
                ])
            ], className="mb-4"),
          # Charts Row 1: Sentiment Analysis with enhanced padding
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(
                            id="sentiment-by-city-chart",
                            config={'displayModeBar': False},
                            style={'padding': '15px'}
                        )
                    ], style={'padding': '2rem 1.5rem'})
                ], className="chart-card")
            ], width=12)
        ], className="mb-5"),        # Charts Row 2: Enhanced Token Analysis with larger display
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-thumbs-up me-2", style={'color': '#10B981', 'fontSize': '1.2rem'}),
                            html.H5("✨ Top Positive Keywords by City", className="mb-0", style={'color': '#065F46', 'fontWeight': '600'}),
                            html.Small("Most frequently mentioned positive aspects", className="text-muted d-block mt-1")
                        ])
                    ], style={'backgroundColor': '#ECFDF5', 'border': 'none', 'borderBottom': '3px solid #10B981'}),
                    dbc.CardBody([
                        dcc.Graph(
                            id="positive-tokens-chart",
                            config={
                                'displayModeBar': True,
                                'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                                'displaylogo': False
                            },
                            style={'padding': '10px', 'minHeight': '600px'}
                        )
                    ], style={'padding': '1rem'})
                ], className="chart-card shadow-lg")
            ], width=12, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-thumbs-down me-2", style={'color': '#EF4444', 'fontSize': '1.2rem'}),
                            html.H5("⚠️ Top Negative Keywords by City", className="mb-0", style={'color': '#991B1B', 'fontWeight': '600'}),
                            html.Small("Most frequently mentioned issues and concerns", className="text-muted d-block mt-1")
                        ])
                    ], style={'backgroundColor': '#FEF2F2', 'border': 'none', 'borderBottom': '3px solid #EF4444'}),
                    dbc.CardBody([
                        dcc.Graph(
                            id="negative-tokens-chart",
                            config={
                                'displayModeBar': True,
                                'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                                'displaylogo': False
                            },
                            style={'padding': '10px', 'minHeight': '600px'}
                        )
                    ], style={'padding': '1rem'})
                ], className="chart-card shadow-lg")
            ], width=12)
        ], className="mb-5"),
          # Charts Row 3: Hotel Performance with enhanced padding
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(
                            id="hotel-performance-chart",
                            config={'displayModeBar': False},
                            style={'padding': '15px'}
                        )
                    ], style={'padding': '2rem 1.5rem'})
                ], className="chart-card")
            ], width=12)
        ], className="mb-5"),
          # Charts Row 4: Traveler Types with enhanced padding
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(
                            id="traveler-type-chart",
                            config={'displayModeBar': False},
                            style={'padding': '15px'}
                        )
                    ], style={'padding': '2rem 1.5rem'})
                ], className="chart-card")
            ], width=12)        ], className="mb-5"),
          # Charts Row 5: Top Nationalities with modern UI
        dbc.Row([
            dbc.Col([
                dbc.Card([                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-globe-americas me-2", style={'color': 'white', 'fontSize': '1.2rem'}),
                            html.H5("🌍 Visitor Nationalities by City", className="mb-0", style={'color': 'white', 'fontWeight': '600'}),
                            html.Small("(Click on chart bars for detailed breakdown)", className="text-white-50 d-block mt-1")
                        ])
                    ], className="modern-card-header", style={'cursor': 'pointer'}),                    dbc.CardBody([
                        dcc.Graph(
                            id="nationalities-chart",
                            config={
                                'displayModeBar': False,
                                'scrollZoom': False,
                                'doubleClick': 'reset'
                            },
                            style={'padding': '15px', 'cursor': 'pointer'}
                        )
                    ], style={'padding': '2rem 1.5rem'}, className="chart-click-hint")
                ], className="chart-card nationality-chart-container")
            ], width=12)        ], className="mb-5"),
        
        # Charts Row 6: Top 3 Nationalities Satisfaction by City
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-smile me-2", style={'color': 'white', 'fontSize': '1.2rem'}),
                            html.H5("😊 Top 3 Nationalities Satisfaction by City", className="mb-0", style={'color': 'white', 'fontWeight': '600'}),
                            html.Small("(Click on chart bars for complete satisfaction breakdown)", className="text-white-50 d-block mt-1")
                        ])
                    ], className="modern-card-header", style={'cursor': 'pointer'}),
                    dbc.CardBody([
                        dcc.Graph(
                            id="nationality-satisfaction-chart",
                            config={
                                'displayModeBar': False,
                                'scrollZoom': False,
                                'doubleClick': 'reset'
                            },
                            style={'padding': '15px', 'cursor': 'pointer'}
                        )
                    ], style={'padding': '2rem 1.5rem'}, className="chart-click-hint")
                ], className="chart-card nationality-satisfaction-container")
            ], width=12)
        ], className="mb-5"),
        
        # Geographic Insights with enhanced padding
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("🔍 Key Geographic Insights", className="mb-0", style={'color': '#1f2937'})
                    ], style={'backgroundColor': '#f8f9fa', 'border': 'none'}),
                    dbc.CardBody([
                        html.Div(id="geographic-insights", className="geography-insights")
                    ], style={'padding': '2rem 1.5rem'})
                ], className="chart-card")
            ], width=12)        ], className="mb-5")
        
    ], fluid=True, className="geography-page"),
      # Modal for detailed nationality information
    dbc.Modal([
        dbc.ModalHeader([
            html.H4(id="nationality-modal-title", className="modal-title"),
            dbc.Button("×", id="close-nationality-modal", className="btn-close", n_clicks=0)
        ], className="nationality-modal-header"),
        dbc.ModalBody(id="nationality-modal-body", className="nationality-modal-body"),
        dbc.ModalFooter([
            dbc.Button("Close", id="close-nationality-modal-footer", className="btn btn-secondary", n_clicks=0)
        ], className="nationality-modal-footer")
    ], id="nationality-modal", size="xl", is_open=False, className="nationality-modal-content"),
    
    # Modal for detailed nationality satisfaction information
    dbc.Modal([
        dbc.ModalHeader([
            html.H4(id="nationality-satisfaction-modal-title", className="modal-title"),
            dbc.Button("×", id="close-nationality-satisfaction-modal", className="btn-close", n_clicks=0)
        ], className="nationality-modal-header"),
        dbc.ModalBody(id="nationality-satisfaction-modal-body", className="nationality-modal-body"),
        dbc.ModalFooter([
            dbc.Button("Close", id="close-nationality-satisfaction-modal-footer", className="btn btn-secondary", n_clicks=0)
        ], className="nationality-modal-footer")
    ], id="nationality-satisfaction-modal", size="xl", is_open=False, className="nationality-modal-content")
    ])

# Callbacks
@callback(
    [Output('cities-overview', 'children'),
     Output('total-reviews', 'children'),
     Output('total-hotels', 'children'),
     Output('overall-sentiment', 'children'),
     Output('city-filter-dropdown', 'options'),
     Output('city-filter-dropdown', 'value')],
    [Input('url', 'pathname')]  # Use URL pathname to trigger initial load
)
def update_geography_overview(_):
    df = get_geography_data()
    
    if df.empty:
        return ("No data", "0", "0", "N/A", [], [])
    
    cities = sorted(df['City'].unique())
    total_reviews = len(df)
    total_hotels = df['Hotel Name'].nunique()
    
    # Calculate overall sentiment
    sentiment_counts = df['sentiment_standardized'].value_counts()
    positive_pct = (sentiment_counts.get(1, 0) / total_reviews * 100) if total_reviews > 0 else 0
    
    city_options = [{'label': city, 'value': city} for city in cities]
    
    return (
        f"{len(cities)} cities",
        f"{total_reviews:,}",
        f"{total_hotels:,}",
        f"{positive_pct:.1f}% positive",
        city_options,
        cities  # Select all cities by default
    )

@callback(
    [Output('sentiment-by-city-chart', 'figure'),
     Output('positive-tokens-chart', 'figure'),
     Output('negative-tokens-chart', 'figure'),
     Output('hotel-performance-chart', 'figure'),
     Output('traveler-type-chart', 'figure'),
     Output('nationalities-chart', 'figure'),
     Output('nationality-satisfaction-chart', 'figure'),
     Output('geographic-insights', 'children')],
    [Input('city-filter-dropdown', 'value')]
)
def update_geography_charts(selected_cities):
    df = get_geography_data()
    
    if df.empty:
        empty_fig = go.Figure().add_annotation(
            text="No data available",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="gray")
        )
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, "No data available for analysis."
    
    # Filter by selected cities
    if selected_cities:
        df_filtered = df[df['City'].isin(selected_cities)]
    else:
        df_filtered = df
    
    if df_filtered.empty:
        empty_fig = go.Figure().add_annotation(
            text="No data for selected cities",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="gray")
        )
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, "No data available for selected cities."    # Create charts with enhanced token display
    sentiment_fig = create_sentiment_by_city_chart(df_filtered)
    positive_tokens_fig = create_top_tokens_by_city_chart(df_filtered, 'positive', top_n=8)  # Show more tokens
    negative_tokens_fig = create_top_tokens_by_city_chart(df_filtered, 'negative', top_n=8)  # Show more tokens
    hotel_performance_fig = create_hotel_performance_by_city_chart(df_filtered)
    traveler_type_fig = create_traveler_type_by_city_chart(df_filtered)
    nationality_fig = create_nationality_bar_chart(df_filtered)  # Use bar chart for better readability
    nationality_satisfaction_fig = create_nationality_satisfaction_chart(df_filtered)
      # Generate insights
    insights = generate_geographic_insights(df_filtered)
    
    return (
        sentiment_fig,
        positive_tokens_fig, 
        negative_tokens_fig,
        hotel_performance_fig,
        traveler_type_fig,
        nationality_fig,
        nationality_satisfaction_fig,
        insights
    )

def generate_geographic_insights(df):
    """Generate dynamic geographic insights"""
    if df.empty:
        return html.P("No data available for generating insights.")
    
    insights = []
    cities = df['City'].unique()
    
    # Overall statistics
    total_reviews = len(df)
    positive_reviews = len(df[df['sentiment_standardized'] == 1])
    positive_rate = (positive_reviews / total_reviews * 100) if total_reviews > 0 else 0
    
    insights.append(
        dbc.Alert([
            html.H6("📊 Overall Summary", className="alert-heading"),
            html.P(f"Analyzing {total_reviews:,} reviews across {len(cities)} cities with {positive_rate:.1f}% positive sentiment.")
        ], color="info")
    )
    
    # City-specific insights
    if len(cities) > 1:
        city_sentiments = []
        for city in cities:
            city_data = df[df['City'] == city]
            city_positive = len(city_data[city_data['sentiment_standardized'] == 1])
            city_total = len(city_data)
            city_rate = (city_positive / city_total * 100) if city_total > 0 else 0
            city_sentiments.append((city, city_rate, city_total))
        
        # Best performing city
        best_city = max(city_sentiments, key=lambda x: x[1])
        insights.append(
            dbc.Alert([
                html.H6("🏆 Top Performing City", className="alert-heading"),
                html.P(f"{best_city[0]} leads with {best_city[1]:.1f}% positive sentiment from {best_city[2]:,} reviews.")
            ], color="success")
        )
        
        # City with most reviews
        most_reviewed_city = max(city_sentiments, key=lambda x: x[2])
        if most_reviewed_city[0] != best_city[0]:
            insights.append(
                dbc.Alert([
                    html.H6("📈 Most Reviewed City", className="alert-heading"),
                    html.P(f"{most_reviewed_city[0]} has the most reviews ({most_reviewed_city[2]:,}) with {most_reviewed_city[1]:.1f}% positive sentiment.")
                ], color="primary")
            )
      # Hotel insights
    if 'Hotel Name' in df.columns:
        hotels_per_city = df.groupby('City')['Hotel Name'].nunique()
        city_with_most_hotels = hotels_per_city
        most_hotels_count = hotels_per_city.max()
        
        insights.append(
            dbc.Alert([
                html.H6("🏨 Hotel Diversity", className="alert-heading"),
                html.P(f"{city_with_most_hotels} offers the most hotel options with {most_hotels_count} different properties.")
            ], color="warning")
        )
    
    # Nationality insights
    nationality_col = None
    for col in ['Nationality', 'nationality', 'Country', 'country']:
        if col in df.columns:
            nationality_col = col
            break
    
    if nationality_col is not None:
        # Find most diverse city (most nationalities)
        city_diversity = df.groupby('City')[nationality_col].nunique()
        most_diverse_city = city_diversity.idxmax()
        diversity_count = city_diversity.max()
        
        # Find most common nationality overall
        top_nationality = df[nationality_col].mode()
        if not top_nationality.empty:
            top_nat = top_nationality.iloc[0]
            total_from_top_nat = df[df[nationality_col] == top_nat].shape[0]
            percentage = (total_from_top_nat / len(df)) * 100
            
            insights.append(
                dbc.Alert([
                    html.H6("🌍 International Appeal", className="alert-heading"),
                    html.P(f"{most_diverse_city} attracts the most diverse visitors ({diversity_count} different nationalities). {top_nat} visitors make up {percentage:.1f}% of all reviews.")
                ], color="info")
            )
    
    return html.Div(insights)

# Callbacks for city filter buttons
@callback(
    Output('city-filter-dropdown', 'value', allow_duplicate=True),
    [Input('select-all-cities', 'n_clicks'),
     Input('clear-cities', 'n_clicks')],
    [State('city-filter-dropdown', 'options')],
    prevent_initial_call=True
)
def handle_city_filter_buttons(select_all_clicks, clear_clicks, options):
    """Handle select all and clear buttons for city filter"""
    if not options:
        return []
    
    ctx_triggered = ctx.triggered[0]['prop_id'] if ctx.triggered else None
    
    if ctx_triggered == 'select-all-cities.n_clicks':
        return [option['value'] for option in options]
    elif ctx_triggered == 'clear-cities.n_clicks':
        return []
    
    return []

# Callback for nationality modal
@callback(
    [Output('nationality-modal', 'is_open'),
     Output('nationality-modal-title', 'children'),
     Output('nationality-modal-body', 'children')],
    [Input('nationalities-chart', 'clickData'),
     Input('close-nationality-modal', 'n_clicks'),
     Input('close-nationality-modal-footer', 'n_clicks')],
    [State('nationality-modal', 'is_open'),
     State('city-filter-dropdown', 'value')],
    prevent_initial_call=True
)
def show_nationality_modal(clickData, close1, close2, is_open, selected_cities):
    """Show detailed nationality modal when chart is clicked"""
    ctx_id = ctx.triggered_id if hasattr(ctx, 'triggered_id') else None
    
    # Close modal
    if ctx_id in ['close-nationality-modal', 'close-nationality-modal-footer']:
        return False, "", ""
    
    # Open modal with city data
    if clickData and clickData.get('points'):
        point = clickData['points'][0]
        city = None
        
        # Get city from click data
        if 'x' in point:
            city = point['x']
        elif 'label' in point:
            city = point['label']
        elif 'customdata' in point:
            city = point['customdata']
        
        if not city:
            # If no specific city, show all cities data
            df = get_geography_data()
            if selected_cities:
                df = df[df['City'].isin(selected_cities)]
            
            modal_title = "🌍 Visitor Nationalities - All Cities"
            modal_body = create_detailed_nationality_content(df, selected_city=None)
            return True, modal_title, modal_body
          # Load data and create modal content for specific city
        df = get_geography_data()
        if selected_cities:
            df = df[df['City'].isin(selected_cities)]
        
        modal_title = f"🌍 Visitor Nationalities for {city}"
        modal_body = create_detailed_nationality_content(df, selected_city=city)
        
        return True, modal_title, modal_body
    
    return False, "", ""

# Callback for nationality satisfaction modal
@callback(
    [Output('nationality-satisfaction-modal', 'is_open'),
     Output('nationality-satisfaction-modal-title', 'children'),
     Output('nationality-satisfaction-modal-body', 'children')],
    [Input('nationality-satisfaction-chart', 'clickData'),
     Input('close-nationality-satisfaction-modal', 'n_clicks'),
     Input('close-nationality-satisfaction-modal-footer', 'n_clicks')],
    [State('nationality-satisfaction-modal', 'is_open'),
     State('city-filter-dropdown', 'value')],
    prevent_initial_call=True
)
def show_nationality_satisfaction_modal(clickData, close1, close2, is_open, selected_cities):
    """Show detailed nationality satisfaction modal when chart is clicked"""
    ctx_id = ctx.triggered_id if hasattr(ctx, 'triggered_id') else None
    
    # Close modal
    if ctx_id in ['close-nationality-satisfaction-modal', 'close-nationality-satisfaction-modal-footer']:
        return False, "", ""
    
    # Open modal with city data
    if clickData and clickData.get('points'):
        point = clickData['points'][0]
        city = None
        
        # Get city from click data
        if 'x' in point:
            city = point['x']
        elif 'label' in point:
            city = point['label']
        elif 'customdata' in point:
            city = point['customdata']
        
        # Load data and create modal content
        df = get_geography_data()
        if selected_cities:
            df = df[df['City'].isin(selected_cities)]
        
        if city:
            modal_title = f"😊 All Nationalities Satisfaction for {city}"
            modal_body = create_detailed_nationality_satisfaction_content(df, selected_city=city)
        else:
            modal_title = "😊 All Nationalities Satisfaction - All Cities"
            modal_body = create_detailed_nationality_satisfaction_content(df, selected_city=None)
        
        return True, modal_title, modal_body
    
    return False, "", ""

def create_detailed_nationality_satisfaction_content(df, selected_city=None):
    """Create detailed nationality satisfaction breakdown content for modal"""
    if df.empty:
        return html.Div([
            html.P("No nationality satisfaction data available.", className="text-muted text-center")
        ])
    
    # Check if nationality column exists
    nationality_col = None
    for col in ['Nationality', 'nationality', 'Country', 'country']:
        if col in df.columns:
            nationality_col = col
            break
    
    if nationality_col is None:
        return html.Div([
            html.P("No nationality information found in the dataset.", className="text-muted text-center")
        ])
    
    content = []
    
    # If no specific city is selected, show all cities
    if selected_city is None:
        cities = sorted(df['City'].unique())
        
        # Add overall summary first
        total_reviews = len(df)
        avg_satisfaction = ((df['sentiment_standardized'].mean() + 1) / 2) * 100
        total_nationalities = df[nationality_col].nunique()
        
        content.append(
            html.Div([
                html.H5([
                    html.I(className="fas fa-smile me-2", style={'color': '#10b981'}),
                    "Overall Satisfaction Summary"
                ], className="mb-3", style={'color': '#1f2937', 'borderBottom': '2px solid #e5e7eb', 'paddingBottom': '0.5rem'}),
                
                # Overall stats
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Total Reviews", className="card-subtitle mb-2 text-muted"),
                                html.H4(f"{total_reviews:,}", className="card-title mb-0", style={'color': '#16a34a'})
                            ])
                        ], className="text-center mb-3 modal-stat-card", style={'border': 'none', 'background': '#f0fdf4'})
                    ], width=4),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Overall Satisfaction", className="card-subtitle mb-2 text-muted"),
                                html.H4(f"{avg_satisfaction:.1f}%", className="card-title mb-0", style={'color': '#2563eb'})
                            ])
                        ], className="text-center mb-3 modal-stat-card", style={'border': 'none', 'background': '#eff6ff'})
                    ], width=4),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Total Nationalities", className="card-subtitle mb-2 text-muted"),
                                html.H4(f"{total_nationalities}", className="card-title mb-0", style={'color': '#dc2626'})
                            ])
                        ], className="text-center mb-3 modal-stat-card", style={'border': 'none', 'background': '#fef2f2'})
                    ], width=4)
                ]),
                html.Hr(style={'margin': '2rem 0'})
            ])
        )
    else:
        cities = [selected_city]
    
    # Show data for each city
    for city in cities:
        city_data = df[df['City'] == city]
        if city_data.empty:
            continue
            
        # Get nationality satisfaction for this city
        nationality_satisfaction = city_data.groupby(nationality_col).agg({
            'sentiment_standardized': ['mean', 'count']
        }).round(3)
        
        nationality_satisfaction.columns = ['avg_satisfaction', 'review_count']
        nationality_satisfaction = nationality_satisfaction.reset_index()
        
        # Filter nationalities with at least 1 review
        nationality_satisfaction = nationality_satisfaction[nationality_satisfaction['review_count'] >= 1]
        
        if nationality_satisfaction.empty:
            continue
        
        # Convert satisfaction to percentage and sort by satisfaction
        nationality_satisfaction['satisfaction_pct'] = ((nationality_satisfaction['avg_satisfaction'] + 1) / 2) * 100
        nationality_satisfaction = nationality_satisfaction.sort_values('satisfaction_pct', ascending=False)
        
        total_reviews = len(city_data)
        
        # Create city section
        city_section = [
            html.Div([
                html.H5([
                    html.I(className="fas fa-map-marker-alt me-2", style={'color': '#667eea'}),
                    f"{city}"
                ], className="mb-3", style={'color': '#1f2937', 'borderBottom': '2px solid #e5e7eb', 'paddingBottom': '0.5rem'}),
                
                # Overview stats
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Total Reviews", className="card-subtitle mb-2 text-muted"),
                                html.H4(f"{total_reviews:,}", className="card-title mb-0", style={'color': '#16a34a'})
                            ])
                        ], className="text-center mb-3 modal-stat-card", style={'border': 'none', 'background': '#f0fdf4'})
                    ], width=4),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Avg City Satisfaction", className="card-subtitle mb-2 text-muted"),
                                html.H4(f"{nationality_satisfaction['satisfaction_pct'].mean():.1f}%", className="card-title mb-0", style={'color': '#2563eb'})
                            ])
                        ], className="text-center mb-3 modal-stat-card", style={'border': 'none', 'background': '#eff6ff'})
                    ], width=4),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Nationalities Count", className="card-subtitle mb-2 text-muted"),
                                html.H4(f"{len(nationality_satisfaction)}", className="card-title mb-0", style={'color': '#dc2626'})
                            ])
                        ], className="text-center mb-3 modal-stat-card", style={'border': 'none', 'background': '#fef2f2'})
                    ], width=4)
                ]),
                
                # Detailed satisfaction table
                html.H6("😊 Complete Nationalities Satisfaction Breakdown", className="mt-4 mb-3", style={'color': '#374151'}),
                dbc.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("Rank", style={'width': '8%'}),
                            html.Th("Nationality", style={'width': '35%'}),
                            html.Th("Satisfaction", style={'width': '25%'}),
                            html.Th("Reviews", style={'width': '15%'}),
                            html.Th("Score", style={'width': '17%'})
                        ])
                    ]),
                    html.Tbody([
                        html.Tr([
                            html.Td([
                                dbc.Badge(f"#{idx+1}", 
                                         color="success" if row['satisfaction_pct'] >= 75 else 
                                               "warning" if row['satisfaction_pct'] >= 50 else "danger", 
                                         className="me-1")
                            ]),
                            html.Td([
                                html.Div([
                                    html.I(className="fas fa-flag me-2", style={'color': '#6b7280'}),
                                    html.Strong(row[nationality_col]) if idx < 5 else row[nationality_col]
                                ])
                            ]),
                            html.Td([
                                html.Div([
                                    f"{row['satisfaction_pct']:.1f}%",
                                    dbc.Progress(
                                        value=row['satisfaction_pct'],
                                        color="success" if row['satisfaction_pct'] >= 75 else 
                                              "warning" if row['satisfaction_pct'] >= 50 else "danger",
                                        className="mt-1 nationality-progress",
                                        style={'height': '4px'}
                                    )
                                ])
                            ]),
                            html.Td(f"{int(row['review_count'])}"),
                            html.Td([
                                dbc.Badge(
                                    "Excellent" if row['satisfaction_pct'] >= 80 else
                                    "Good" if row['satisfaction_pct'] >= 65 else
                                    "Average" if row['satisfaction_pct'] >= 50 else
                                    "Poor",
                                    color="success" if row['satisfaction_pct'] >= 65 else
                                          "warning" if row['satisfaction_pct'] >= 50 else "danger"
                                )
                            ])
                        ]) for idx, (_, row) in enumerate(nationality_satisfaction.iterrows())
                    ])
                ], striped=True, hover=True, responsive=True, className="modal-nationality-table")
            ], className="mb-4 city-nationality-section")
        ]
        
        content.extend(city_section)
        
        # Add separator if multiple cities
        if len(cities) > 1 and city != cities[-1]:
            content.append(html.Hr(style={'margin': '2rem 0', 'border': '2px solid #e5e7eb'}))
    
    return html.Div(content) if content else html.Div([
        html.P("No nationality satisfaction data found for the selected cities.", className="text-muted text-center")
    ])

def create_detailed_nationality_content(df, selected_city=None):
    """Create detailed nationality breakdown content for modal"""
    if df.empty:
        return html.Div([
            html.P("No nationality data available.", className="text-muted text-center")
        ])
    
    # Check if nationality column exists
    nationality_col = None
    for col in ['Nationality', 'nationality', 'Country', 'country']:
        if col in df.columns:
            nationality_col = col
            break
    
    if nationality_col is None:
        return html.Div([
            html.P("No nationality information found in the dataset.", className="text-muted text-center")
        ])
    
    content = []
    
    # If no specific city is selected, show all cities
    if selected_city is None:
        cities = sorted(df['City'].unique())
        
        # Add overall summary first
        total_reviews = len(df)
        total_nationalities = df[nationality_col].nunique()
        overall_nationality_counts = df[nationality_col].value_counts()
        
        content.append(
            html.Div([
                html.H5([
                    html.I(className="fas fa-globe me-2", style={'color': '#667eea'}),
                    "Overall Nationality Summary"
                ], className="mb-3", style={'color': '#1f2937', 'borderBottom': '2px solid #e5e7eb', 'paddingBottom': '0.5rem'}),
                
                # Overall stats
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Total Reviews", className="card-subtitle mb-2 text-muted"),
                                html.H4(f"{total_reviews:,}", className="card-title mb-0", style={'color': '#16a34a'})
                            ])
                        ], className="text-center mb-3 modal-stat-card", style={'border': 'none', 'background': '#f0fdf4'})
                    ], width=4),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Total Nationalities", className="card-subtitle mb-2 text-muted"),
                                html.H4(f"{total_nationalities}", className="card-title mb-0", style={'color': '#2563eb'})
                            ])
                        ], className="text-center mb-3 modal-stat-card", style={'border': 'none', 'background': '#eff6ff'})
                    ], width=4),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Most Common", className="card-subtitle mb-2 text-muted"),
                                html.H4(f"{overall_nationality_counts.index[0] if not overall_nationality_counts.empty else 'N/A'}", 
                                        className="card-title mb-0", style={'color': '#dc2626', 'fontSize': '1rem'})
                            ])
                        ], className="text-center mb-3 modal-stat-card", style={'border': 'none', 'background': '#fef2f2'})
                    ], width=4)
                ]),
                html.Hr(style={'margin': '2rem 0'})
            ])
        )
    else:
        cities = [selected_city]
    
    # Show data for each city
    for city in cities:
        city_data = df[df['City'] == city]
        if city_data.empty:
            continue
            
        # Get nationality statistics for this city
        nationality_counts = city_data[nationality_col].value_counts()
        total_reviews = len(city_data)
        
        if nationality_counts.empty:
            continue
        
        # Create city section
        city_section = [
            html.Div([
                html.H5([
                    html.I(className="fas fa-map-marker-alt me-2", style={'color': '#667eea'}),
                    f"{city}"
                ], className="mb-3", style={'color': '#1f2937', 'borderBottom': '2px solid #e5e7eb', 'paddingBottom': '0.5rem'}),
                
                # Overview stats
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Total Reviews", className="card-subtitle mb-2 text-muted"),
                                html.H4(f"{total_reviews:,}", className="card-title mb-0", style={'color': '#16a34a'})
                            ])
                        ], className="text-center mb-3 modal-stat-card", style={'border': 'none', 'background': '#f0fdf4'})
                    ], width=4),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Unique Nationalities", className="card-subtitle mb-2 text-muted"),
                                html.H4(f"{len(nationality_counts)}", className="card-title mb-0", style={'color': '#2563eb'})
                            ])
                        ], className="text-center mb-3 modal-stat-card", style={'border': 'none', 'background': '#eff6ff'})
                    ], width=4),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Top Nationality", className="card-subtitle mb-2 text-muted"),
                                html.H4(f"{nationality_counts.index[0]}", className="card-title mb-0", style={'color': '#dc2626', 'fontSize': '1rem'})
                            ])
                        ], className="text-center mb-3 modal-stat-card", style={'border': 'none', 'background': '#fef2f2'})
                    ], width=4)
                ]),
                
                # Detailed breakdown table with more entries
                html.H6("📊 Detailed Nationality Breakdown", className="mt-4 mb-3", style={'color': '#374151'}),
                dbc.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("Rank", style={'width': '10%'}),
                            html.Th("Nationality", style={'width': '40%'}),
                            html.Th("Reviews", style={'width': '25%'}),
                            html.Th("Percentage", style={'width': '25%'})
                        ])
                    ]),
                    html.Tbody([
                        html.Tr([
                            html.Td([
                                dbc.Badge(f"#{idx+1}", color="primary" if idx < 3 else "secondary", className="me-1")
                            ]),
                            html.Td([
                                html.Div([
                                    html.I(className="fas fa-flag me-2", style={'color': '#6b7280'}),
                                    html.Strong(nationality) if idx < 3 else nationality
                                ])
                            ]),
                            html.Td(f"{count:,}"),
                            html.Td([
                                html.Div([
                                    f"{(count/total_reviews)*100:.1f}%",
                                    dbc.Progress(
                                        value=(count/total_reviews)*100,
                                        color="success" if idx < 3 else "info",
                                        className="mt-1 nationality-progress",
                                        style={'height': '4px'}
                                    )
                                ])
                            ])
                        ]) for idx, (nationality, count) in enumerate(nationality_counts.head(15).items())
                    ])
                ], striped=True, hover=True, responsive=True, className="modal-nationality-table")
            ], className="mb-4 city-nationality-section")
        ]
        
        content.extend(city_section)
        
        # Add separator if multiple cities
        if len(cities) > 1 and city != cities[-1]:
            content.append(html.Hr(style={'margin': '2rem 0', 'border': '2px solid #e5e7eb'}))
    
    return html.Div(content) if content else html.Div([
        html.P("No nationality data found for the selected cities.", className="text-muted text-center")
    ])
