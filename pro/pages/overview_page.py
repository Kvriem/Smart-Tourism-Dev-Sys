import pandas as pd
import dash_bootstrap_components as dbc
from dash import html, dcc
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import ast
import time
from collections import Counter
from constants import SATISFACTION_LEGEND_HTML, CITY_RATING_LEGEND_HTML, CHART_TEXT_COLORS, CHART_BG
from database_config import load_data_from_database
from performance_cache import cache_result, cache_chart_result, cache_processed_result, performance_cache
from data_processing_optimized import get_cities_fast, filter_data_fast, get_basic_stats_fast

from optimized_charts import (
    create_kpi_cards_cached,
    create_satisfaction_trend_chart_cached,
    create_word_frequency_chart_cached,
    create_nationality_chart_cached,
    create_city_satisfaction_chart_cached,
    create_tourism_economics_chart_cached,
    create_seasonal_analytics_chart_cached
)

# Optimized caching with performance improvements
CACHE_DURATION = 300  # 5 minutes cache for fast page switching

def standardize_sentiment(sentiment_val):
    """
    Convert sentiment to standardized numeric format for consistent aggregation
    Returns: 1 for positive, -1 for negative, 0 for neutral
    """
    if pd.isna(sentiment_val):
        return 0  # Neutral
    
    sentiment_str = str(sentiment_val).lower().strip()
    
    # Handle various sentiment formats
    if sentiment_str in ['positive', 'pos', '1', '1.0', 'good', 'great', 'excellent', 'very positive']:
        return 1
    elif sentiment_str in ['negative', 'neg', '-1', '-1.0', 'bad', 'poor', 'terrible', 'very negative']:
        return -1
    elif sentiment_str in ['neutral', 'neu', '0', '0.0', 'average', 'ok', 'fair']:
        return 0
    else:
        # Try to parse as numeric
        try:
            numeric_val = float(sentiment_str)
            if numeric_val > 0.6:
                return 1
            elif numeric_val < 0.4:
                return -1
            else:
                return 0
        except:
            return 0  # Default to neutral if can't parse

def clear_data_cache():
    """Clear the data cache to force fresh loading"""
    performance_cache.clear_all()
    print("🗑️  Performance cache cleared - next load will be fresh")
    return True

def force_database_load():
    """Force load data directly from database, bypassing all cache and fallbacks"""
    print("🚀 FORCING DATABASE LOAD - NO CACHE, NO FALLBACKS")
    try:
        return load_data_from_database(force_reload=True)
    except Exception as e:
        print(f"❌ Force load failed: {e}")
        return pd.DataFrame()



@cache_result("load_overview_data", ttl=300)
def load_data():
    """Load and process the data for overview metrics with optimized caching"""
    start_time = time.time()
    
    try:
        df = load_data_from_database()
        if df.empty:
            print("⚠️ No data loaded - returning empty DataFrame")
            return df
        
        print(f"✅ Data loaded successfully: {len(df)} records")
        load_time = time.time() - start_time
        print(f"⏱️ Overview data loading took {load_time:.2f} seconds")
        
        return df
        
    except Exception as e:
        print(f"❌ Error in load_data: {e}")
        return pd.DataFrame()

def calculate_kpis(df, city_filter="all", start_date=None, end_date=None):
    """Calculate KPI metrics based on the filtered data"""
    print(f"Debug KPIs - city_filter: {city_filter}, start_date: {start_date}, end_date: {end_date}")
    print(f"Debug KPIs - Initial dataset size: {len(df)}")
    print(f"Debug KPIs - Available columns: {list(df.columns)}")
    
    if df.empty:
        return {
            'total_visitors': 0,
            'positive_reviews': 0,
            'negative_reviews': 0,
            'satisfaction_rate': 0
        }
    
    # Check for required columns and find alternatives if needed
    city_column = None
    sentiment_column = None
    date_column = None
    
    # Find city column
    for col in df.columns:
        if col.lower() in ['city', 'location', 'destination']:
            city_column = col
            break
    
    # Find sentiment column
    for col in df.columns:
        if 'sentiment' in col.lower() and 'classification' in col.lower():
            sentiment_column = col
            break
    
    # Find date column
    for col in df.columns:
        if 'review' in col.lower() and 'date' in col.lower():
            date_column = col
            break
    
    print(f"Debug KPIs - Using columns: city={city_column}, sentiment={sentiment_column}, date={date_column}")
    
    # Apply city filter
    if city_filter != "all" and city_column:
        df = df[df[city_column] == city_filter]
        print(f"Debug KPIs - After city filter: {len(df)}")
    
    # Apply date filter if dates are provided
    if (start_date or end_date) and date_column:
        # Convert the date column to datetime
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        print(f"Debug KPIs - Date column converted")
        
        if start_date:
            start_date = pd.to_datetime(start_date)
            df = df[df[date_column] >= start_date]
            print(f"Debug KPIs - After start date filter: {len(df)}")
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            df = df[df[date_column] <= end_date]
            print(f"Debug KPIs - After end date filter: {len(df)}")
    
    if df.empty:
        print("Debug KPIs - Dataset is empty after filtering")
        return {
            'total_visitors': 0,
            'positive_reviews': 0,
            'negative_reviews': 0,
            'satisfaction_rate': 0        }
    
    # Calculate metrics
    total_visitors = len(df)
    
    if sentiment_column:
        # Apply sentiment standardization using the shared function
        df['sentiment_numeric'] = df[sentiment_column].apply(standardize_sentiment)
        
        # Count positive and negative reviews using standardized values
        positive_reviews = len(df[df['sentiment_numeric'] == 1])
        negative_reviews = len(df[df['sentiment_numeric'] == -1])
        
        print(f"Debug KPIs - Sentiment standardization: positive={positive_reviews}, negative={negative_reviews}, total={total_visitors}")
    else:
        print("Warning: No sentiment column found, using default values")
        positive_reviews = 0
        negative_reviews = 0
    
    # Calculate satisfaction rate (positive reviews / total reviews * 100)
    satisfaction_rate = (positive_reviews / total_visitors * 100) if total_visitors > 0 else 0
    
    print(f"Debug KPIs - Final results: visitors={total_visitors}, positive={positive_reviews}, negative={negative_reviews}, satisfaction={satisfaction_rate}")
    
    return {
        'total_visitors': total_visitors,
        'positive_reviews': positive_reviews,
        'negative_reviews': negative_reviews,
        'satisfaction_rate': round(satisfaction_rate, 1)
    }

def create_satisfaction_trend_chart(df, city_filter="all", start_date=None, end_date=None):
    """Create a modern satisfaction trend over time chart with enhanced UI/UX"""
    if df.empty:
        # Return empty chart with modern styling
        fig = go.Figure()
        fig.add_annotation(
            text="📊 No data available for the selected filters",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=18, color="#64748b", family="Inter, -apple-system, sans-serif")
        )
        fig.update_layout(
            title={
                'text': "✨ Satisfaction Trend Analysis",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'color': '#1e293b', 'family': 'Inter, -apple-system, sans-serif'}
            },
            xaxis_title="Timeline",
            yaxis_title="Review Volume",
            height=480,
            template="plotly_white",
            plot_bgcolor='rgba(248, 250, 252, 0.4)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig
    
    # Apply filters same as KPI calculation
    filtered_df = df.copy()
    
    # Apply city filter
    if city_filter != "all":
        filtered_df = filtered_df[filtered_df['City'] == city_filter]
    
    # Apply date filter if dates are provided
    if start_date or end_date:
        filtered_df['Review Date'] = pd.to_datetime(filtered_df['Review Date'], errors='coerce')
        
        if start_date:
            start_date = pd.to_datetime(start_date)
            filtered_df = filtered_df[filtered_df['Review Date'] >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            filtered_df = filtered_df[filtered_df['Review Date'] <= end_date]
    
    if filtered_df.empty:
        # Return empty chart with modern styling
        fig = go.Figure()
        fig.add_annotation(
            text="📊 No data available for the selected date range",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=18, color="#64748b", family="Inter, -apple-system, sans-serif")
        )
        fig.update_layout(
            title={
                'text': "✨ Satisfaction Trend Analysis",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'color': '#1e293b', 'family': 'Inter, -apple-system, sans-serif'}
            },
            xaxis_title="Timeline",
            yaxis_title="Review Volume",
            height=480,
            template="plotly_white",
            plot_bgcolor='rgba(248, 250, 252, 0.4)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig
      # Convert Review Date to datetime
    filtered_df['Review Date'] = pd.to_datetime(filtered_df['Review Date'], errors='coerce')
      # Convert sentiment to standardized format for accurate aggregation
    filtered_df['sentiment_numeric'] = filtered_df['sentiment classification'].apply(standardize_sentiment)
    
    # Group by month to show trend over time
    filtered_df['Month'] = filtered_df['Review Date'].dt.to_period('M')
    
    # Calculate reviews by month with corrected aggregation
    monthly_stats = filtered_df.groupby('Month').agg({
        'sentiment_numeric': [
            'count', 
            lambda x: (x == 1).sum(),    # Positive reviews
            lambda x: (x == -1).sum(),   # Negative reviews
            lambda x: (x == 0).sum()     # Neutral reviews
        ]
    }).round(2)
    
    # Flatten column names
    monthly_stats.columns = ['total_reviews', 'positive_reviews', 'negative_reviews', 'neutral_reviews']
    
    # Also calculate satisfaction rate for reference
    monthly_stats['satisfaction_rate'] = (monthly_stats['positive_reviews'] / monthly_stats['total_reviews'] * 100).round(1)
    monthly_stats = monthly_stats.reset_index()
    
    # Convert period back to datetime for plotting
    monthly_stats['Date'] = monthly_stats['Month'].dt.to_timestamp()
      # Create the modern chart with enhanced styling
    fig = go.Figure()
      # Enhanced modern color palette with premium gradients
    modern_colors = {
        'total': '#4c1d95',      # Deep purple for sophistication
        'positive': '#059669',   # Rich emerald for positivity  
        'negative': '#dc2626',   # Classic red for attention
        'accent': '#0ea5e9',     # Sky blue for accents
        'grid': '#e2e8f0',       # Light gray for grid
        'text': '#1e293b'        # Dark slate for text
    }
    
    # Add an invisible trace for unified hover header (shows date and satisfaction rate)
    fig.add_trace(go.Scatter(
        x=monthly_stats['Date'],
        y=[0] * len(monthly_stats),  # Invisible line at y=0
        mode='markers',
        marker=dict(size=0, opacity=0),  # Completely invisible
        showlegend=False,
        hovertemplate='<b style="color: #1e293b;">📅 %{x|%B %Y}</b><br>' +
                     '<span style="color: #059669;">📈 Satisfaction Rate:</span> <b>%{customdata:.1f}%</b><br>' +
                     '<extra></extra>',
        customdata=monthly_stats['satisfaction_rate'],
        name=""  # No name to avoid showing in legend
    ))    # Add total reviews line with enhanced premium styling and smooth curves
    fig.add_trace(go.Scatter(
        x=monthly_stats['Date'],
        y=monthly_stats['total_reviews'],
        name='📊 Total Reviews',
        mode='lines+markers',
        line=dict(
            shape='spline', 
            color=modern_colors['total'], 
            width=3,
            smoothing=1.2,  # Enhanced curve smoothing
            dash='solid'
        ),
        marker=dict(
            size=10,
            color=modern_colors['total'],
            line=dict(color='rgba(255,255,255,0.8)', width=3),
            symbol='circle',
            opacity=0.9
        ),
        fill=None,
        opacity=0.95,
        customdata=monthly_stats[['satisfaction_rate']],
        hovertemplate='<span style="color: #4c1d95; font-weight: bold;">📊 Total Reviews:</span> <b>%{y:,.0f}</b><br>' +
                     '<extra></extra>'
    ))    # Add positive reviews area with premium gradient styling
    fig.add_trace(go.Scatter(
        x=monthly_stats['Date'],
        y=monthly_stats['positive_reviews'],
        name='😊 Positive Reviews',
        mode='lines+markers',
        line=dict(
            shape='spline', 
            color=modern_colors['positive'], 
            width=3,
            smoothing=1.2
        ),
        marker=dict(
            size=8,
            color=modern_colors['positive'],
            line=dict(color='rgba(255,255,255,0.9)', width=2.5),
            symbol='circle',
            opacity=0.9
        ),
        fill='tonexty',
        fillcolor='rgba(5, 150, 105, 0.08)',  # More subtle fill
        opacity=0.95,
        customdata=monthly_stats[['satisfaction_rate']],
        hovertemplate='<span style="color: #059669; font-weight: bold;">😊 Positive Reviews:</span> <b>%{y:,.0f}</b><br>' +
                     '<extra></extra>'
    ))    # Add negative reviews area with refined styling
    fig.add_trace(go.Scatter(
        x=monthly_stats['Date'],
        y=monthly_stats['negative_reviews'],
        name='😞 Negative Reviews',
        mode='lines+markers',
        line=dict(
            shape='spline', 
            color=modern_colors['negative'], 
            width=3,
            smoothing=1.2
        ),
        marker=dict(
            size=8,
            color=modern_colors['negative'],
            line=dict(color='rgba(255,255,255,0.9)', width=2.5),
            symbol='circle',
            opacity=0.9
        ),
        fill='tozeroy',
        fillcolor='rgba(220, 38, 38, 0.06)',  # More subtle negative fill
        opacity=0.95,
        customdata=monthly_stats[['satisfaction_rate']],
        hovertemplate='<span style="color: #dc2626; font-weight: bold;">😞 Negative Reviews:</span> <b>%{y:,.0f}</b><br>' +
                     '<extra></extra>'
    ))
    
    # Add modern threshold lines with enhanced styling
    # Excellent threshold (80%)
    fig.add_hline(
        y=80,
        line_dash="dot",
        line_color="#059669",
        line_width=3,
        opacity=0.8,
        annotation_text="⭐ Excellent (80%+)",
        annotation_position="top right",        annotation=dict(
            font=dict(color="#059669", size=13, family="Inter, -apple-system, sans-serif", weight="bold"),
            bgcolor="rgba(5, 150, 105, 0.15)",
            bordercolor="#059669",
            borderwidth=2,
            borderpad=8
        )
    )
    
    # Good satisfaction threshold (70%)
    fig.add_hline(
        y=70,
        line_dash="dot",
        line_color="#06d6a0",
        line_width=2.5,
        opacity=0.7,
        annotation_text="✅ Good (70%+)",
        annotation_position="bottom right",        annotation=dict(
            font=dict(color="#06d6a0", size=12, family="Inter, -apple-system, sans-serif", weight="normal"),
            bgcolor="rgba(6, 214, 160, 0.12)",
            bordercolor="#06d6a0",
            borderwidth=1.5,
            borderpad=6
        )
    )
    
    # Poor satisfaction threshold (50%)
    fig.add_hline(
        y=50,
        line_dash="dot",
        line_color="#f72585",
        line_width=2.5,
        opacity=0.7,
        annotation_text="⚠️ Needs Improvement (Below 50%)",
        annotation_position="top left",        annotation=dict(
            font=dict(color="#f72585", size=12, family="Inter, -apple-system, sans-serif", weight="normal"),
            bgcolor="rgba(247, 37, 133, 0.12)",
            bordercolor="#f72585",
            borderwidth=1.5,
            borderpad=6
        )
    )
    
    # Add dynamic average line
    avg_satisfaction = monthly_stats['satisfaction_rate'].mean()
    fig.add_hline(
        y=avg_satisfaction,
        line_dash="dash",
        line_color="#64748b",
        line_width=2,
        opacity=0.6,
        annotation_text=f"📊 Average: {avg_satisfaction:.1f}%",
        annotation_position="bottom left",        annotation=dict(
            font=dict(color="#64748b", size=11, family="Inter, -apple-system, sans-serif"),
            bgcolor="rgba(100, 116, 139, 0.1)",
            bordercolor="#64748b",
            borderwidth=1,
            borderpad=5
        )
    )    # Update layout with premium design and disabled animations
    fig.update_layout(
        title={
            'text': "✨ Satisfaction Trend Analysis - Premium Review Insights",
            'x': 0.5,
            'xanchor': 'center',
            'font': {
                'size': 26, 
                'color': '#1e293b', 
                'family': 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
                'weight': 'bold'
            }
        },
        xaxis_title="📅 Timeline",
        yaxis_title="📈 Review Volume",
        height=520,
        template=None,
        hovermode='x unified',
        plot_bgcolor='rgba(248, 250, 252, 0.4)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=85, r=85, t=110, b=85),
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", 
            size=14, 
            color="#475569"
        ),
        showlegend=True,
        # Disable all animations and interactions for static display
        transition={'duration': 0},        dragmode=False,        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.98)",
            bordercolor="rgba(148, 163, 184, 0.4)",
            borderwidth=1.5,
            itemsizing="constant",
            itemwidth=30,
            font=dict(
                family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                size=13,
                color="#1e293b",
                weight=500
            ),
            tracegroupgap=15
        ),        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.98)",
            bordercolor="rgba(148, 163, 184, 0.5)",
            font=dict(
                family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                size=14,
                color="#0f172a",
                weight=500
            ),
            align="left",
            namelength=0
        ))
    
    # Update axes with modern styling
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(148, 163, 184, 0.2)',
        showline=True,
        linewidth=2,        linecolor='rgba(148, 163, 184, 0.4)',
        tickfont=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            size=12,
            color="#64748b",
            weight=400
        ),
        title_font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            size=14,
            color="#1e293b",
            weight=600
        ),
        tickformat='%b %Y'
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(148, 163, 184, 0.2)',
        showline=True,
        linewidth=2,
        linecolor='rgba(148, 163, 184, 0.4)',
        rangemode='tozero',        tickfont=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            size=12,
            color="#64748b",
            weight=400
        ),
        title_font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            size=14,
            color="#1e293b",
            weight=600
        ),
        tickformat='.0f'
    )
    
    return fig

def create_word_frequency_chart(df, city_filter="all", start_date=None, end_date=None):
    """Create a chart showing top 10 positive and negative words"""
    if df.empty:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for the selected filters",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title="Top Mentioned Words - Positive vs Negative",
            xaxis_title="Frequency",
            yaxis_title="Words",
            height=400,
            template="plotly_white"
        )
        return fig
    
    # Apply filters same as KPI calculation
    filtered_df = df.copy()
    
    # Apply city filter
    if city_filter != "all":
        filtered_df = filtered_df[filtered_df['City'] == city_filter]
    
    # Apply date filter if dates are provided
    if start_date or end_date:
        filtered_df['Review Date'] = pd.to_datetime(filtered_df['Review Date'], errors='coerce')
        
        if start_date:
            start_date = pd.to_datetime(start_date)
            filtered_df = filtered_df[filtered_df['Review Date'] >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            filtered_df = filtered_df[filtered_df['Review Date'] <= end_date]
    
    if filtered_df.empty:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for the selected filters",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title="Top Mentioned Words - Positive vs Negative",
            xaxis_title="Frequency",
            yaxis_title="Words",
            height=400,
            template="plotly_white"
        )
        return fig
      # Extract positive and negative words with consistent counting logic
    positive_counter = Counter()
    negative_counter = Counter()
    
    for _, row in filtered_df.iterrows():
        # Parse positive tokens - Handle PostgreSQL array format
        pos_tokens_value = row['positive tokens']
        if pos_tokens_value is not None and not (isinstance(pos_tokens_value, float) and pd.isna(pos_tokens_value)):
            try:
                # Handle different possible formats from PostgreSQL
                if isinstance(pos_tokens_value, list):
                    # Already a list from PostgreSQL
                    word_tokens = [token.lower().strip() for token in pos_tokens_value if token and token.strip()]
                elif isinstance(pos_tokens_value, str):
                    # String representation of array
                    if pos_tokens_value.startswith('{') and pos_tokens_value.endswith('}'):
                        # PostgreSQL array format: {word1,word2,word3}
                        clean_tokens = pos_tokens_value[1:-1]  # Remove { and }
                        word_tokens = [token.lower().strip().strip('"').strip("'") for token in clean_tokens.split(',') if token.strip()]
                    else:                        # Try to evaluate as Python list
                        pos_tokens = ast.literal_eval(pos_tokens_value)
                        if isinstance(pos_tokens, list):
                            word_tokens = [token.lower().strip() for token in pos_tokens if token and token.strip()]
                        else:
                            continue
                else:
                    continue
            except:
                continue
            
            # Count each word occurrence in this review (same logic as modal)
            for word in set(word_tokens):  # Use set to avoid double counting in same token list
                word_count = word_tokens.count(word)
                positive_counter[word] += word_count
        
        # Parse negative tokens - Handle PostgreSQL array format
        neg_tokens_value = row['negative tokens']
        if neg_tokens_value is not None and not (isinstance(neg_tokens_value, float) and pd.isna(neg_tokens_value)):
            try:
                # Handle different possible formats from PostgreSQL
                if isinstance(neg_tokens_value, list):
                    # Already a list from PostgreSQL
                    word_tokens = [token.lower().strip() for token in neg_tokens_value if token and token.strip()]
                elif isinstance(neg_tokens_value, str):
                    # String representation of array
                    if neg_tokens_value.startswith('{') and neg_tokens_value.endswith('}'):
                        # PostgreSQL array format: {word1,word2,word3}
                        clean_tokens = neg_tokens_value[1:-1]  # Remove { and }
                        word_tokens = [token.lower().strip().strip('"').strip("'") for token in clean_tokens.split(',') if token.strip()]
                    else:                        # Try to evaluate as Python list
                        neg_tokens = ast.literal_eval(neg_tokens_value)
                        if isinstance(neg_tokens, list):
                            word_tokens = [token.lower().strip() for token in neg_tokens if token and token.strip()]
                        else:
                            continue
                else:
                    continue
            except:
                continue
            
            # Count each word occurrence in this review (same logic as modal)
            for word in set(word_tokens):  # Use set to avoid double counting in same token list
                word_count = word_tokens.count(word)
                negative_counter[word] += word_count
    
    # Calculate total mentions for each word (positive + negative)
    all_words = set(positive_counter.keys()) | set(negative_counter.keys())
    total_counter = {}
    for word in all_words:
        total_counter[word] = positive_counter.get(word, 0) + negative_counter.get(word, 0)
    
    # Get top 10 words by total mentions (across both sentiments)
    top_words_by_total = sorted(total_counter.items(), key=lambda x: x[1], reverse=True)[:10]
    top_words = [word for word, count in top_words_by_total]
      # Create the chart with the same words in both sections
    fig = go.Figure()
    
    if top_words:        # Positive section: show positive counts for all top words
        pos_counts_for_top_words = [positive_counter.get(word, 0) for word in top_words]
        fig.add_trace(go.Bar(
            y=top_words,
            x=pos_counts_for_top_words,  # Show actual positive counts for top words
            name='Positive Words',
            orientation='h',
            marker=dict(color='#10b981'),
            hovertemplate='<b>%{y}</b><br>Positive Mentions: %{x}<br>Total Mentions: %{customdata[1]}<br>Click for detailed insights<br><extra></extra>',
            text=pos_counts_for_top_words,
            textposition='outside',
            customdata=[['positive', total_counter[word]] for word in top_words]  # Store word type and total count
        ))
          # Negative section: show negative counts for the same top words
        neg_counts_for_top_words = [negative_counter.get(word, 0) for word in top_words]
        neg_counts_negative = [-count for count in neg_counts_for_top_words]  # Negative for visual display
        fig.add_trace(go.Bar(
            y=top_words,
            x=neg_counts_negative,  # Show actual negative counts (as negative values)
            name='Negative Words',
            orientation='h',
            marker=dict(color='#ef4444'),
            hovertemplate='<b>%{y}</b><br>Negative Mentions: %{text}<br>Total Mentions: %{customdata[1]}<br>Click for detailed insights<br><extra></extra>',
            text=neg_counts_for_top_words,  # Show positive numbers in hover
            textposition='outside',
            customdata=[['negative', total_counter[word]] for word in top_words]  # Store word type and total count
        ))
    else:
        # No words found - show message
        fig.add_annotation(
            text="No word frequency data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
    
    # Update layout
    fig.update_layout(
        title={
            'text': "Top 10 Most Mentioned Words",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2c3e50'}
        },
        xaxis_title="Frequency",
        yaxis_title="Words",
        height=500,
        template="plotly_white",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=120, r=60, t=80, b=60),
        font=dict(family="Arial, sans-serif", size=12, color="#2c3e50"),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1
        ),
        barmode='overlay',
        hovermode='closest'
    )
    
    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(0,0,0,0.1)',
        showline=True,
        linewidth=1,
        linecolor='rgba(0,0,0,0.2)',
        zeroline=True,
        zerolinecolor='rgba(0,0,0,0.3)',
        zerolinewidth=2
    )
    
    fig.update_yaxes(
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor='rgba(0,0,0,0.2)',
        tickfont=dict(size=11)
    )
    
    # Add vertical line at zero
    fig.add_vline(x=0, line_width=2, line_color="rgba(0,0,0,0.3)")
    return fig

def create_nationality_chart(df, city_filter="all", start_date=None, end_date=None):
    """Create a chart showing top 10 nationalities visiting cities"""
    if df.empty:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for the selected filters",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title="Top 10 Nationalities",
            xaxis_title="Number of Visitors",
            yaxis_title="Nationality",
            height=400,
            template="plotly_white"
        )
        return fig
    
    # Apply filters same as other functions
    filtered_df = df.copy()
    
    # Apply city filter
    if city_filter != "all":
        filtered_df = filtered_df[filtered_df['City'] == city_filter]
    
    # Apply date filter if dates are provided
    if start_date or end_date:
        filtered_df['Review Date'] = pd.to_datetime(filtered_df['Review Date'], errors='coerce')
        
        if start_date:
            start_date = pd.to_datetime(start_date)
            filtered_df = filtered_df[filtered_df['Review Date'] >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            filtered_df = filtered_df[filtered_df['Review Date'] <= end_date]
    
    if filtered_df.empty:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for the selected filters",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title="Top 10 Nationalities",
            xaxis_title="Number of Visitors",
            yaxis_title="Nationality",
            height=400,
            template="plotly_white"
        )
        return fig
      # Count nationalities - use the correct column name from database
    nationality_col = None
    for col in filtered_df.columns:
        if 'nationality' in col.lower():
            nationality_col = col
            break
    
    if nationality_col is None:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No nationality column found in data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title="Top 10 Nationalities",
            xaxis_title="Number of Visitors",
            yaxis_title="Nationality",
            height=400,
            template="plotly_white"
        )
        return fig
    
    nationality_counts = filtered_df[nationality_col].value_counts().head(10)
    
    if nationality_counts.empty:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No nationality data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title="Top 10 Nationalities",
            xaxis_title="Number of Visitors",
            yaxis_title="Nationality",
            height=400,
            template="plotly_white"
        )
        return fig
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=nationality_counts.values,
        y=nationality_counts.index,
        orientation='h',
        marker=dict(
            color='#3b82f6',
            line=dict(color='rgba(255,255,255,0.8)', width=1)
        ),
        hovertemplate='<b>%{y}</b><br>Visitors: %{x:,}<br>Click for detailed insights<br><extra></extra>',
        text=nationality_counts.values,
        textposition='outside',
        name='Nationalities'
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': "Top 10 Nationalities",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2c3e50'}
        },
        xaxis_title="Number of Visitors",
        yaxis_title="Nationality",
        height=500,
        template="plotly_white",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=120, r=60, t=80, b=60),
        font=dict(family="Arial, sans-serif", size=12, color="#2c3e50"),
        showlegend=False,
        hovermode='closest'
    )
    
    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(0,0,0,0.1)',
        showline=True,
        linewidth=1,
        linecolor='rgba(0,0,0,0.2)',
        zeroline=True,
        zerolinecolor='rgba(0,0,0,0.3)',
        zerolinewidth=2
    )
    
    fig.update_yaxes(
        showgrid=False,
        showline=True,
        linewidth=1,        linecolor='rgba(0,0,0,0.2)',
        tickfont=dict(size=11),
        categoryorder='total ascending'  # Sort by value ascending (largest at top)
    )
    
    return fig

def create_city_satisfaction_chart(df, city_filter="all", start_date=None, end_date=None):
    """Create a chart comparing satisfaction rates across cities (only when city_filter='all')"""
    # Only show this chart when "All Cities" is selected
    if city_filter != "all":
        # Return a completely empty figure with minimal height when specific city is selected
        fig = go.Figure()
        fig.update_layout(
            height=10,  # Minimal height (Plotly minimum), CSS will hide it anyway
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            showlegend=False
        )
        return fig
    
    if df.empty:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for the selected filters",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title="City Satisfaction Comparison",
            xaxis_title="Cities",
            yaxis_title="Satisfaction Rate (%)",
            height=450,
            template="plotly_white"
        )
        return fig
    
    # Apply date filter if dates are provided
    filtered_df = df.copy()
    if start_date or end_date:
        filtered_df['Review Date'] = pd.to_datetime(filtered_df['Review Date'], errors='coerce')
        
        if start_date:
            start_date = pd.to_datetime(start_date)
            filtered_df = filtered_df[filtered_df['Review Date'] >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            filtered_df = filtered_df[filtered_df['Review Date'] <= end_date]
    
    if filtered_df.empty:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for the selected date range",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title="City Satisfaction Comparison",
            xaxis_title="Cities",
            yaxis_title="Satisfaction Rate (%)",
            height=450,
            template="plotly_white"
        )
        return fig
      # Calculate satisfaction rate by city
    city_stats = []
    cities = filtered_df['City'].unique()
    
    for city in cities:
        city_df = filtered_df[filtered_df['City'] == city]
        total_reviews = len(city_df)
        
        # Apply sentiment standardization using the shared function
        city_df = city_df.copy()
        city_df['sentiment_numeric'] = city_df['sentiment classification'].apply(standardize_sentiment)
        
        # Count positive reviews using standardized values
        positive_reviews = len(city_df[city_df['sentiment_numeric'] == 1])
        satisfaction_rate = (positive_reviews / total_reviews * 100) if total_reviews > 0 else 0
        
        city_stats.append({
            'city': city,
            'satisfaction_rate': satisfaction_rate,
            'total_reviews': total_reviews,
            'positive_reviews': positive_reviews
        })
    
    # Sort cities by satisfaction rate (descending)
    city_stats.sort(key=lambda x: x['satisfaction_rate'], reverse=True)
    
    # Prepare data for plotting
    cities_list = [stat['city'] for stat in city_stats]
    satisfaction_rates = [stat['satisfaction_rate'] for stat in city_stats]
    total_reviews = [stat['total_reviews'] for stat in city_stats]
    positive_reviews = [stat['positive_reviews'] for stat in city_stats]
      # Create enhanced color scale with gradients
    colors = []
    color_names = []
    border_colors = []
    for rate in satisfaction_rates:
        if rate >= 80:
            colors.append('#059669')  # Darker green for excellent satisfaction (80%+)
            color_names.append('Excellent')
            border_colors.append('#047857')
        elif rate >= 70:
            colors.append('#10b981')  # Green for very good satisfaction (70-80%)
            color_names.append('Very Good')
            border_colors.append('#059669')
        elif rate >= 60:
            colors.append('#34d399')  # Light green for good satisfaction (60-70%)
            color_names.append('Good')
            border_colors.append('#10b981')
        elif rate >= 50:
            colors.append('#fbbf24')  # Yellow for average satisfaction (50-60%)
            color_names.append('Average')
            border_colors.append('#f59e0b')
        elif rate >= 40:
            colors.append('#fb923c')  # Orange for below average (40-50%)
            color_names.append('Below Average')
            border_colors.append('#ea580c')
        else:
            colors.append('#ef4444')  # Red for poor satisfaction (<40%)
            color_names.append('Poor')
            border_colors.append('#dc2626')
    
    # Truncate long city names for better display
    display_cities = []
    for city in cities_list:
        if len(city) > 12:
            display_cities.append(city[:10] + "...")
        else:
            display_cities.append(city)
    
    # Create the enhanced bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=display_cities,
        y=satisfaction_rates,
        marker=dict(
            color=colors,
            line=dict(color=border_colors, width=2),
            pattern=dict(
                shape="",  # No pattern for solid colors
                solidity=0.8
            )
        ),
        hovertemplate='<b>%{customdata[3]}</b><br>' +  # Full city name in hover
                     '<span style="color:%{customdata[4]}">● %{customdata[5]}</span><br>' +
                     'Satisfaction Rate: <b>%{y:.1f}%</b><br>' +
                     'Total Reviews: <b>%{customdata[0]:,}</b><br>' +
                     'Positive Reviews: <b>%{customdata[1]:,}</b><br>' +
                     'Negative Reviews: <b>%{customdata[2]:,}</b><br>' +
                     '<extra></extra>',
        customdata=[[total, positive, total-positive, full_city, color, color_name] 
                   for total, positive, full_city, color, color_name in 
                   zip(total_reviews, positive_reviews, cities_list, colors, color_names)],
        text=[f'{rate:.1f}%' for rate in satisfaction_rates],
        textposition='outside',
        textfont=dict(size=12, color='#2c3e50', family="Arial Black"),
        name='Satisfaction Rate',
        width=0.6  # Make bars slightly thinner for better spacing
    ))
      # Add enhanced satisfaction threshold lines with better styling
    fig.add_hline(
        y=80,
        line_dash="dot",
        line_color="#059669",
        line_width=2.5,
        annotation_text="★ Excellent (80%+)",
        annotation_position="bottom right",
        annotation=dict(
            font=dict(color="#047857", size=12, family="Arial Black"),
            bgcolor="rgba(5, 150, 105, 0.15)",
            bordercolor="#059669",
            borderwidth=2,
            borderpad=4
        )
    )
    
    fig.add_hline(
        y=70,
        line_dash="dot",
        line_color="#10b981",
        line_width=2,
        annotation_text="✓ Very Good (70%+)",
        annotation_position="top right",
        annotation=dict(
            font=dict(color="#059669", size=11, family="Arial"),
            bgcolor="rgba(16, 185, 129, 0.12)",
            bordercolor="#10b981",
            borderwidth=1.5,
            borderpad=3
        )
    )
    
    fig.add_hline(
        y=60,
        line_dash="dot",
        line_color="#34d399",
        line_width=1.5,
        annotation_text="○ Good (60%+)",
        annotation_position="bottom left",
        annotation=dict(
            font=dict(color="#10b981", size=10, family="Arial"),
            bgcolor="rgba(52, 211, 153, 0.1)",
            bordercolor="#34d399",
            borderwidth=1,
            borderpad=2
        )
    )
    
    fig.add_hline(
        y=50,
        line_dash="dot",
        line_color="#fbbf24",
        line_width=1.5,
        annotation_text="~ Average (50%+)",
        annotation_position="top left",
        annotation=dict(
            font=dict(color="#f59e0b", size=10, family="Arial"),
            bgcolor="rgba(251, 191, 36, 0.1)",
            bordercolor="#fbbf24",
            borderwidth=1,
            borderpad=2
        )
    )
    
    # Calculate overall average
    overall_avg = sum(satisfaction_rates) / len(satisfaction_rates) if satisfaction_rates else 0
    fig.add_hline(
        y=overall_avg,
        line_dash="dash",
        line_color="#6b7280",
        line_width=2,
        annotation_text=f"Overall Average: {overall_avg:.1f}%",
        annotation_position="top right",        annotation=dict(
            font=dict(color="#6b7280", size=11, family="Arial, sans-serif"),
            bgcolor="rgba(107, 114, 128, 0.1)",
            bordercolor="#6b7280",
            borderwidth=1,
            borderpad=2
        )
    )
    
    # Update layout with enhanced styling and better spacing
    fig.update_layout(
        title={
            'text': "🏆 Tourist Satisfaction by City - Which Cities Do Tourists Love Most?",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#1f2937', 'family': 'Arial Black'}
        },
        xaxis_title="Cities",
        yaxis_title="Satisfaction Rate (%)",
        height=520,  # Increased height to accommodate better spacing
        template="plotly_white",
        plot_bgcolor='rgba(248, 250, 252, 0.5)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=80, r=80, t=120, b=140),  # Increased bottom margin for legend
        font=dict(family="Arial, sans-serif", size=12, color="#374151"),        showlegend=False,
        hovermode='closest',
        annotations=[
            dict(
                text=CITY_RATING_LEGEND_HTML,
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=-0.18,
                xanchor='center', yanchor='top',
                font=dict(size=12, color=CHART_TEXT_COLORS['secondary'], family="Arial"),
                bgcolor=CHART_BG['overlay'],
                bordercolor="#d1d5db",
                borderwidth=1,
                borderpad=8
            )
        ]
    )
    
    # Update axes with improved styling and text rotation
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(156, 163, 175, 0.3)',
        showline=True,
        linewidth=2,
        linecolor='#9ca3af',
        tickangle=-45,  # Angle the text to prevent overlapping
        tickfont=dict(size=11, color="#374151", family="Arial"),
        title_font=dict(size=14, color="#1f2937", family="Arial Black"),
        tickmode='array',
        tickvals=list(range(len(display_cities))),
        ticktext=display_cities
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(156, 163, 175, 0.3)',
        showline=True,
        linewidth=2,
        linecolor='#9ca3af',
        range=[0, min(105, max(satisfaction_rates) + 10)],  # Dynamic range based on data
        tickfont=dict(size=11, color="#374151", family="Arial"),
        title_font=dict(size=14, color="#1f2937", family="Arial Black"),
        ticksuffix="%"
    )
    
    return fig

def analyze_hotels_by_sentiment(hotels_with_word, selected_word):
    """Analyze hotels by sentiment for the selected word"""
    from collections import defaultdict
    
    # Group hotels by sentiment context
    hotel_sentiment_data = defaultdict(lambda: {
        'positive_mentions': 0,
        'negative_mentions': 0,
        'total_mentions': 0,
        'reviews_count': 0,
        'city': '',
        'overall_sentiment_avg': 0,
        'sentiment_scores': []
    })
    
    for hotel_info in hotels_with_word:
        hotel_name = hotel_info['name']
        hotel_sentiment_data[hotel_name]['city'] = hotel_info['city']
        hotel_sentiment_data[hotel_name]['total_mentions'] += hotel_info['mentions']
        hotel_sentiment_data[hotel_name]['reviews_count'] += 1
        
        # Track overall sentiment scores for this hotel
        if pd.notna(hotel_info['overall_sentiment']):
            try:
                sentiment_score = float(hotel_info['overall_sentiment'])
                hotel_sentiment_data[hotel_name]['sentiment_scores'].append(sentiment_score)
            except:
                pass
        
        # Count mentions by sentiment context
        if 'positive' in hotel_info['sentiment_contexts']:
            hotel_sentiment_data[hotel_name]['positive_mentions'] += hotel_info['mentions']
        if 'negative' in hotel_info['sentiment_contexts']:
            hotel_sentiment_data[hotel_name]['negative_mentions'] += hotel_info['mentions']
    
    # Calculate average sentiment scores
    for hotel_name in hotel_sentiment_data:
        scores = hotel_sentiment_data[hotel_name]['sentiment_scores']
        if scores:
            hotel_sentiment_data[hotel_name]['overall_sentiment_avg'] = sum(scores) / len(scores)
    
    # Create sorted lists for positive and negative contexts
    positive_hotels = []
    negative_hotels = []
    
    for hotel_name, data in hotel_sentiment_data.items():
        hotel_summary = {
            'name': hotel_name,
            'city': data['city'],
            'total_mentions': data['total_mentions'],
            'positive_mentions': data['positive_mentions'],
            'negative_mentions': data['negative_mentions'],
            'reviews_count': data['reviews_count'],
            'overall_sentiment_avg': round(data['overall_sentiment_avg'], 2),
            'dominant_sentiment': 'positive' if data['positive_mentions'] > data['negative_mentions'] else 'negative'
        }
        
        # Categorize based on where the word appears most frequently
        if data['positive_mentions'] > 0:
            positive_hotels.append({
                **hotel_summary,
                'sentiment_score': data['positive_mentions'] / data['total_mentions'] if data['total_mentions'] > 0 else 0
            })
        
        if data['negative_mentions'] > 0:
            negative_hotels.append({
                **hotel_summary,
                'sentiment_score': data['negative_mentions'] / data['total_mentions'] if data['total_mentions'] > 0 else 0
            })
    
    # Sort by mentions and sentiment relevance
    positive_hotels.sort(key=lambda x: (x['positive_mentions'], x['sentiment_score']), reverse=True)
    negative_hotels.sort(key=lambda x: (x['negative_mentions'], x['sentiment_score']), reverse=True)
    
    return {
        'top_positive_hotels': positive_hotels[:5],
        'top_negative_hotels': negative_hotels[:5],
        'total_hotels_analyzed': len(hotel_sentiment_data)
    }

def analyze_word_details(df, selected_word, word_type, city_filter="all", start_date=None, end_date=None):
    """Analyze detailed statistics for a specific word"""
    # Apply filters same as other functions
    filtered_df = df.copy()
    
    # Apply city filter
    if city_filter != "all":
        filtered_df = filtered_df[filtered_df['City'] == city_filter]
    
    # Apply date filter if dates are provided
    if start_date or end_date:
        filtered_df['Review Date'] = pd.to_datetime(filtered_df['Review Date'], errors='coerce')
        
        if start_date:
            start_date = pd.to_datetime(start_date)
            filtered_df = filtered_df[filtered_df['Review Date'] >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            filtered_df = filtered_df[filtered_df['Review Date'] <= end_date]
    
    if filtered_df.empty:
        return {
            'word': selected_word,
            'type': word_type,
            'total_mentions': 0,
            'reviews_count': 0,
            'cities_mentioned': [],
            'monthly_trend': [],
            'associated_words': [],
            'sentiment_score': 0,
            'insights': [],
            'recommendations': []        }
    
    # Count total mentions and reviews containing the word
    total_mentions = 0
    reviews_with_word = []
    cities_mentioned = []
    hotels_with_word = []
    positive_mentions = 0
    negative_mentions = 0
    
    # We'll analyze both positive and negative mentions regardless of word_type
    # to get comprehensive sentiment breakdown
    
    for idx, row in filtered_df.iterrows():# Check both positive and negative tokens for comprehensive analysis
        word_found_in_review = False
        word_mentions_in_review = 0
        sentiment_contexts = []
        
        for token_type, token_column in [('positive', 'positive tokens'), ('negative', 'negative tokens')]:
            token_value = row[token_column]
            if token_value is not None and not (isinstance(token_value, float) and pd.isna(token_value)) and isinstance(token_value, str) and token_value.strip():
                try:
                    tokens = ast.literal_eval(token_value)
                    if isinstance(tokens, list):
                        word_tokens = [token.lower().strip() for token in tokens if token.strip()]
                    else:
                        continue
                except:
                    if isinstance(token_value, str):
                        clean_tokens = token_value.replace('[', '').replace(']', '').replace("'", '').replace('"', '')
                        word_tokens = [token.lower().strip() for token in clean_tokens.split(',') if token.strip()]
                    else:
                        continue
                
                if selected_word.lower() in word_tokens:
                    word_count = word_tokens.count(selected_word.lower())
                    word_mentions_in_review += word_count
                    word_found_in_review = True
                    sentiment_contexts.append(token_type)
                    
                    # Count by sentiment type
                    if token_type == 'positive':
                        positive_mentions += word_count
                    else:
                        negative_mentions += word_count
        
        if word_found_in_review:
            total_mentions += word_mentions_in_review
              # Add hotel information
            hotel_column = 'Hotel Name' if 'Hotel Name' in row else 'hotel_name'
            hotel_info = {
                'name': row[hotel_column] if hotel_column in row else 'Unknown Hotel',
                'city': row['City'],
                'mentions': word_mentions_in_review,
                'sentiment_contexts': sentiment_contexts,
                'review_date': row['Review Date'],
                'overall_sentiment': row.get('sentiment classification', 0)
            }
            hotels_with_word.append(hotel_info)
            
            # Add to reviews list
            review_info = dict(row)
            review_info['word_mentions'] = word_mentions_in_review
            review_info['sentiment_contexts'] = sentiment_contexts
            reviews_with_word.append(review_info)
            
            if row['City'] not in cities_mentioned:
                cities_mentioned.append(row['City'])
    
    # Calculate monthly trend
    monthly_trend = []
    if reviews_with_word:
        review_dates = [pd.to_datetime(review['Review Date'], errors='coerce') for review in reviews_with_word]
        review_dates = [date for date in review_dates if pd.notna(date)]
        
        if review_dates:
            df_temp = pd.DataFrame({'date': review_dates})
            df_temp['month'] = df_temp['date'].dt.to_period('M')
            monthly_counts = df_temp.groupby('month').size().reset_index(name='count')
            monthly_trend = [{'month': str(period), 'count': count} for period, count in zip(monthly_counts['month'], monthly_counts['count'])]
      # Find associated words (co-occurring words in the same reviews)
    associated_words = []
    all_associated = []
    
    for review in reviews_with_word:
        # Get words from both positive and negative tokens
        for col in ['positive tokens', 'negative tokens']:
            if pd.notna(review[col]) and review[col].strip():
                try:
                    tokens = ast.literal_eval(review[col])
                    if isinstance(tokens, list):
                        tokens = [token.lower().strip() for token in tokens if token.strip() and token.lower() != selected_word.lower()]
                        all_associated.extend(tokens)
                except:
                    if isinstance(review[col], str):
                        clean_tokens = review[col].replace('[', '').replace(']', '').replace("'", '').replace('"', '')
                        tokens = [token.lower().strip() for token in clean_tokens.split(',') if token.strip() and token.lower() != selected_word.lower()]
                        all_associated.extend(tokens)
    
    if all_associated:
        associated_counter = Counter(all_associated)
        associated_words = [{'word': word, 'count': count} for word, count in associated_counter.most_common(10)]
    
    # Analyze hotels by sentiment
    hotel_analysis = analyze_hotels_by_sentiment(hotels_with_word, selected_word)
    
    # Calculate sentiment impact score (for government insights)
    sentiment_score = 0
    if word_type == 'positive':
        sentiment_score = min(100, (total_mentions / len(filtered_df)) * 100) if len(filtered_df) > 0 else 0
    else:
        sentiment_score = -min(100, (total_mentions / len(filtered_df)) * 100) if len(filtered_df) > 0 else 0
    
    # Generate government-focused insights
    insights = generate_government_insights(selected_word, word_type, total_mentions, len(reviews_with_word), 
                                          len(cities_mentioned), cities_mentioned, sentiment_score)
      # Generate actionable recommendations
    recommendations = generate_government_recommendations(selected_word, word_type, total_mentions, 
                                                        len(reviews_with_word), cities_mentioned, associated_words)
    
    return {
        'word': selected_word,
        'type': word_type,
        'total_mentions': total_mentions,
        'positive_mentions': positive_mentions,
        'negative_mentions': negative_mentions,
        'reviews_count': len(reviews_with_word),
        'cities_mentioned': cities_mentioned,
        'monthly_trend': monthly_trend,
        'associated_words': associated_words,
        'sentiment_score': round(sentiment_score, 1),
        'hotel_analysis': hotel_analysis,
        'insights': insights,
        'recommendations': recommendations
    }

def generate_government_insights(word, word_type, mentions, reviews, cities_count, cities, sentiment_score):
    """Generate enhanced government-focused insights for the selected word with real-time data analysis"""
    insights = []
    
    # Load current data for dynamic analysis
    try:
        df = load_data()
        total_reviews = len(df)
          # Only include insights based on significant data
        if total_reviews > 0 and reviews >= 10:  # Only for statistically meaningful data
            review_impact_percentage = (reviews / total_reviews * 100)
            
            # Market penetration insight - only if significant
            if review_impact_percentage >= 2.0:  # At least 2% of all reviews
                insights.append(f"**Market Impact**: '{word}' affects {review_impact_percentage:.1f}% of tourist experiences ({reviews:,} reviews). This represents significant market exposure requiring attention.")
            
            # Geographic distribution - only for multi-city impacts
            if cities_count > 1:
                insights.append(f"**Geographic Scope**: Issue spans {cities_count} destinations: {', '.join(cities[:3])}{'...' if cities_count > 3 else ''}. Coordination across multiple locations required.")
            
            # Frequency-based priority - only for high-frequency issues
            if mentions >= 30:
                insights.append(f"**High Frequency Alert**: {mentions:,} mentions indicate this is a recurring theme requiring systematic intervention.")
        
        # Service quality and trends analysis
        if cities_count > 3:
            coordination_cost = cities_count * 15000  # Estimated coordination cost per city
            insights.append(f"� **National Scale Challenge**: Impacts {cities_count} destinations including {', '.join(cities[:2])} and {cities_count-2} others. Requires ${coordination_cost:,.0f} inter-governmental coordination budget and unified tourism standards.")
        elif cities_count > 1:
            insights.append(f"🏛️ **Regional Coordination Needed**: Spans {', '.join(cities)}, requiring standardized protocols and shared resource allocation. Perfect for testing scalable tourism policies.")
        else:
            city_name = cities[0] if cities else 'key tourism area'
            insights.append(f"📍 **Concentrated Impact Zone**: Focused in {city_name} - ideal for rapid pilot implementation and measurable results within 30-60 days.")
        
        # Real-time trend analysis using actual data
        try:
            # Search for the word in review content (handling different column names)
            review_cols = [col for col in df.columns if 'review' in col.lower() and 'translated' in col.lower()]
            if review_cols:
                word_df = df[df[review_cols[0]].str.contains(word, case=False, na=False)]
                if len(word_df) > 0:
                    review_dates = pd.to_datetime(word_df['Review Date'], errors='coerce').dropna()
                    if len(review_dates) > 0:
                        # Calculate recent trend (last 3 months vs previous 3 months)
                        max_date = review_dates.max()
                        recent_cutoff = max_date - pd.DateOffset(months=3)
                        previous_cutoff = max_date - pd.DateOffset(months=6)
                        
                        recent_count = len(review_dates[review_dates >= recent_cutoff])
                        previous_count = len(review_dates[(review_dates >= previous_cutoff) & (review_dates < recent_cutoff)])
                        
                        if previous_count > 0:
                            trend_change = ((recent_count - previous_count) / previous_count) * 100
                            if abs(trend_change) > 20:
                                trend_direction = "↗️ increasing" if trend_change > 0 else "↘️ decreasing"
                                insights.append(f"📈 **Emerging Trend Alert**: '{word}' mentions are {trend_direction} by {abs(trend_change):.0f}% over the last 3 months. {('Urgent intervention required' if word_type == 'negative' else 'Opportunity to capitalize') if abs(trend_change) > 50 else ('Monitor closely' if word_type == 'negative' else 'Scale successful practices')}.")
        except Exception:
            pass  # Skip trend analysis if data processing fails
        
        # Service quality assessment based on data only
        if word_type == 'positive':
            if mentions >= 20:
                insights.append(f"**Excellence Indicator**: Consistently positive feedback suggests this is a competitive advantage to maintain and scale.")
        else:  # negative
            if mentions >= 15:
                insights.append(f"**Service Gap**: Repeated negative mentions indicate systematic service delivery issues requiring operational improvements.")        # Sector-specific government insights based on word categories
            
            # Marketing leverage analysis
            marketing_roi = min(mentions * 1000, 500000)  # Estimated marketing value per mention
            insights.append(f"� **Marketing Leverage**: '{word}' excellence is worth ${marketing_roi:,.0f} in equivalent marketing value. Use in promotional campaigns targeting international markets.")
            
        
        # Sector-specific government insights based on word categories
        word_lower = word.lower()
        if word_lower in ['wifi', 'internet', 'connection', 'technology']:
            digital_investment = max(50000, mentions * 500)  # Investment needed per mention
            insights.append(f"🌐 **Digital Tourism Infrastructure Gap**: Requires ${digital_investment:,.0f} investment in tourism-grade internet infrastructure. Coordinate with Ministry of Communications for 5G tourism zones and smart destination initiatives.")
            
        elif word_lower in ['clean', 'cleanliness', 'hygiene', 'sanitation']:
            health_budget = max(75000, mentions * 750)  # Health improvement budget
            insights.append(f"🏥 **Public Health Tourism Priority**: Allocate ${health_budget:,.0f} for enhanced sanitation protocols. Partner with Ministry of Health for tourism facility health certification programs.")
            
        elif word_lower in ['transport', 'transportation', 'location', 'accessibility']:
            transport_investment = max(100000, mentions * 1000)  # Transport infrastructure
            insights.append(f"🚌 **Tourism Mobility Infrastructure**: Requires ${transport_investment:,.0f} investment in tourism transportation. Coordinate with Ministry of Transport for dedicated tourism routes and accessibility improvements.")
            
        elif word_lower in ['staff', 'service', 'reception', 'hospitality']:
            training_budget = max(40000, mentions * 400)  # Training program budget
            insights.append(f"👥 **Human Capital Development**: Invest ${training_budget:,.0f} in hospitality training programs. Develop national tourism service standards and certification programs with Ministry of Education.")
            
        elif word_lower in ['food', 'cuisine', 'restaurant', 'dining']:
            culinary_investment = max(60000, mentions * 600)  # Culinary tourism budget
            insights.append(f"🍽️ **Culinary Tourism Development**: Allocate ${culinary_investment:,.0f} for culinary excellence programs. Create Egyptian cuisine certification and chef training initiatives for tourism establishments.")
        
        # Strategic recommendations based on data analysis
        if review_impact_percentage > 5:
            insights.append(f"🎯 **National Tourism Strategy Impact**: With {review_impact_percentage:.1f}% market penetration, '{word}' should be integrated into Egypt's 2030 Tourism Vision and allocated dedicated budget in national tourism development plans.")
        
    except Exception as e:
        # Simplified fallback insights
        if word_type == 'positive':
            insights.append(f"✅ **Tourism Excellence Indicator**: '{word}' represents successful service delivery affecting {reviews:,} tourist experiences. Replicate these practices for broader tourism improvement.")
        else:
            insights.append(f"⚠️ **Service Gap Priority**: '{word}' indicates service deficiency affecting {reviews:,} tourist experiences. Immediate government intervention required to protect tourism reputation.")
    
    return insights

def generate_government_recommendations(word, word_type, mentions, reviews, cities, associated_words):
    """Generate data-driven government recommendations based on actual metrics"""
    recommendations = []
    
    try:
        df = load_data()
        total_reviews = len(df)
        review_impact = (reviews / total_reviews * 100) if total_reviews > 0 else 0
        
        # Calculate reliable data-driven metrics
        frequency_score = min(mentions / 100, 1.0)  # Normalize frequency to 0-1 scale
        geographic_spread = len(cities) if cities else 1
        impact_level = "HIGH" if review_impact > 5 else "MEDIUM" if review_impact > 2 else "LOW"
        
        if word_type == 'positive':
            # Excellence replication recommendations based on data
            recommendations.extend([
                {
                    'category': '🚀 Excellence Standards Replication',
                    'priority': 'STRATEGIC' if frequency_score > 0.5 else 'HIGH',
                    'action': f"Replicate '{word}' success standards across {geographic_spread} cities. Current {mentions:,} positive mentions indicate proven service delivery model.",
                    'departments': ['Ministry of Tourism', 'Tourism Development Authority', 'Governorate Tourism Offices'],
                    'timeline': '60-90 days implementation',
                    'data_basis': f"{review_impact:.1f}% of reviews, {mentions:,} mentions across {geographic_spread} locations",
                    'success_metrics': [
                        f"Standardize '{word}' procedures in all major tourism facilities",
                        f"Achieve consistent service delivery across {geographic_spread} cities",
                        "Establish quality certification program based on proven practices"
                    ]
                },
                {
                    'category': '🌍 Marketing Excellence Showcase',
                    'priority': 'HIGH' if impact_level == "HIGH" else 'MEDIUM',
                    'action': f"Feature '{word}' excellence in marketing campaigns. {review_impact:.1f}% tourist satisfaction provides credible marketing foundation.",
                    'departments': ['Egyptian Tourism Authority', 'Ministry of Foreign Affairs', 'Tourism Marketing Board'],
                    'timeline': '30-45 days campaign launch',
                    'data_basis': f"Based on {reviews:,} positive reviews and {frequency_score*100:.0f}% frequency score",
                    'success_metrics': [
                        "Increase brand recognition for Egyptian tourism excellence",
                        "Improve destination competitiveness ranking",
                        "Generate measurable increase in booking inquiries"
                    ]
                }
            ])
            
            if len(cities) > 2:
                coordination_value = len(cities) * 50000  # Value of coordination
                recommendations.append({
                    'category': '🏛️ Inter-Governorate Excellence Network',
                    'priority': 'MEDIUM',
                    'action': f"Create formal excellence-sharing network across {len(cities)} cities. Standardize '{word}' practices and establish monthly performance benchmarking with ${coordination_value:,.0f} coordination investment.",
                    'departments': ['Regional Development Ministry', 'Local Government Units', 'Tourism Quality Assurance'],
                    'timeline': '90-120 days network establishment',
                    'budget_range': f'${coordination_value:,.0f} - ${coordination_value * 1.5:,.0f}',
                    'expected_outcome': 'Improved service efficiency and tourist satisfaction',
                    'success_metrics': [
                        f"Achieve 85% consistency across {len(cities)} cities",
                        "Reduce service delivery variance by 60%",
                        "Establish monthly performance monitoring system"
                    ]
                })
            
            # Innovation and technology integration for positive aspects
            if mentions > 40:
                tech_investment = min(mentions * 2000, 300000)
                recommendations.append({
                    'category': '💡 Digital Innovation Amplification',
                    'priority': 'MEDIUM',
                    'action': f"Develop AI-powered '{word}' optimization system and mobile apps to enhance tourist experience. ${tech_investment:,.0f} investment in smart tourism technology.",
                    'departments': ['Ministry of Communications', 'Digital Transformation Unit', 'Tourism Innovation Lab'],
                    'timeline': '120-180 days development',
                    'budget_range': f'${tech_investment:,.0f} - ${tech_investment * 1.3:,.0f}',
                    'expected_outcome': 'Enhanced digital tourism experience and service quality',
                    'success_metrics': [
                        "Deploy smart tourism solutions in 15+ facilities",
                        "Increase digital tourist engagement by 70%",
                        "Create replicable digital excellence model"
                    ]                })        
        else:  # negative word
            # Problem severity assessment based on data
            severity_level = "CRITICAL" if mentions > 50 else "HIGH" if mentions > 20 else "MEDIUM"
            urgency_timeline = "72 hours" if mentions > 50 else "7-14 days" if mentions > 20 else "30 days"
            
            recommendations.extend([
                {
                    'category': '🚨 Service Quality Response',
                    'priority': severity_level,
                    'action': f"Address '{word}' service issues affecting {reviews:,} reviews ({review_impact:.1f}% of tourist feedback). Deploy targeted improvement program across {geographic_spread} affected locations.",
                    'departments': ['Tourism Crisis Management', 'Service Quality Control', 'Operations Management'],
                    'timeline': f'{urgency_timeline} activation, 30-60 days resolution',
                    'data_basis': f"{mentions:,} mentions, {geographic_spread} cities affected, {frequency_score*100:.0f}% frequency score",
                    'success_metrics': [
                        f"Reduce '{word}' complaints by 70% within 60 days",
                        f"Implement quality standards across {geographic_spread} locations",
                        "Establish preventive monitoring system"
                    ]
                },
                {
                    'category': '🔧 Root Cause Analysis & Fix',
                    'priority': 'HIGH',
                    'action': f"Conduct comprehensive analysis of '{word}' issues and implement systematic solutions. Focus on {geographic_spread} affected destinations with data-driven improvement plan.",
                    'departments': ['Quality Assurance', 'Operations Research', 'Tourism Standards Authority'],
                    'timeline': '14-30 days analysis, 60-90 days implementation',
                    'data_basis': f"Analysis of {reviews:,} reviews across {geographic_spread} locations",
                    'success_metrics': [
                        "Identify and eliminate root causes of service failures",
                        "Achieve compliance with national tourism standards",
                        "Prevent issue recurrence through systematic improvements"
                    ]                }
            ])        
        # Multi-location coordination for widespread issues
        if geographic_spread > 2:
            recommendations.append({
                'category': '🌐 Multi-Location Coordination',
                'priority': 'HIGH',
                'action': f"Coordinate improvement efforts across {geographic_spread} affected cities to ensure consistent service quality standards.",
                'departments': ['Inter-Governmental Coordination', 'Regional Tourism Directors', 'Quality Standards Authority'],
                'timeline': '30-60 days synchronized implementation',
                'data_basis': f"Issue affects {geographic_spread} cities with {mentions:,} total mentions",
                'success_metrics': [
                    f"Achieve uniform service standards across {geographic_spread} cities",
                    "Establish consistent quality monitoring procedures",
                    "Create replicable improvement framework"
                ]
            })
        
        # Associated issues analysis for comprehensive solutions        if associated_words and len(associated_words) > 2:
            related_terms = ', '.join([w['word'] for w in associated_words[:3]])
            
            recommendations.append({
                'category': '🔗 Comprehensive Service Strategy',
                'priority': 'STRATEGIC',
                'action': f"Address '{word}' alongside related issues: {related_terms}. Implement integrated improvement approach for holistic service enhancement.",
                'departments': ['Tourism Experience Design', 'Service Integration Unit', 'Customer Journey Analytics'],
                'timeline': '90-180 days comprehensive implementation',
                'data_basis': f"Analysis includes {len(associated_words)} related service areas",
                'success_metrics': [
                    "Improve overall tourist satisfaction ratings",
                    f"Create integrated solution addressing {len(associated_words)} related issues",
                    "Establish comprehensive service quality framework"
                ]
            })
    
    except Exception as e:
        # Fallback to simplified recommendations
        recommendations = [
            {
                'category': 'Investigation & Analysis',
                'priority': 'HIGH',
                'action': f"Conduct immediate analysis of '{word}' patterns affecting {reviews:,} tourists",
                'timeline': '14-21 days',
                'departments': ['Tourism Analysis Unit']
            },
            {
                'category': 'Service Enhancement',
                'priority': 'MEDIUM',
                'action': f"Implement targeted improvements for '{word}'-related issues",
                'timeline': '30-60 days',                'departments': ['Service Quality Department']
            }
        ]
    
    return recommendations

def create_tourism_economics_chart(df, city_filter="all", start_date=None, end_date=None):
    """Create Tourism Analytics chart with data-driven insights (no revenue estimates)"""
    print(f"DEBUG Analytics Chart: Input df shape: {df.shape if not df.empty else 'Empty'}")
    
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="⚠️ No data available for Tourism Analytics",
            xref="paper", yref="paper", x=0.5, y=0.5,
            xanchor='center', yanchor='middle', showarrow=False,
            font=dict(size=16, color="#ef4444", family="Inter, sans-serif")
        )
        fig.update_layout(
            title="Tourism Analytics - Visitor Engagement Analysis",
            height=450,
            template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig

    try:
        # Apply filters based on actual data structure
        filtered_df = df.copy()
        
        # Apply city filter using the actual 'City' column
        if city_filter != "all" and 'City' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['City'] == city_filter]
        
        # Apply date filter using the actual 'Review Date' column
        if (start_date or end_date) and 'Review Date' in filtered_df.columns:
            filtered_df['Review Date'] = pd.to_datetime(filtered_df['Review Date'], errors='coerce')
            
            if start_date:
                start_date = pd.to_datetime(start_date)
                filtered_df = filtered_df[filtered_df['Review Date'] >= start_date]
            
            if end_date:
                end_date = pd.to_datetime(end_date)
                filtered_df = filtered_df[filtered_df['Review Date'] <= end_date]
        
        if filtered_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="⚠️ No Egyptian tourism data available for selected filters",
                xref="paper", yref="paper", x=0.5, y=0.5,
                xanchor='center', yanchor='middle', showarrow=False,
                font=dict(size=16, color="#ef4444", family="Inter, sans-serif")
            )
            fig.update_layout(
                title="Egyptian Tourism Economic Impact Analysis",
                height=450,
                template="plotly_white",
                paper_bgcolor='rgba(0,0,0,0)'
            )
            return fig        # Find nationality and sentiment columns dynamically (like nationality chart does)
        nationality_column = None
        for col in filtered_df.columns:
            if 'nationality' in col.lower():
                nationality_column = col
                break
        
        sentiment_column = None
        for col in filtered_df.columns:
            if 'sentiment' in col.lower():
                sentiment_column = col
                break
        
        print(f"DEBUG: Found nationality column: '{nationality_column}'")
        print(f"DEBUG: Found sentiment column: '{sentiment_column}'")
        print(f"DEBUG: Available columns: {list(filtered_df.columns)}")
        
        if not nationality_column:
            fig = go.Figure()
            fig.add_annotation(
                text="⚠️ No nationality data found in database",
                xref="paper", yref="paper", x=0.5, y=0.5,
                xanchor='center', yanchor='middle', showarrow=False,
                font=dict(size=16, color="#ef4444", family="Inter, sans-serif")
            )
            fig.update_layout(
                title="Egyptian Tourism Economic Impact Analysis",
                height=450,
                template="plotly_white",
                paper_bgcolor='rgba(0,0,0,0)'
            )
            return fig
        
        if not sentiment_column:
            fig = go.Figure()
            fig.add_annotation(
                text="⚠️ No sentiment data found in database",
                xref="paper", yref="paper", x=0.5, y=0.5,
                xanchor='center', yanchor='middle', showarrow=False,
                font=dict(size=16, color="#ef4444", family="Inter, sans-serif")
            )
            fig.update_layout(
                title="Egyptian Tourism Economic Impact Analysis",
                height=450,
                template="plotly_white",
                paper_bgcolor='rgba(0,0,0,0)'
            )
            return fig        # Calculate data-driven metrics instead of economic estimates
        nationality_stats = []
        
        print(f"DEBUG: Processing {len(filtered_df[nationality_column].unique())} unique nationalities")
        
        for nationality in filtered_df[nationality_column].dropna().unique():
            # Ensure nationality is a string and handle various data types
            nationality_str = str(nationality).strip()
            if not nationality_str or nationality_str.lower() in ['nan', 'none', '', 'null']:
                continue
                
            # Filter data for this nationality - ensure proper comparison
            nat_data = filtered_df[filtered_df[nationality_column].astype(str) == nationality_str]
            
            if len(nat_data) == 0:
                continue
              # Calculate actual metrics from available data with robust type handling
            total_visitors = int(len(nat_data))
            
            # Ensure sentiment column is numeric and handle any string values
            sentiment_data = nat_data[sentiment_column].copy()
            
            # Convert sentiment to numeric with comprehensive error handling
            try:
                # First attempt: direct numeric conversion
                sentiment_numeric = pd.to_numeric(sentiment_data, errors='coerce')
                # Count non-null values after conversion
                valid_sentiment = sentiment_numeric.dropna()
                positive_reviews = int(len(valid_sentiment[valid_sentiment == 1]))
                negative_reviews = int(len(valid_sentiment[valid_sentiment == -1]))
                neutral_reviews = int(total_visitors - positive_reviews - negative_reviews)
            except Exception as e:
                print(f"DEBUG: Numeric conversion failed for {nationality_str}: {e}")
                try:
                    # Fallback: string-based analysis
                    sentiment_str = sentiment_data.astype(str)
                    positive_reviews = int(len(sentiment_str[sentiment_str.str.contains('1', na=False)]))
                    negative_reviews = int(len(sentiment_str[sentiment_str.str.contains('-1', na=False)]))
                    neutral_reviews = int(total_visitors - positive_reviews - negative_reviews)
                except Exception as e2:
                    print(f"DEBUG: String fallback failed for {nationality_str}: {e2}")
                    # Final fallback: assume equal distribution
                    positive_reviews = int(total_visitors // 3)
                    negative_reviews = int(total_visitors // 3)
                    neutral_reviews = int(total_visitors - positive_reviews - negative_reviews)
            
            # Calculate satisfaction rate (0-100%) with safe division
            satisfaction_rate = float((positive_reviews / total_visitors * 100) if total_visitors > 0 else 0)
            
            # Calculate market share based on actual visitor numbers
            market_share = float((total_visitors / len(filtered_df)) * 100) if len(filtered_df) > 0 else 0
            
            # Calculate visitor engagement score (visitors weighted by satisfaction)
            engagement_score = float(total_visitors * (1 + satisfaction_rate / 100))
              # Determine visitor classification based on actual data
            if total_visitors >= 50 and satisfaction_rate >= 60:
                visitor_category = "High Value Segment"
                category_color = "#059669"
            elif total_visitors >= 20 and satisfaction_rate >= 40:
                visitor_category = "Strategic Segment"
                category_color = "#0891b2"
            elif total_visitors >= 10:
                visitor_category = "Growth Potential"
                category_color = "#f59e0b"
            else:
                visitor_category = "Emerging Market"
                category_color = "#ef4444"
            
            nationality_stats.append({
                'nationality': nationality_str,  # Use cleaned string
                'total_visitors': total_visitors,
                'satisfaction_rate': satisfaction_rate,
                'market_share': market_share,
                'engagement_score': engagement_score,
                'visitor_category': visitor_category,
                'category_color': category_color,
                'positive_reviews': positive_reviews,
                'negative_reviews': negative_reviews,
                'neutral_reviews': neutral_reviews
            })
        
        if not nationality_stats:
            fig = go.Figure()
            fig.add_annotation(
                text="⚠️ No nationality statistics could be calculated",
                xref="paper", yref="paper", x=0.5, y=0.5,
                xanchor='center', yanchor='middle', showarrow=False,
                font=dict(size=16, color="#ef4444", family="Inter, sans-serif")
            )
            fig.update_layout(                title="Tourism Analytics - Visitor Engagement Analysis",
                height=450,
                template="plotly_white",
                paper_bgcolor='rgba(0,0,0,0)'
            )
            return fig
        
        # Sort by engagement score and take top 12 for better readability
        nationality_stats = sorted(nationality_stats, key=lambda x: x['engagement_score'], reverse=True)[:12]
        
        print(f"DEBUG: Created stats for {len(nationality_stats)} nationalities")
          # Prepare chart data with explicit type conversion
        nationalities = [str(stat['nationality']) for stat in nationality_stats]
        engagement_scores = [float(stat['engagement_score']) for stat in nationality_stats]
        satisfaction_rates = [float(stat['satisfaction_rate']) for stat in nationality_stats]
        visitor_counts = [int(stat['total_visitors']) for stat in nationality_stats]
        market_shares = [float(stat['market_share']) for stat in nationality_stats]
        visitor_categories = [str(stat['visitor_category']) for stat in nationality_stats]
        
        # Debug: Print data types to identify issues
        print(f"DEBUG Chart Data Types:")
        print(f"  nationalities: {type(nationalities[0]) if nationalities else 'empty'}")
        print(f"  engagement_scores: {type(engagement_scores[0]) if engagement_scores else 'empty'}")
        print(f"  satisfaction_rates: {type(satisfaction_rates[0]) if satisfaction_rates else 'empty'}")
        print(f"  visitor_counts: {type(visitor_counts[0]) if visitor_counts else 'empty'}")
        print(f"  market_shares: {type(market_shares[0]) if market_shares else 'empty'}")
        print(f"  visitor_categories: {type(visitor_categories[0]) if visitor_categories else 'empty'}")
        
        # Validate all lists have the same length
        data_lengths = [len(nationalities), len(engagement_scores), len(satisfaction_rates), 
                       len(visitor_counts), len(market_shares), len(visitor_categories)]
        if len(set(data_lengths)) > 1:
            print(f"ERROR: Data length mismatch: {data_lengths}")
            raise ValueError(f"Data arrays have mismatched lengths: {data_lengths}")
        
        # Create modern gradient colors for each category
        modern_colors = {
            "High Value Segment": "#10b981",      # Emerald green
            "Strategic Segment": "#3b82f6",       # Blue  
            "Growth Potential": "#f59e0b",        # Amber
            "Emerging Market": "#ef4444"          # Red
        }
        
        # Apply modern colors
        enhanced_colors = [modern_colors.get(cat, "#6b7280") for cat in visitor_categories]
        
        # Create the chart with modern styling
        fig = go.Figure()
          # Add engagement score bars with modern design and safe type conversion
        fig.add_trace(go.Bar(
            y=nationalities,
            x=engagement_scores,
            orientation='h',
            marker=dict(
                color=enhanced_colors,
                line=dict(color='rgba(255, 255, 255, 0.8)', width=2),
                opacity=0.9,
                # Add subtle shadow effect
                pattern=dict(shape="", solidity=0.9)
            ),
            text=[f"{float(score):.0f}" for score in engagement_scores],  # Ensure float conversion
            textposition='inside',            textfont=dict(
                color='white',
                size=11,
                family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
                weight=600  # Changed from '600' string to 600 integer
            ),
            hovertemplate='<b style="color:#1e293b; font-size:14px;">%{y}</b><br>' +
                         '<span style="color:#64748b;">Engagement Score:</span> <b>%{x:.1f}</b><br>' +
                         '<span style="color:#64748b;">Total Visitors:</span> <b>%{customdata[0]:,d}</b><br>' +
                         '<span style="color:#64748b;">Satisfaction Rate:</span> <b>%{customdata[1]:.1f}%</b><br>' +
                         '<span style="color:#64748b;">Market Share:</span> <b>%{customdata[2]:.1f}%</b><br>' +
                         '<span style="color:#64748b;">Category:</span> <b>%{customdata[3]}</b>' +
                         '<extra></extra>',
            customdata=list(zip(
                [int(v) for v in visitor_counts],      # Ensure integers
                [float(v) for v in satisfaction_rates], # Ensure floats
                [float(v) for v in market_shares],      # Ensure floats
                [str(v) for v in visitor_categories]    # Ensure strings
            )),
            name='Visitor Engagement',
            width=0.65  # Optimized bar width for modern look
        ))# Update layout with modern design and better annotation positioning
        fig.update_layout(            title={
                'text': "Tourism Analytics - Visitor Engagement by Nationality<br><sub style='font-size:11px; color:#64748b; font-weight:400;'>📊 Based on actual visitor counts and satisfaction rates from database</sub>",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'family': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif', 'color': '#1e293b', 'weight': 600}  # Changed to integer
            },            xaxis_title={
                'text': "Engagement Score (Visitors × Satisfaction Factor)",
                'font': {'size': 13, 'family': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif', 'color': '#475569', 'weight': 500}  # Changed to integer
            },
            yaxis_title={
                'text': "Visitor Nationality",
                'font': {'size': 13, 'family': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif', 'color': '#475569', 'weight': 500}  # Changed to integer
            },
            height=500,  # Increased height to match layout
            template="plotly_white",
            showlegend=False,
            margin=dict(l=160, r=60, t=90, b=80),  # Reduced margins for compact size
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(148, 163, 184, 0.1)',
                gridwidth=1,
                tickfont=dict(size=11, family='Inter, -apple-system, BlinkMacSystemFont, sans-serif', color='#64748b'),
                title_standoff=25,
                showline=True,
                linecolor='rgba(148, 163, 184, 0.2)',
                linewidth=1
            ),            yaxis=dict(
                showgrid=False,  # Clean modern look
                tickfont=dict(size=11, family='Inter, -apple-system, BlinkMacSystemFont, sans-serif', color='#475569', weight=500),  # Changed to integer
                automargin=True,
                title_standoff=25,
                showline=False
            ),
            plot_bgcolor='rgba(248, 250, 252, 0.3)',  # Subtle modern background            paper_bgcolor='rgba(255, 255, 255, 0)',
            # Move annotation to a floating card position instead of bottom
            annotations=[]
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating tourism analytics chart: {e}")
        import traceback
        traceback.print_exc()
        
        fig = go.Figure()
        fig.add_annotation(
            text=f"⚠️ Error analyzing tourism data: {str(e)[:50]}...",
            xref="paper", yref="paper", x=0.5, y=0.5,
            xanchor='center', yanchor='middle', showarrow=False,
            font=dict(size=16, color="#ef4444", family="Inter, sans-serif")
        )
        fig.update_layout(
            title="Tourism Analytics - Visitor Engagement Analysis",
            height=450,
            template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig

def create_seasonal_analytics_chart(df, city_filter="all", start_date=None, end_date=None):
    """Create Seasonal & Temporal Analytics chart with dynamic data"""
    print(f"DEBUG Seasonal Chart: Input df shape: {df.shape if not df.empty else 'Empty'}")
    print(f"DEBUG Seasonal Chart: Input columns: {list(df.columns) if not df.empty else 'None'}")
    
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="⚠️ No data available for Seasonal Analysis",
            xref="paper", yref="paper", x=0.5, y=0.5,
            xanchor='center', yanchor='middle', showarrow=False,
            font=dict(size=16, color="#ef4444", family="Inter, sans-serif")
        )
        fig.update_layout(
            title="📅 Seasonal Tourism Patterns & Satisfaction Heatmap",
            height=450,
            template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig    
    try:
        # Apply filters
        filtered_df = df.copy()
        
        if city_filter != "all":
            city_column = None
            possible_city_cols = ['City', 'city', 'destination', 'location']
            for col in filtered_df.columns:
                if col in possible_city_cols:
                    city_column = col
                    break
            if city_column:
                filtered_df = filtered_df[filtered_df[city_column] == city_filter]
        
        # Use flexible column detection for date
        date_column = None
        possible_date_cols = ['Review Date', 'review_date', 'date']
        for col in filtered_df.columns:
            if col in possible_date_cols or 'date' in col.lower():
                date_column = col
                break
        
        print(f"DEBUG Seasonal: Found date column: {date_column}")
        
        if not date_column:
            fig = go.Figure()
            fig.add_annotation(
                text="⚠️ No date column found for temporal analysis",
                xref="paper", yref="paper", x=0.5, y=0.5,
                xanchor='center', yanchor='middle', showarrow=False,
                font=dict(size=16, color="#ef4444", family="Inter, sans-serif")
            )
            fig.update_layout(
                title="📅 Seasonal Tourism Patterns & Satisfaction Heatmap",
                height=450,
                template="plotly_white",
                paper_bgcolor='rgba(0,0,0,0)'
            )
            return fig
        
        # Use flexible sentiment column detection
        sentiment_column = None
        possible_sentiment_cols = ['sentiment classification', 'sentiment_classification', 'sentiment']
        for col in filtered_df.columns:
            if col in possible_sentiment_cols or 'sentiment' in col.lower():
                sentiment_column = col
                break
        
        print(f"DEBUG Seasonal: Found sentiment column: {sentiment_column}")
        
        if not sentiment_column:
            fig = go.Figure()
            fig.add_annotation(
                text="⚠️ No sentiment data found",
                xref="paper", yref="paper", x=0.5, y=0.5,
                xanchor='center', yanchor='middle', showarrow=False,
                font=dict(size=16, color="#ef4444", family="Inter, sans-serif")
            )
            fig.update_layout(
                title="📅 Seasonal Tourism Patterns & Satisfaction Heatmap",
                height=450,
                template="plotly_white",
                paper_bgcolor='rgba(0,0,0,0)'
            )
            return fig
        
        # Convert date column to datetime
        filtered_df[date_column] = pd.to_datetime(filtered_df[date_column], errors='coerce')
        
        # Apply date filter if provided
        if start_date:
            start_date = pd.to_datetime(start_date)
            filtered_df = filtered_df[filtered_df[date_column] >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            filtered_df = filtered_df[filtered_df[date_column] <= end_date]
          # Remove rows with invalid dates
        filtered_df = filtered_df.dropna(subset=[date_column])
        
        if filtered_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="⚠️ No valid date data available",
                xref="paper", yref="paper", x=0.5, y=0.5,
                xanchor='center', yanchor='middle', showarrow=False,
                font=dict(size=16, color="#ef4444", family="Inter, sans-serif")
            )
            fig.update_layout(
                title="📅 Seasonal Tourism Patterns & Satisfaction Heatmap",
                height=450,
                template="plotly_white",
                paper_bgcolor='rgba(0,0,0,0)'
            )
            return fig
        
        # Extract temporal features
        filtered_df['month'] = filtered_df[date_column].dt.month
        filtered_df['month_name'] = filtered_df[date_column].dt.month_name()
        filtered_df['day_of_week'] = filtered_df[date_column].dt.day_name()
        
        # Calculate monthly statistics
        monthly_stats = []
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for month_num, month_name in enumerate(month_names, 1):
            month_data = filtered_df[filtered_df['month'] == month_num]
            if len(month_data) > 0:
                total_visits = len(month_data)
                positive_visits = len(month_data[month_data[sentiment_column] == 1])
                satisfaction_rate = (positive_visits / total_visits * 100) if total_visits > 0 else 0
                
                monthly_stats.append({
                    'month': month_name,
                    'month_num': month_num,
                    'total_visits': total_visits,
                    'satisfaction_rate': satisfaction_rate
                })
            else:
                monthly_stats.append({
                    'month': month_name,
                    'month_num': month_num,
                    'total_visits': 0,
                    'satisfaction_rate': 0                })
        
        if not monthly_stats or all(stat['total_visits'] == 0 for stat in monthly_stats):
            fig = go.Figure()
            fig.add_annotation(
                text="⚠️ No temporal patterns found in data",
                xref="paper", yref="paper", x=0.5, y=0.5,
                xanchor='center', yanchor='middle', showarrow=False,
                font=dict(size=16, color="#ef4444", family="Inter, sans-serif")
            )
            fig.update_layout(
                title="📅 Seasonal Tourism Patterns & Satisfaction Heatmap",
                height=450,
                template="plotly_white",
                paper_bgcolor='rgba(0,0,0,0)'
            )
            return fig
        
        # Create monthly visits bar chart
        months = [stat['month'] for stat in monthly_stats]
        visits = [stat['total_visits'] for stat in monthly_stats]
        satisfaction = [stat['satisfaction_rate'] for stat in monthly_stats]
        
        fig = go.Figure()
        
        # Add bar chart for visitor volume
        fig.add_trace(go.Bar(
            x=months,
            y=visits,
            name='Monthly Visitors',
            marker=dict(
                color=satisfaction,
                colorscale='RdYlGn',
                cmin=0,
                cmax=100,                colorbar=dict(
                    title="Satisfaction Rate (%)",
                    tickmode="linear",
                    tick0=0,
                    dtick=20,
                    len=0.7
                ),
                line=dict(color='rgba(255,255,255,0.8)', width=2)
            ),
            hovertemplate='<b>%{x}</b><br>' +
                         '👥 Visitors: %{y:,.0f}<br>' +
                         '😊 Satisfaction: %{customdata:.1f}%<br>' +
                         '<extra></extra>',
            customdata=satisfaction
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="📅 Seasonal Tourism Patterns & Satisfaction",
                font=dict(size=18, color="#1e293b", family="Inter, -apple-system, sans-serif", weight="bold"),
                x=0.02,
                y=0.95
            ),
            xaxis=dict(
                title="Month",
                showgrid=False,
                title_font=dict(size=12, color="#64748b"),
                tickfont=dict(size=10, color="#64748b"),
                tickangle=45
            ),
            yaxis=dict(
                title="Number of Visitors",
                showgrid=True,
                gridcolor='rgba(226, 232, 240, 0.6)',
                title_font=dict(size=12, color="#64748b"),
                tickfont=dict(size=10, color="#64748b")
            ),
            height=450,
            template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(248, 250, 252, 0.8)',
            margin=dict(l=60, r=120, t=60, b=80),
            showlegend=False,            font=dict(family="Inter, -apple-system, sans-serif")
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating seasonal chart: {e}")
        import traceback
        traceback.print_exc()
        
        fig = go.Figure()
        fig.add_annotation(
            text=f"⚠️ Error: {str(e)[:50]}...",
            xref="paper", yref="paper", x=0.5, y=0.5,
            xanchor='center', yanchor='middle', showarrow=False,
            font=dict(size=16, color="#ef4444", family="Inter, sans-serif")
        )
        fig.update_layout(
            title="📅 Seasonal Tourism Patterns & Satisfaction",
            height=450,
            template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig
        filtered_df['month'] = filtered_df[date_column].dt.month
        filtered_df['month_name'] = filtered_df[date_column].dt.month_name()
        filtered_df['day_of_week'] = filtered_df[date_column].dt.day_name()
        
        # Create seasonal heatmap data
        seasonal_data = filtered_df.groupby(['month', 'day_of_week']).agg({
            'sentiment classification': ['count', lambda x: (x == 1).sum()]
        }).round(2)
        
        seasonal_data.columns = ['total_visits', 'positive_visits']
        seasonal_data['satisfaction_rate'] = (seasonal_data['positive_visits'] / seasonal_data['total_visits'] * 100).fillna(0).round(1)
        seasonal_data = seasonal_data.reset_index()
        
        # Create pivot table for heatmap
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Create heatmap data
        heatmap_data = []
        visit_counts = []
        
        for month in range(1, 13):
            month_row = []
            count_row = []
            for day in day_order:
                day_data = seasonal_data[(seasonal_data['month'] == month) & (seasonal_data['day_of_week'] == day)]
                if not day_data.empty:
                    satisfaction = day_data['satisfaction_rate'].iloc[0]
                    count = day_data['total_visits'].iloc[0]
                else:
                    satisfaction = 0
                    count = 0
                month_row.append(satisfaction)
                count_row.append(count)
            heatmap_data.append(month_row)
            visit_counts.append(count_row)
        
        # Create the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=day_order,
            y=month_names,
            colorscale='RdYlGn',
            zmin=0,
            zmax=100,
            hoverongaps=False,
            hovertemplate='<b>%{y} - %{x}</b><br>' +
                         'Satisfaction Rate: %{z:.1f}%<br>' +
                         'Total Visits: %{customdata}<br>' +
                         '<extra></extra>',
            customdata=visit_counts,            colorbar=dict(
                title="Satisfaction Rate (%)",
                tickmode="linear",
                tick0=0,
                dtick=20,
                len=0.7
            )
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="📅 Seasonal Tourism Patterns & Satisfaction Heatmap",
                font=dict(size=18, color="#1e293b", family="Inter, -apple-system, sans-serif", weight="bold"),
                x=0.02,
                y=0.95
            ),
            xaxis=dict(
                title="Day of Week",
                showgrid=False,
                title_font=dict(size=12, color="#64748b"),
                tickfont=dict(size=10, color="#64748b")
            ),
            yaxis=dict(
                title="Month",
                showgrid=False,
                title_font=dict(size=12, color="#64748b"),
                tickfont=dict(size=10, color="#64748b")
            ),
            height=450,
            template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(248, 250, 252, 0.8)',            margin=dict(l=60, r=120, t=60, b=50),
            font=dict(family="Inter, -apple-system, sans-serif")
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating seasonal chart: {e}")
        fig = go.Figure()
        fig.add_annotation(
            text="⚠️ Error processing temporal data",
            xref="paper", yref="paper", x=0.5, y=0.5,
            xanchor='center', yanchor='middle', showarrow=False,
            font=dict(size=16, color="#ef4444", family="Inter, sans-serif")
        )
        fig.update_layout(
            title="Seasonal & Temporal Analytics",
            height=450,
            template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig

def create_kpi_card(title, value, subtitle, icon, color_class="primary"):
    """Create a KPI card component with enhanced styling"""
    # Enhanced icon mapping with emojis
    icon_emoji_map = {
        "fa-users": "👥",
        "fa-thumbs-up": "👍",
        "fa-thumbs-down": "👎", 
        "fa-smile": "😊",
        "fa-meh": "😐",
        "fa-frown": "😟"
    }
    
    emoji = icon_emoji_map.get(icon, "📊")
    
    return dbc.Col([
        html.Div([
            html.Div([                html.Div([
                    html.Span(emoji, className="kpi-card-emoji"),
                    html.I(className=f"fas {icon} fa-2x text-{color_class} kpi-card-fa-icon")
                ], className="kpi-card-icon-container"),
                html.H3(str(value), className="card-value"),
                html.H6(title, className="card-title"),
                html.P(subtitle, className="card-subtitle")
            ], className="text-center")
        ], className="overview-card h-100")
    ], lg=3, md=6, sm=12, className="mb-4")

def create_word_analysis_modal():
    """Create modal component for word analysis details"""
    return dbc.Modal([
        dbc.ModalHeader([
            html.H4(id="modal-word-title", className="modal-title"),
            dbc.Button([
                html.I(className="fas fa-times")
            ], id="close-modal", className="btn-close", n_clicks=0)
        ], className="enhanced-modal-header"),
          dbc.ModalBody([            # Enhanced Loading spinner
            html.Div(id="modal-loading", children=[
                html.Div([
                    dbc.Spinner(color="primary", size="lg", spinnerClassName="modal-spinner"),
                    html.Div([
                        html.I(className="fas fa-chart-line text-primary me-2 modal-title-icon"),
                        html.H5("Analyzing Word Statistics", className="text-primary mb-2"),
                        html.P("Please wait while we process the data and generate insights...", className="text-muted")
                    ], className="mt-4")
                ], className="d-flex flex-column align-items-center")
            ], className="modal-loading"),
            
            # Enhanced Content container
            html.Div(id="modal-content", className="modal-content-hidden")
        ], className="modal-body-custom")    ], id="word-analysis-modal", size="xl", is_open=False, backdrop=True, scrollable=True,
    className="enhanced-modal modal-z-index")

def create_nationality_analysis_modal():
    """Create modal component for nationality analysis details"""
    return dbc.Modal([
        dbc.ModalHeader([
            html.H4(id="modal-nationality-title", className="modal-title"),
            dbc.Button([
                html.I(className="fas fa-times")
            ], id="close-nationality-modal", className="btn-close", n_clicks=0)
        ], className="enhanced-modal-header"),        dbc.ModalBody([
            # Enhanced Loading spinner
            html.Div(id="modal-nationality-loading", children=[
                html.Div([
                    dbc.Spinner(color="primary", size="lg", spinnerClassName="modal-spinner"),
                    html.Div([
                        html.I(className="fas fa-globe text-primary me-2 modal-title-icon"),
                        html.H5("Analyzing Nationality Statistics", className="text-primary mb-2"),
                        html.P("Please wait while we process demographic data and generate insights...", className="text-muted")
                    ], className="mt-4")
                ], className="d-flex flex-column align-items-center")
            ], className="modal-loading"),
            
            # Enhanced Content container
            html.Div(id="modal-nationality-content", className="modal-content-hidden")
        ], className="modal-body-custom")    ], id="nationality-analysis-modal", size="xl", is_open=False, backdrop=True, scrollable=True,
    className="enhanced-modal modal-z-index")

def create_modal_content(word_data):
    """Create modern, government-focused modal content for word analysis"""
    word = word_data['word']
    word_type = word_data['type']
    
    # Dynamic color scheme based on word type and impact
    if word_type == 'positive':
        color_scheme = {
            'primary': '#10b981',
            'light': '#d1fae5',
            'border': '#6ee7b7',
            'icon': 'fa-thumbs-up',
            'gradient': 'linear-gradient(135deg, #10b981 0%, #059669 100%)'
        }
    else:
        color_scheme = {
            'primary': '#ef4444',
            'light': '#fee2e2',
            'border': '#fca5a5',
            'icon': 'fa-exclamation-triangle',
            'gradient': 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)'
        }
    
    return html.Div([
        # Executive Summary Header
        html.Div([
            html.Div([
                html.I(className=f"fas {color_scheme['icon']} fa-2x", 
                      style={"color": color_scheme['primary'], "marginRight": "1rem"}),
                html.Div([
                    html.H4(f"Analysis: '{word.title()}'", 
                           style={"color": color_scheme['primary'], "margin": "0", "fontWeight": "700"}),
                    html.P(f"{word_type.title()} Impact Analysis | Government Tourism Intelligence", 
                          className="text-muted mb-0", style={"fontSize": "0.9rem"})
                ])
            ], className="d-flex align-items-center mb-4")
        ]),
        
        # Key Performance Metrics (Compact 4-card layout)
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-chart-line fa-lg", 
                                  style={"color": color_scheme['primary']}),
                            html.H4(f"{word_data['total_mentions']:,}", 
                                   className="metric-value mb-1"),
                            html.P("Total Mentions", className="metric-label mb-0")
                        ], className="text-center")
                    ], className="metric-card", style={
                        "background": f"linear-gradient(145deg, white 0%, {color_scheme['light']} 100%)",
                        "border": f"1px solid {color_scheme['border']}",
                        "borderRadius": "16px",
                        "padding": "1.5rem",
                        "height": "100%"
                    })
                ], width=3),
                
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-users fa-lg", 
                                  style={"color": color_scheme['primary']}),
                            html.H4(f"{word_data['reviews_count']:,}", 
                                   className="metric-value mb-1"),
                            html.P("Tourist Reviews", className="metric-label mb-0")
                        ], className="text-center")
                    ], className="metric-card", style={
                        "background": f"linear-gradient(145deg, white 0%, {color_scheme['light']} 100%)",
                        "border": f"1px solid {color_scheme['border']}",
                        "borderRadius": "16px",
                        "padding": "1.5rem",
                        "height": "100%"
                    })
                ], width=3),
                
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-map-marked-alt fa-lg", 
                                  style={"color": color_scheme['primary']}),
                            html.H4(f"{len(word_data['cities_mentioned'])}", 
                                   className="metric-value mb-1"),
                            html.P("Cities Affected", className="metric-label mb-0")
                        ], className="text-center")
                    ], className="metric-card", style={
                        "background": f"linear-gradient(145deg, white 0%, {color_scheme['light']} 100%)",
                        "border": f"1px solid {color_scheme['border']}",
                        "borderRadius": "16px",
                        "padding": "1.5rem",
                        "height": "100%"
                    })
                ], width=3),
                
                dbc.Col([
                    html.Div([
                        html.Div([                            html.I(className="fas fa-chart-line fa-lg", 
                                  style={"color": color_scheme['primary']}),
                            html.H4(f"{min(word_data['reviews_count'] / 10, 100):.1f}%", 
                                   className="metric-value mb-1"),
                            html.P("Service Impact Score", className="metric-label mb-0")
                        ], className="text-center")
                    ], className="metric-card", style={
                        "background": f"linear-gradient(145deg, white 0%, {color_scheme['light']} 100%)",
                        "border": f"1px solid {color_scheme['border']}",
                        "borderRadius": "16px",
                        "padding": "1.5rem",
                        "height": "100%"
                    })
                ], width=3)
            ], className="mb-4")
        ]),
        
        # Sentiment Breakdown (Only if both positive and negative mentions exist)
        html.Div([
            html.H6([
                html.I(className="fas fa-chart-pie me-2", style={"color": color_scheme['primary']}),
                "Sentiment Distribution"
            ], className="section-title mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Span("👍 Positive Context", className="d-block fw-bold text-success"),
                        html.H5(f"{word_data.get('positive_mentions', 0):,}", className="text-success mb-0")
                    ], className="text-center p-3 bg-light rounded")
                ], width=6),
                dbc.Col([
                    html.Div([
                        html.Span("👎 Negative Context", className="d-block fw-bold text-danger"),
                        html.H5(f"{word_data.get('negative_mentions', 0):,}", className="text-danger mb-0")
                    ], className="text-center p-3 bg-light rounded")
                ], width=6)
            ], className="mb-4")
        ]) if word_data.get('positive_mentions', 0) > 0 and word_data.get('negative_mentions', 0) > 0 else html.Div(),
        
        # Geographic Impact
        html.Div([
            html.H6([
                html.I(className="fas fa-globe me-2", style={"color": color_scheme['primary']}),
                "Geographic Distribution"
            ], className="section-title mb-3"),
            html.Div([
                dbc.Badge(city, color="primary", className="me-1 mb-1", 
                         style={"background": color_scheme['primary']}) 
                for city in word_data['cities_mentioned'][:8]  # Limit to top 8 cities
            ] + ([dbc.Badge(f"+{len(word_data['cities_mentioned'])-8} more", 
                           color="secondary", className="me-1 mb-1")] 
                 if len(word_data['cities_mentioned']) > 8 else []))
        ], className="mb-4"),
        
        # Government Intelligence Insights
        html.Div([
            html.H6([
                html.I(className="fas fa-lightbulb me-2", style={"color": color_scheme['primary']}),
                "Strategic Intelligence"
            ], className="section-title mb-3"),
            html.Div([
                html.Div([
                    dcc.Markdown(insight, className="insight-item mb-2")
                    for insight in word_data.get('insights', [])[:4]  # Limit to top 4 insights
                ]) if word_data.get('insights') else html.P("Generating insights...", className="text-muted")
            ], className="insights-content")
        ], className="mb-4"),
        
        # Actionable Recommendations
        html.Div([
            html.H6([
                html.I(className="fas fa-tasks me-2", style={"color": color_scheme['primary']}),
                "Priority Actions"
            ], className="section-title mb-3"),
            html.Div([
                create_modern_recommendation_card(rec, color_scheme)
                for rec in word_data.get('recommendations', [])[:3]  # Limit to top 3 recommendations
            ]) if word_data.get('recommendations') else html.P("Generating recommendations...", className="text-muted")
        ])
    ], className="modal-analysis-content")

def create_modern_recommendation_card(recommendation, color_scheme):
    """Create a modern, compact recommendation card"""
    priority_colors = {
        'CRITICAL': '#dc2626',
        'EMERGENCY': '#dc2626', 
        'NATIONAL PRIORITY': '#7c3aed',
        'STRATEGIC': '#2563eb',
        'HIGH': '#ea580c',
        'URGENT': '#dc2626',
        'MEDIUM': '#059669',
        'LOW': '#6b7280'
    }
    
    priority = recommendation.get('priority', 'MEDIUM')
    priority_color = priority_colors.get(priority, '#6b7280')
    
    return html.Div([
        html.Div([
            html.Div([
                html.Span(recommendation.get('category', 'Action'), 
                         className="fw-bold d-block mb-1",
                         style={"color": color_scheme['primary'], "fontSize": "0.9rem"}),
                dbc.Badge(priority, color="primary", className="mb-2",
                         style={"background": priority_color, "fontSize": "0.7rem"})
            ]),
            html.P(recommendation.get('action', ''), 
                  className="recommendation-text mb-2",
                  style={"fontSize": "0.85rem", "lineHeight": "1.4"}),
            
            # Compact metrics row
            html.Div([
                html.Small([
                    html.I(className="fas fa-clock me-1"),
                    recommendation.get('timeline', 'TBD')
                ], className="text-muted me-3"),
                html.Small([
                    html.I(className="fas fa-chart-line me-1"),
                    recommendation.get('expected_roi', 'TBD ROI')
                ], className="text-muted")
            ], className="d-flex")
        ])
    ], className="recommendation-card mb-3", style={
        "background": "linear-gradient(145deg, #ffffff 0%, #f8fafc 100%)",
        "border": f"1px solid {color_scheme['border']}",
        "borderRadius": "12px",
        "padding": "1rem",
        "borderLeft": f"4px solid {priority_color}"
    })

def create_nationality_modal_content(nationality_data):
    """Create modern, government-focused modal content for nationality analysis"""
    nationality = nationality_data['nationality']
    
    # Color scheme
    color_scheme = {
        'primary': '#3b82f6',
        'light': '#dbeafe',
        'border': '#93c5fd',
        'icon': 'fa-globe',
        'gradient': 'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)'
    }
    
    return html.Div([
        # Executive Summary Header
        html.Div([
            html.Div([
                html.I(className=f"fas {color_scheme['icon']} fa-2x", 
                      style={"color": color_scheme['primary'], "marginRight": "1rem"}),
                html.Div([
                    html.H4(f"{nationality} Market Analysis", 
                           style={"color": color_scheme['primary'], "margin": "0", "fontWeight": "700"}),
                    html.P("Tourism Market Intelligence | Strategic Development Insights", 
                          className="text-muted mb-0", style={"fontSize": "0.9rem"})
                ])
            ], className="d-flex align-items-center mb-4")
        ]),
        
        # Key Performance Metrics (Compact 4-card layout)
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-users fa-lg", 
                                  style={"color": color_scheme['primary']}),
                            html.H4(f"{nationality_data['total_visitors']:,}", 
                                   className="metric-value mb-1"),
                            html.P("Total Visitors", className="metric-label mb-0")
                        ], className="text-center")
                    ], className="metric-card", style={
                        "background": f"linear-gradient(145deg, white 0%, {color_scheme['light']} 100%)",
                        "border": f"1px solid {color_scheme['border']}",
                        "borderRadius": "16px",
                        "padding": "1.5rem",
                        "height": "100%"
                    })
                ], width=3),
                
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-smile fa-lg", 
                                  style={"color": color_scheme['primary']}),
                            html.H4(f"{nationality_data['satisfaction_rate']:.1f}%", 
                                   className="metric-value mb-1"),
                            html.P("Satisfaction Rate", className="metric-label mb-0")
                        ], className="text-center")
                    ], className="metric-card", style={
                        "background": f"linear-gradient(145deg, white 0%, {color_scheme['light']} 100%)",
                        "border": f"1px solid {color_scheme['border']}",
                        "borderRadius": "16px",
                        "padding": "1.5rem",
                        "height": "100%"
                    })
                ], width=3),
                
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-map-marked-alt fa-lg", 
                                  style={"color": color_scheme['primary']}),
                            html.H4(f"{len(nationality_data.get('cities_visited', {}))}", 
                                   className="metric-value mb-1"),
                            html.P("Cities Visited", className="metric-label mb-0")
                        ], className="text-center")
                    ], className="metric-card", style={
                        "background": f"linear-gradient(145deg, white 0%, {color_scheme['light']} 100%)",
                        "border": f"1px solid {color_scheme['border']}",
                        "borderRadius": "16px",
                        "padding": "1.5rem",
                        "height": "100%"
                    })
                ], width=3),
                
                dbc.Col([                    html.Div([
                        html.Div([
                            html.I(className="fas fa-chart-bar fa-lg", 
                                  style={"color": color_scheme['primary']}),
                            html.H4(f"{min(nationality_data['total_visitors'] / 20, 100):.1f}%", 
                                   className="metric-value mb-1"),
                            html.P("Market Presence", className="metric-label mb-0")
                        ], className="text-center")
                    ], className="metric-card", style={
                        "background": f"linear-gradient(145deg, white 0%, {color_scheme['light']} 100%)",
                        "border": f"1px solid {color_scheme['border']}",
                        "borderRadius": "16px",
                        "padding": "1.5rem",
                        "height": "100%"
                    })
                ], width=3)
            ], className="mb-4")
        ]),
        
        # Tourist Preference Analysis (Compact layout)
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.H6([
                        html.I(className="fas fa-heart me-2", style={"color": "#10b981"}),
                        "Top Preferences"
                    ], className="section-title mb-2"),
                    html.Div([
                        html.Div([
                            html.Span(f"{item['thing'].title()}", className="d-block fw-bold", 
                                     style={"fontSize": "0.85rem"}),
                            dbc.Badge(f"{item['mentions']}", color="success", pill=True, 
                                     className="badge-sm")
                        ], className="compact-item mb-2")
                        for item in nationality_data.get('top_liked_things', [])[:4]  # Top 4
                    ]) if nationality_data.get('top_liked_things') else html.P("No data", className="text-muted small")
                ], width=6),
                
                dbc.Col([
                    html.H6([
                        html.I(className="fas fa-exclamation-triangle me-2", style={"color": "#ef4444"}),
                        "Main Concerns"
                    ], className="section-title mb-2"),
                    html.Div([
                        html.Div([
                            html.Span(f"{item['thing'].title()}", className="d-block fw-bold", 
                                     style={"fontSize": "0.85rem"}),
                            dbc.Badge(f"{item['mentions']}", color="danger", pill=True, 
                                     className="badge-sm")
                        ], className="compact-item mb-2")
                        for item in nationality_data.get('top_disliked_things', [])[:4]  # Top 4
                    ]) if nationality_data.get('top_disliked_things') else html.P("No data", className="text-muted small")
                ], width=6)
            ], className="mb-4")
        ]),
        
        # Geographic Distribution
        html.Div([
            html.H6([
                html.I(className="fas fa-globe me-2", style={"color": color_scheme['primary']}),
                "Geographic Distribution"
            ], className="section-title mb-2"),
            html.Div([
                dbc.Badge(f"{city} ({count:,})", color="primary", className="me-1 mb-1", 
                         style={"background": color_scheme['primary']}) 
                for city, count in list(nationality_data.get('cities_visited', {}).items())[:6]  # Top 6 cities
            ] + ([dbc.Badge(f"+{len(nationality_data.get('cities_visited', {}))-6} more", 
                           color="secondary", className="me-1 mb-1")] 
                 if len(nationality_data.get('cities_visited', {})) > 6 else []))
        ], className="mb-4"),
        
        # Strategic Market Insights
        html.Div([
            html.H6([
                html.I(className="fas fa-chart-line me-2", style={"color": color_scheme['primary']}),
                "Market Intelligence"
            ], className="section-title mb-3"),
            html.Div([
                html.Div([
                    dcc.Markdown(insight, className="insight-item mb-2")
                    for insight in nationality_data.get('insights', [])[:4]  # Top 4 insights
                ]) if nationality_data.get('insights') else html.P("Generating insights...", className="text-muted")
            ], className="insights-content")
        ], className="mb-4"),
        
        # Strategic Recommendations
        html.Div([
            html.H6([
                html.I(className="fas fa-rocket me-2", style={"color": color_scheme['primary']}),
                "Strategic Actions"
            ], className="section-title mb-3"),
            html.Div([
                create_modern_nationality_recommendation_card(rec, color_scheme)
                for rec in nationality_data.get('recommendations', [])[:3]  # Top 3 recommendations
            ]) if nationality_data.get('recommendations') else html.P("Generating recommendations...", className="text-muted")
        ])
    ], className="modal-analysis-content")

def create_modern_nationality_recommendation_card(recommendation, color_scheme):
    """Create a modern, compact nationality recommendation card"""
    priority_colors = {
        'NATIONAL PRIORITY': '#7c3aed',
        'STRATEGIC': '#2563eb',
        'EMERGENCY': '#dc2626',
        'CRITICAL': '#dc2626',
        'HIGH': '#ea580c',
        'URGENT': '#dc2626',
        'MEDIUM': '#059669',
        'LOW': '#6b7280'
    }
    
    priority = recommendation.get('priority', 'MEDIUM')
    priority_color = priority_colors.get(priority, '#6b7280')
    
    return html.Div([
        html.Div([
            html.Div([
                html.Span(recommendation.get('category', 'Strategy'), 
                         className="fw-bold d-block mb-1",
                         style={"color": color_scheme['primary'], "fontSize": "0.9rem"}),
                dbc.Badge(priority, color="primary", className="mb-2",
                         style={"background": priority_color, "fontSize": "0.7rem"})
            ]),
            html.P(recommendation.get('action', ''), 
                  className="recommendation-text mb-2",
                  style={"fontSize": "0.85rem", "lineHeight": "1.4"}),
            
            # Compact metrics row
            html.Div([
                html.Small([
                    html.I(className="fas fa-clock me-1"),
                    recommendation.get('timeline', 'TBD')
                ], className="text-muted me-3"),
                html.Small([
                    html.I(className="fas fa-chart-line me-1"),
                    recommendation.get('expected_roi', 'TBD ROI')
                ], className="text-muted")
            ], className="d-flex")
        ])    ], className="recommendation-card mb-3", style={
        "background": "linear-gradient(145deg, #ffffff 0%, #f8fafc 100%)",
        "border": f"1px solid {color_scheme['border']}",
        "borderRadius": "12px",
        "padding": "1rem",
        "borderLeft": f"4px solid {priority_color}"
    })

def create_compact_nationality_recommendation_card(recommendation):
    """Create a compact recommendation card component for nationality analysis"""
    priority_colors = {
        'High': '#ef4444',
        'Medium': '#f59e0b',
        'Low': '#10b981'
    }
    
    priority = recommendation.get('priority', 'Medium')
    priority_color = priority_colors.get(priority, '#6b7280')
    
    return html.Div([
        html.Div([
            dbc.Badge(priority, style={"background": priority_color, "font-size": "0.7rem"}, className="mb-1"),
            html.P(recommendation.get('action', ''), className="compact-rec-action"),
            html.Small([
                html.I(className="fas fa-building me-1"),
                recommendation.get('department', 'Tourism Board')
            ], className="text-muted")
        ], className="compact-rec-content")
    ], className="compact-recommendation-card")

def create_nationality_recommendation_card(recommendation):
    """Create a recommendation card component for nationality analysis"""
    # Color coding by priority
    priority_colors = {
        'High': {'bg': '#fef2f2', 'border': '#ef4444', 'icon': 'fa-exclamation-circle', 'badge': 'danger'},
        'Medium': {'bg': '#fffbeb', 'border': '#f59e0b', 'icon': 'fa-info-circle', 'badge': 'warning'},
        'Low': {'bg': '#f0fdf4', 'border': '#10b981', 'icon': 'fa-check-circle', 'badge': 'success'}
    }
    
    priority = recommendation.get('priority', 'Medium')
    colors = priority_colors.get(priority, priority_colors['Medium'])
    
    return html.Div([
        html.Div([
            html.Div([
                html.I(className=f"fas {colors['icon']} me-2"),
                html.Strong(recommendation.get('category', 'General'), className="recommendation-category"),
                dbc.Badge(priority, color=colors['badge'], className="ms-auto")
            ], className="d-flex align-items-center justify-content-between mb-2"),
            
            html.P(recommendation.get('action', 'No action specified'), className="recommendation-action mb-2"),
            
            html.Small([
                html.I(className="fas fa-bullseye me-1"),
                html.Strong("Expected Impact: "),
                recommendation.get('expected_impact', 'Not specified')
            ], className="recommendation-impact text-muted")
        ], className="p-3", style={
            "background": colors['bg'], 
            "border-radius": "8px", 
            "border-left": f"4px solid {colors['border']}"
        })
    ], className="recommendation-card mb-3")

def create_compact_recommendation_card(recommendation):
    """Create a compact recommendation card component for enhanced modal"""
    priority_colors = {
        'Critical': '#dc2626',
        'High': '#ea580c', 
        'Medium': '#d97706',
        'Low': '#65a30d'
    }
    
    priority_color = priority_colors.get(recommendation['priority'], '#6b7280')
    
    return html.Div([
        html.Div([
            dbc.Badge(recommendation['priority'], style={"background": priority_color, "font-size": "0.7rem"}, className="mb-1"),
            html.P(recommendation['action'], className="compact-rec-action"),
            html.Small([
                html.I(className="fas fa-user-tie me-1"),
                recommendation['responsible']
            ], className="text-muted")
        ], className="compact-rec-content")
    ], className="compact-recommendation-card")

def create_hotel_analysis_content(hotel_analysis, color_scheme):
    """Create hotel analysis visualization content"""
    if not hotel_analysis or hotel_analysis.get('total_hotels_analyzed', 0) == 0:
        return html.Div([
            html.P("No hotel-specific data available for this word.", className="text-muted text-center",
                   style={"padding": "2rem"})
        ])
    
    top_positive = hotel_analysis.get('top_positive_hotels', [])
    top_negative = hotel_analysis.get('top_negative_hotels', [])
    
    return html.Div([
        dbc.Row([
            # Top Positive Hotels Column
            dbc.Col([
                html.Div([
                    html.H6([
                        html.I(className="fas fa-thumbs-up me-2", style={"color": "#10b981"}),
                        "Top 5 Hotels - Positive Context"
                    ], style={"color": "#10b981", "font-weight": "600", "margin-bottom": "1rem"}),
                    
                    html.Div([
                        create_hotel_card(hotel, "positive") for hotel in top_positive
                    ]) if top_positive else html.P("No hotels found in positive context.", className="text-muted")
                ])
            ], width=6),
            
            # Top Negative Hotels Column
            dbc.Col([
                html.Div([
                    html.H6([
                        html.I(className="fas fa-thumbs-down me-2", style={"color": "#ef4444"}),
                        "Top 5 Hotels - Negative Context"
                    ], style={"color": "#ef4444", "font-weight": "600", "margin-bottom": "1rem"}),
                    
                    html.Div([
                        create_hotel_card(hotel, "negative") for hotel in top_negative
                    ]) if top_negative else html.P("No hotels found in negative context.", className="text-muted")
                ])
            ], width=6)
        ])
    ])

def create_hotel_card(hotel_data, sentiment_type):
    """Create individual hotel card for analysis"""
    sentiment_color = "#10b981" if sentiment_type == "positive" else "#ef4444"
    sentiment_bg = "#f0fdf4" if sentiment_type == "positive" else "#fef2f2"
    
    # Calculate percentage of mentions in this sentiment
    total_mentions = hotel_data['total_mentions']
    sentiment_mentions = hotel_data['positive_mentions'] if sentiment_type == "positive" else hotel_data['negative_mentions']
    sentiment_percentage = (sentiment_mentions / total_mentions * 100) if total_mentions > 0 else 0
    
    return html.Div([
        dbc.Card([
            dbc.CardBody([
                # Hotel name and city
                html.Div([
                    html.H6(hotel_data['name'], className="card-title", 
                           style={"color": "#1f2937", "margin-bottom": "0.25rem", "font-weight": "600"}),
                    html.Small(f"📍 {hotel_data['city']}", className="text-muted")
                ], style={"margin-bottom": "0.75rem"}),
                
                # Statistics row
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Strong(f"{sentiment_mentions}", style={"color": sentiment_color, "font-size": "1.1rem"}),
                            html.Br(),
                            html.Small("Mentions", className="text-muted")
                        ], className="text-center")
                    ], width=4),
                    
                    dbc.Col([
                        html.Div([
                            html.Strong(f"{hotel_data['reviews_count']}", style={"color": "#6b7280", "font-size": "1.1rem"}),
                            html.Br(),
                            html.Small("Reviews", className="text-muted")
                        ], className="text-center")
                    ], width=4),
                    
                    dbc.Col([
                        html.Div([
                            html.Strong(f"{sentiment_percentage:.1f}%", style={"color": sentiment_color, "font-size": "1.1rem"}),
                            html.Br(),
                            html.Small("Relevance", className="text-muted")
                        ], className="text-center")
                    ], width=4)
                ], className="g-0"),
                
                # Overall sentiment indicator (if available)
                html.Div([
                    html.Small([
                        html.I(className="fas fa-chart-line me-1"),
                        f"Overall Sentiment: {hotel_data.get('overall_sentiment_avg', 'N/A')}"
                    ], className="text-muted")
                ], style={"margin-top": "0.5rem", "padding-top": "0.5rem", "border-top": "1px solid #e5e7eb"})
            ])
        ], style={
            "border": f"1px solid {sentiment_color}20",
            "background": sentiment_bg,
            "margin-bottom": "0.75rem",
            "border-radius": "8px"
        })    ])

def create_overview_content(city_filter="all", start_date=None, end_date=None):
    """Create the overview page content with KPIs and charts - optimized for performance"""
    import time
    start_time = time.time()
      # Load data and calculate KPIs with performance tracking
    df = load_data()
    
    # Debug: Print available columns for troubleshooting
    print(f"DEBUG: Available columns in dataframe: {list(df.columns) if not df.empty else 'DataFrame is empty'}")
    print(f"DEBUG: DataFrame shape: {df.shape if not df.empty else 'Empty'}")
    if not df.empty:
        print(f"DEBUG: Sample of first few rows:")
        print(df.head(2).to_string())
    
    if df.empty:
        return html.Div([
            html.Div([
                html.I(className="fas fa-exclamation-triangle fa-3x text-warning mb-3"),
                html.H4("No Data Available", className="text-warning"),
                html.P("Unable to load data. Please try refreshing.", className="text-muted")
            ], className="text-center p-5")
        ])
    
    kpis = calculate_kpis(df, city_filter, start_date, end_date)
    
    # Determine satisfaction status and color efficiently
    satisfaction_rate = kpis['satisfaction_rate']
    if satisfaction_rate >= 70:
        satisfaction_color = "success"
        satisfaction_icon = "fa-smile"
    elif satisfaction_rate >= 50:
        satisfaction_color = "warning"
        satisfaction_icon = "fa-meh"
    else:
        satisfaction_color = "danger"
        satisfaction_icon = "fa-frown"
      # Create date range text for insights
    date_range_text = "all time"
    if start_date and end_date:
        date_range_text = f"from {start_date} to {end_date}"
    elif start_date:
        date_range_text = f"from {start_date} onwards"
    elif end_date:
        date_range_text = f"up to {end_date}"
    
    # Create charts with lazy loading approach for better performance
    try:
        import traceback  # Import traceback for debugging
        print(f"📊 Creating charts with data shape: {df.shape}")
        print(f"📊 Data columns: {df.columns.tolist()}")
        print(f"📊 City filter: {city_filter}, Start date: {start_date}, End date: {end_date}")
        
        # Debug column availability for chart functions
        required_columns = {
            'satisfaction_chart': ['Review Date', 'sentiment classification'],
            'word_frequency_chart': ['positive tokens', 'negative tokens'],
            'nationality_chart': ['Nationality', 'reviewer_nationality', 'Reviewer Nationality'],
            'city_satisfaction_chart': ['City', 'sentiment classification']        }
        
        for chart_name, cols in required_columns.items():
            available_cols = [col for col in cols if col in df.columns]
            missing_cols = [col for col in cols if col not in df.columns]
            print(f"📊 {chart_name}: Available columns: {available_cols}, Missing: {missing_cols}")
        
        # Test individual chart creation with better error handling
        try:
            satisfaction_chart = create_satisfaction_trend_chart(df, city_filter, start_date, end_date)
            print("✅ Satisfaction chart created successfully")
        except Exception as e:
            print(f"❌ Satisfaction chart failed: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            satisfaction_chart = go.Figure().add_annotation(
                text="⚠️ Satisfaction chart temporarily unavailable",
                xref="paper", yref="paper", x=0.5, y=0.5,
                xanchor='center', yanchor='middle', showarrow=False,
                font=dict(size=16, color="#ef4444", family="Inter, sans-serif")
            ).update_layout(title="Satisfaction Trend Over Time", height=400, template="plotly_white", paper_bgcolor='rgba(0,0,0,0)')
        
        try:
            word_frequency_chart = create_word_frequency_chart(df, city_filter, start_date, end_date)
            print("✅ Word frequency chart created successfully")
        except Exception as e:
            print(f"❌ Word frequency chart failed: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            word_frequency_chart = go.Figure().add_annotation(
                text="⚠️ Word frequency chart temporarily unavailable",
                xref="paper", yref="paper", x=0.5, y=0.5,
                xanchor='center', yanchor='middle', showarrow=False,
                font=dict(size=16, color="#ef4444", family="Inter, sans-serif")
            ).update_layout(title="Word Frequency Analysis", height=400, template="plotly_white", paper_bgcolor='rgba(0,0,0,0)')
        
        try:
            nationality_chart = create_nationality_chart(df, city_filter, start_date, end_date)
            print("✅ Nationality chart created successfully")
        except Exception as e:
            print(f"❌ Nationality chart failed: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            nationality_chart = go.Figure().add_annotation(
                text="⚠️ Nationality chart temporarily unavailable",
                xref="paper", yref="paper", x=0.5, y=0.5,
                xanchor='center', yanchor='middle', showarrow=False,
                font=dict(size=16, color="#ef4444", family="Inter, sans-serif")            ).update_layout(title="Top Nationalities", height=400, template="plotly_white", paper_bgcolor='rgba(0,0,0,0)')
        
        try:
            city_satisfaction_chart = create_city_satisfaction_chart(df, city_filter, start_date, end_date)
            print("✅ City satisfaction chart created successfully")
        except Exception as e:
            print(f"❌ City satisfaction chart failed: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            city_satisfaction_chart = go.Figure().add_annotation(
                text="⚠️ City satisfaction chart temporarily unavailable",
                xref="paper", yref="paper", x=0.5, y=0.5,
                xanchor='center', yanchor='middle', showarrow=False,
                font=dict(size=16, color="#ef4444", family="Inter, sans-serif")
            ).update_layout(title="City Satisfaction Comparison", height=400, template="plotly_white", paper_bgcolor='rgba(0,0,0,0)')
        
        try:
            economics_chart = create_tourism_economics_chart(df, city_filter, start_date, end_date)
            print("✅ Tourism economics chart created successfully")
        except Exception as e:
            print(f"❌ Tourism economics chart failed: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            economics_chart = go.Figure().add_annotation(
                text="⚠️ Economics chart temporarily unavailable",
                xref="paper", yref="paper", x=0.5, y=0.5,
                xanchor='center', yanchor='middle', showarrow=False,
                font=dict(size=16, color="#ef4444", family="Inter, sans-serif")
            ).update_layout(title="Tourism Analytics - Visitor Engagement Analysis", height=350, template="plotly_white", paper_bgcolor='rgba(0,0,0,0)')
        
        try:
            seasonal_chart = create_seasonal_analytics_chart(df, city_filter, start_date, end_date)
            print("✅ Seasonal analytics chart created successfully")
        except Exception as e:
            print(f"❌ Seasonal analytics chart failed: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            seasonal_chart = go.Figure().add_annotation(
                text="⚠️ Seasonal chart temporarily unavailable",
                xref="paper", yref="paper", x=0.5, y=0.5,
                xanchor='center', yanchor='middle', showarrow=False,
                font=dict(size=16, color="#ef4444", family="Inter, sans-serif")
            ).update_layout(title="Seasonal & Temporal Analytics", height=500, template="plotly_white", paper_bgcolor='rgba(0,0,0,0)')
    
    except Exception as e:
        print(f"❌ Error creating charts: {e}")
        print(f"❌ Traceback: {traceback.format_exc()}")
        # Fallback to empty Plotly figures instead of Div elements
        satisfaction_chart = go.Figure().add_annotation(
            text="⚠️ Chart temporarily unavailable",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="#ef4444", family="Inter, sans-serif")
        ).update_layout(
            title="Satisfaction Trend Over Time",
            height=400,
            template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        word_frequency_chart = go.Figure().add_annotation(
            text="⚠️ Chart temporarily unavailable",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="#ef4444", family="Inter, sans-serif")
        ).update_layout(
            title="Word Frequency Analysis",
            height=400,
            template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        nationality_chart = go.Figure().add_annotation(
            text="⚠️ Chart temporarily unavailable",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="#ef4444", family="Inter, sans-serif")
        ).update_layout(
            title="Top Nationalities",
            height=400,
            template="plotly_white",            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        city_satisfaction_chart = go.Figure().add_annotation(
            text="⚠️ Chart temporarily unavailable",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="#ef4444", family="Inter, sans-serif")
        ).update_layout(
            title="City Satisfaction Comparison",
            height=400,
            template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        economics_chart = go.Figure().add_annotation(
            text="⚠️ Chart temporarily unavailable",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="#ef4444", family="Inter, sans-serif")        ).update_layout(
            title="Tourism Analytics - Visitor Engagement Analysis",
            height=500,
            template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        seasonal_chart = go.Figure().add_annotation(
            text="⚠️ Chart temporarily unavailable",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="#ef4444", family="Inter, sans-serif")
        ).update_layout(
            title="Seasonal & Temporal Analytics",
            height=500,
            template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)'
        )
    
    elapsed = time.time() - start_time
    print(f"✅ Overview content created in {elapsed:.3f}s")
    
    return html.Div([
        # Word Analysis Modal
        create_word_analysis_modal(),
        
        # Nationality Analysis Modal
        create_nationality_analysis_modal(),
        
        # KPIs Section
        html.Div([
            html.H2([
                html.Span("📊", style={"fontSize": "2rem", "marginRight": "1rem"}),
                "Key Performance Indicators"
            ], className="section-title mb-4"),
            
            # KPI Cards Row
            dbc.Row([
                create_kpi_card(
                    title="Total Visitors",
                    value=f"{kpis['total_visitors']:,}",
                    subtitle="Total number of hotel guests",
                    icon="fa-users",
                    color_class="primary"
                ),
                create_kpi_card(
                    title="Positive Reviews",
                    value=f"{kpis['positive_reviews']:,}",
                    subtitle="Happy customers",
                    icon="fa-thumbs-up",
                    color_class="success"
                ),
                create_kpi_card(
                    title="Negative Reviews",
                    value=f"{kpis['negative_reviews']:,}",
                    subtitle="Areas for improvement",
                    icon="fa-thumbs-down",
                    color_class="danger"
                ),
                create_kpi_card(
                    title="Satisfaction Rate",
                    value=f"{kpis['satisfaction_rate']}%",
                    subtitle="Customer happiness score",
                    icon=satisfaction_icon,
                    color_class=satisfaction_color
                )
            ], className="overview-cards-row")
        ], className="kpi-section"),        # Charts Section
        html.Div([
            html.H2([
                html.Span("📈", style={"fontSize": "2rem", "marginRight": "1rem"}),
                "Analytics & Trends"
            ], className="section-title mb-4"),
              # First Row - Satisfaction Trend Chart (Full Width)
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.Span("📊", style={"fontSize": "1.3rem", "marginRight": "0.5rem", "color": "#3b82f6"}),
                            html.Span("Satisfaction Trend Over Time", className="modern-title", style={"fontSize": "1.1rem"})
                        ], className="chart-item-header", style={"marginBottom": "1rem"}),
                        html.Div([                            dcc.Graph(
                                figure=satisfaction_chart,
                                className="trend-chart",
                                config={
                                    'displayModeBar': True,
                                    'displaylogo': False,
                                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                                    'responsive': True
                                },
                                style={"height": "500px"}
                            )                        ], className="chart-item-content")
                    ], className="chart-item enhanced-card", style={"minHeight": "520px"})
                ], width=12)
            ], className="mb-4"),# Second Row - Word Frequency and Nationality Charts Side by Side
            dbc.Row([
                # Word Frequency Chart (Left Column)
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.Span("💬", style={"fontSize": "1.3rem", "marginRight": "0.5rem", "color": "#8b5cf6"}),
                            html.Span("Top Mentioned Words - Positive vs Negative", className="modern-title", style={"fontSize": "1.05rem"})
                        ], className="chart-item-header", style={"marginBottom": "1rem"}),
                        html.Div([                            dcc.Graph(
                                id="word-frequency-chart",
                                figure=word_frequency_chart,
                                className="word-frequency-chart",
                                config={
                                    'displayModeBar': True,
                                    'displaylogo': False,
                                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                                    'responsive': True
                                },
                                style={"height": "450px"}
                            )
                        ], className="chart-item-content")
                    ], className="chart-item enhanced-card word-frequency-container", style={"minHeight": "470px"})
                ], width=6),
                
                # Nationality Chart (Right Column)
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.Span("🌍", style={"fontSize": "1.3rem", "marginRight": "0.5rem", "color": "#10b981"}),
                            html.Span("Top Nationalities Visiting", className="modern-title", style={"fontSize": "1.05rem"})
                        ], className="chart-item-header", style={"marginBottom": "1rem"}),
                        html.Div([                            dcc.Graph(
                                id="nationality-chart",
                                figure=nationality_chart,
                                className="nationality-chart",
                                config={
                                    'displayModeBar': True,
                                    'displaylogo': False,
                                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                                    'responsive': True
                                },
                                style={"height": "450px"}
                            )                        ], className="chart-item-content")
                    ], className="chart-item enhanced-card", style={"minHeight": "470px"})
                ], width=6)
            ], className="mb-4"),
              # Third Row - Tourism Analytics & Seasonal Analytics Charts Side by Side
            dbc.Row([
                # Tourism Analytics Chart (Left Column)
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.Span("💰", style={"fontSize": "1.3rem", "marginRight": "0.5rem", "color": "#f59e0b"}),
                            html.Span("Tourism Analytics - Visitor Engagement", className="modern-title", style={"fontSize": "1.05rem"})
                        ], className="chart-item-header", style={"marginBottom": "1rem"}),
                        html.Div([                            dcc.Graph(
                                id="tourism-economics-chart",
                                figure=economics_chart,
                                className="economics-chart",
                                config={
                                    'displayModeBar': True,
                                    'displaylogo': False,
                                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                                    'responsive': True
                                },
                                style={"height": "500px"}
                            )                        ], className="chart-item-content")
                    ], className="chart-item enhanced-card", style={"minHeight": "520px"})
                ], width=6),
                
                # Seasonal Analytics Chart (Right Column)
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.Span("📅", style={"fontSize": "1.3rem", "marginRight": "0.5rem", "color": "#ec4899"}),
                            html.Span("Seasonal & Temporal Analytics", className="modern-title", style={"fontSize": "1.05rem"})
                        ], className="chart-item-header", style={"marginBottom": "1rem"}),
                        html.Div([                            dcc.Graph(
                                id="seasonal-analytics-chart",
                                figure=seasonal_chart,
                                className="seasonal-chart",
                                config={
                                    'displayModeBar': True,
                                    'displaylogo': False,
                                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                                    'responsive': True
                                },
                                style={"height": "500px"}
                            )
                        ], className="chart-item-content")                    ], className="chart-item enhanced-card", style={"minHeight": "520px"})
                ], width=6)
            ], className="mb-4"),
              # Fourth Row - City Satisfaction Comparison Chart (Full Width) - Only show when "All Cities" selected
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.Span("🏆", style={"fontSize": "1.3rem", "marginRight": "0.5rem", "color": "#3b82f6"}),
                            html.Span("City Satisfaction Comparison", className="modern-title", style={"fontSize": "1.1rem"})
                        ], className="chart-item-header", style={"marginBottom": "1rem"}),
                        html.Div([
                            dcc.Graph(
                                id=f"city-satisfaction-chart-{city_filter}",
                                figure=city_satisfaction_chart,
                                className="city-satisfaction-chart",
                                config={
                                    'displayModeBar': True,
                                    'displaylogo': False,
                                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                                    'responsive': True
                                },
                                style={"height": "480px"}
                            )
                        ], className="chart-item-content")
                    ], className="chart-item enhanced-card", style={"minHeight": "500px"})
                ], width=12)
            ], className="mb-4", style={"display": "block" if city_filter == "all" else "none"})
        ], className="charts-section")
    ])

def get_cities_from_data():
    """Get list of cities from the data for the dropdown filter"""
    try:
        from database_config import get_cities_from_database
        
        # First try to get cities directly from database
        cities = get_cities_from_database()
        
        if cities:
            city_options = [{'label': 'All Cities', 'value': 'all'}]
            city_options.extend([{'label': city, 'value': city} for city in sorted(cities) if city])
            return city_options
        
        # Fallback to loading data if direct query fails
        df = load_data()
        
        # Check if City column exists
        if 'City' not in df.columns:
            print(f"Warning: 'City' column not found in data. Available columns: {list(df.columns)}")
            # Try to find a similar column
            possible_city_columns = [col for col in df.columns if 'city' in col.lower() or 'location' in col.lower() or 'destination' in col.lower()]
            if possible_city_columns:
                city_column = possible_city_columns[0]
                print(f"Using '{city_column}' as city column instead")
                cities = df[city_column].dropna().unique().tolist()
            else:
                print("No suitable city column found. Using default options.")
                return [{'label': 'All Cities', 'value': 'all'}, {'label': 'No Data Available', 'value': 'none'}]
        else:
            cities = df['City'].dropna().unique().tolist()
        
        city_options = [{'label': 'All Cities', 'value': 'all'}]
        city_options.extend([{'label': city, 'value': city} for city in sorted(cities) if city])
        return city_options
        
    except Exception as e:
        print(f"Error in get_cities_from_data: {e}")
        # Return a default option in case of any error
        return [{'label': 'All Cities', 'value': 'all'}, {'label': 'Error Loading Data', 'value': 'error'}]

def analyze_nationality_details(df, selected_nationality, city_filter="all", start_date=None, end_date=None):
    """Analyze detailed statistics for a specific nationality"""
    # Apply filters same as other functions
    filtered_df = df.copy()
    
    # Apply city filter
    if city_filter != "all":
        filtered_df = filtered_df[filtered_df['City'] == city_filter]
    
    # Apply date filter if dates are provided
    if start_date or end_date:
        filtered_df['Review Date'] = pd.to_datetime(filtered_df['Review Date'], errors='coerce')
        
        if start_date:
            start_date = pd.to_datetime(start_date)
            filtered_df = filtered_df[filtered_df['Review Date'] >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            filtered_df = filtered_df[filtered_df['Review Date'] <= end_date]
      # Filter for the specific nationality
    if 'Reviewer Nationality' in filtered_df.columns:
        nationality_df = filtered_df[filtered_df['Reviewer Nationality'] == selected_nationality]
    else:
        nationality_df = filtered_df[filtered_df['Nationality'] == selected_nationality]
    
    # Return empty data structure if no data found for this nationality
    if nationality_df.empty:
        return {
            'nationality': selected_nationality,
            'total_visitors': 0,
            'positive_reviews': 0,
            'negative_reviews': 0,
            'satisfaction_rate': 0,
            'cities_visited': [],
            'hotels_visited': [],
            'top_liked_things': [],
            'top_disliked_things': [],
            'monthly_trend': [],
            'insights': [],
            'recommendations': []
        }
    
    # Calculate basic metrics
    total_visitors = len(nationality_df)
    positive_reviews = len(nationality_df[nationality_df['sentiment classification'] > 0])
    negative_reviews = len(nationality_df[nationality_df['sentiment classification'] < 0])
    satisfaction_rate = (positive_reviews / total_visitors * 100) if total_visitors > 0 else 0
      # Get cities and hotels visited
    cities_visited = nationality_df['City'].value_counts().to_dict()
    
    # Handle hotel name column which might be different between database and CSV
    hotel_column = 'Hotel Name' if 'Hotel Name' in nationality_df.columns else 'hotel_name'
    hotels_visited = nationality_df[hotel_column].value_counts().to_dict()
      # Analyze top liked and disliked things
    liked_counter = Counter()
    disliked_counter = Counter()
    
    for _, row in nationality_df.iterrows():
        # Parse positive tokens for liked things
        pos_tokens_value = row['positive tokens']
        if pos_tokens_value is not None and not (isinstance(pos_tokens_value, float) and pd.isna(pos_tokens_value)) and isinstance(pos_tokens_value, str) and pos_tokens_value.strip():
            try:
                pos_tokens = ast.literal_eval(pos_tokens_value)
                if isinstance(pos_tokens, list):
                    word_tokens = [token.lower().strip() for token in pos_tokens if token.strip()]
                    for word in word_tokens:
                        liked_counter[word] += 1
            except:
                if isinstance(pos_tokens_value, str):
                    clean_tokens = pos_tokens_value.replace('[', '').replace(']', '').replace("'", '').replace('"', '')
                    word_tokens = [token.lower().strip() for token in clean_tokens.split(',') if token.strip()]
                    for word in word_tokens:
                        liked_counter[word] += 1
          # Parse negative tokens for disliked things
        neg_tokens_value = row['negative tokens']
        if neg_tokens_value is not None and not (isinstance(neg_tokens_value, float) and pd.isna(neg_tokens_value)) and isinstance(neg_tokens_value, str) and neg_tokens_value.strip():
            try:
                neg_tokens = ast.literal_eval(neg_tokens_value)
                if isinstance(neg_tokens, list):
                    word_tokens = [token.lower().strip() for token in neg_tokens if token.strip()]
                    for word in word_tokens:
                        disliked_counter[word] += 1
            except:
                if isinstance(neg_tokens_value, str):
                    clean_tokens = neg_tokens_value.replace('[', '').replace(']', '').replace("'", '').replace('"', '')
                    word_tokens = [token.lower().strip() for token in clean_tokens.split(',') if token.strip()]
                    for word in word_tokens:
                        disliked_counter[word] += 1
    
    # Get top 5 liked and disliked things
    top_liked_things = [{'thing': word, 'mentions': count} for word, count in liked_counter.most_common(5)]
    top_disliked_things = [{'thing': word, 'mentions': count} for word, count in disliked_counter.most_common(5)]
    
    # Calculate monthly trend
    monthly_trend = []
    if not nationality_df.empty:
        review_dates = pd.to_datetime(nationality_df['Review Date'], errors='coerce')
        review_dates = review_dates.dropna()
        
        if len(review_dates) > 0:
            df_temp = pd.DataFrame({'date': review_dates})
            df_temp['month'] = df_temp['date'].dt.to_period('M')
            monthly_counts = df_temp.groupby('month').size().reset_index(name='count')
            monthly_trend = [{'month': str(period), 'count': count} for period, count in zip(monthly_counts['month'], monthly_counts['count'])]
      # Generate insights
    insights = generate_nationality_insights(selected_nationality, total_visitors, satisfaction_rate, 
                                           cities_visited, top_liked_things, top_disliked_things)
    
    # Generate recommendations
    recommendations = generate_nationality_recommendations(selected_nationality, total_visitors, satisfaction_rate, 
                                                         cities_visited, top_liked_things, top_disliked_things)
    
    return {
        'nationality': selected_nationality,
        'total_visitors': total_visitors,
        'positive_reviews': positive_reviews,
        'negative_reviews': negative_reviews,
        'satisfaction_rate': round(satisfaction_rate, 1),
        'cities_visited': cities_visited,
        'hotels_visited': hotels_visited,
        'top_liked_things': top_liked_things,
        'top_disliked_things': top_disliked_things,
        'monthly_trend': monthly_trend,
        'insights': insights,
        'recommendations': recommendations
    }

def generate_nationality_insights(nationality, total_visitors, satisfaction_rate, cities_visited, top_liked, top_disliked):
    """Generate only reliable, data-backed insights for nationality analysis"""
    insights = []
    
    try:
        df = load_data()
        total_tourists = len(df)
        
        # Only include insights based on significant data
        if total_tourists > 0 and total_visitors >= 20:  # Only for meaningful data
            market_share = (total_visitors / total_tourists * 100)
            
            # Market size classification - data-based only
            if total_visitors >= 100:
                insights.append(f"**Major Market**: {nationality} represents {market_share:.1f}% of visitors ({total_visitors:,} tourists). This is a significant market requiring dedicated attention.")
            elif total_visitors >= 50:
                insights.append(f"**Growing Market**: {nationality} shows {market_share:.1f}% presence ({total_visitors:,} visitors). Potential for targeted growth initiatives.")
            else:
                insights.append(f"**Emerging Market**: {nationality} currently {market_share:.1f}% share ({total_visitors:,} visitors). Monitor for growth opportunities.")
            
            # Satisfaction analysis - realistic assessment
            if satisfaction_rate >= 75:
                insights.append(f"**High Satisfaction**: {satisfaction_rate:.1f}% satisfaction rate indicates successful service delivery for {nationality} visitors.")
            elif satisfaction_rate >= 60:
                insights.append(f"**Moderate Satisfaction**: {satisfaction_rate:.1f}% satisfaction suggests room for targeted improvements.")
            else:
                insights.append(f"**Low Satisfaction**: {satisfaction_rate:.1f}% satisfaction indicates significant service quality issues requiring attention.")
            
            # Geographic distribution - factual only
            if len(cities_visited) == 1:
                city_name = list(cities_visited.keys())[0]
                insights.append(f"**Single Destination Focus**: {nationality} visitors concentrate in {city_name}. Consider diversification strategies.")
            elif len(cities_visited) >= 4:
                top_cities = sorted(cities_visited.items(), key=lambda x: x[1], reverse=True)[:3]
                insights.append(f"**Multi-City Tourists**: {nationality} visitors spread across {len(cities_visited)} destinations, primarily {', '.join([city[0] for city in top_cities])}.")
            elif len(cities_visited) == 2:
                insights.append(f"**Two-City Pattern**: {nationality} visitors typically visit {' and '.join(cities_visited.keys())}.")
            
            # Preference analysis - only if data exists
            if top_liked and len(top_liked) > 0:
                most_liked = top_liked[0]['thing']
                insights.append(f"**Positive Feedback**: {nationality} visitors particularly appreciate '{most_liked}'. Consider highlighting this in marketing.")
            
            if top_disliked and len(top_disliked) > 0:
                most_disliked = top_disliked[0]['thing']
                insights.append(f"**Service Issue**: {nationality} visitors frequently mention concerns about '{most_disliked}'. Priority area for improvement.")
    
    except Exception:
        # Minimal fallback insights
        if total_visitors >= 20:
            insights.append(f"**Market Presence**: {nationality} represents a notable visitor segment with {total_visitors:,} tourists.")
            if satisfaction_rate >= 70:
                insights.append(f"**Positive Experience**: {satisfaction_rate:.1f}% satisfaction indicates generally good service delivery.")
            else:
                insights.append(f"**Service Opportunity**: {satisfaction_rate:.1f}% satisfaction suggests areas for service improvement.")
      # Ensure we don't return empty insights    if not insights:
        insights.append(f"**Data Note**: Limited data available for {nationality} analysis. More visitor data needed for meaningful insights.")
    
    return insights

def generate_nationality_recommendations(nationality, total_visitors, satisfaction_rate, cities_visited, top_liked, top_disliked):
    """Generate data-driven recommendations for nationality-specific improvements"""
    recommendations = []
    
    try:
        # Calculate data-driven market metrics
        market_size_category = "MAJOR" if total_visitors >= 100 else "MEDIUM" if total_visitors >= 50 else "EMERGING"
        satisfaction_category = "HIGH" if satisfaction_rate >= 80 else "GOOD" if satisfaction_rate >= 70 else "NEEDS_IMPROVEMENT"
        cities_reach = len(cities_visited) if cities_visited else 1
        
        # Strategic market development based on visitor volume and satisfaction
        if total_visitors >= 100:  # Major market
            recommendations.extend([
                {
                    'category': '🎯 Strategic Market Development',
                    'priority': 'HIGH',
                    'action': f"Develop comprehensive {nationality} tourism strategy. Current {total_visitors:,} visitors represent significant market requiring dedicated attention and service optimization.",
                    'departments': ['Ministry of Tourism', 'Ministry of Foreign Affairs', 'Egyptian Tourism Authority'],
                    'timeline': '60-120 days development',
                    'data_basis': f"{total_visitors:,} visitors, {satisfaction_rate:.1f}% satisfaction, {cities_reach} cities visited",
                    'success_metrics': [
                        f"Increase {nationality} visitor satisfaction to 85%+",
                        "Expand destination coverage and service quality",
                        f"Establish Egypt as preferred destination for {nationality} travelers"
                    ]
                },
                {
                    'category': '🤝 Diplomatic Tourism Relations',
                    'priority': 'STRATEGIC',
                    'action': f"Establish formal tourism cooperation with {nationality} government. Develop bilateral agreements, airline partnerships, and cultural exchange programs.",
                    'departments': ['Ministry of Foreign Affairs', 'Tourism Authority', 'Cultural Affairs'],
                    'timeline': '90-180 days negotiation and implementation',
                    'data_basis': f"Based on {total_visitors:,} annual visitors and {satisfaction_rate:.1f}% satisfaction",
                    'success_metrics': [
                        "Establish bilateral tourism agreement",
                        "Launch direct flight routes to key Egyptian destinations",
                        "Create cultural exchange programs"
                    ]
                }
            ])
        
        elif total_visitors >= 50:  # Growth market
            recommendations.extend([
                {
                    'category': '📈 Market Growth Strategy',
                    'priority': 'MEDIUM',
                    'action': f"Launch targeted {nationality} market expansion program. Current {total_visitors:,} visitors show growth potential through focused marketing and service improvements.",
                    'departments': ['Tourism Marketing', 'Destination Development', 'Service Quality'],
                    'timeline': '45-90 days program launch',
                    'data_basis': f"{total_visitors:,} visitors with {satisfaction_rate:.1f}% satisfaction across {cities_reach} destinations",
                    'success_metrics': [
                        f"Increase {nationality} visitor numbers by 40%",
                        "Improve service quality ratings to 80%+",
                        "Expand destination options and experiences"
                    ]
                }
            ])
        
        else:  # Emerging market (< 50 visitors)
            recommendations.extend([
                {
                    'category': '🌱 Market Entry Strategy',
                    'priority': 'MEDIUM',
                    'action': f"Develop {nationality} market entry plan. Current {total_visitors:,} visitors represent untapped potential requiring strategic cultural engagement and targeted marketing.",
                    'departments': ['Market Research', 'Cultural Outreach', 'Tourism Development'],
                    'timeline': '60-120 days strategy development',
                    'data_basis': f"{total_visitors:,} visitors, {satisfaction_rate:.1f}% satisfaction",
                    'success_metrics': [
                        "Double visitor numbers within 12 months",
                        "Achieve 75%+ satisfaction rate",
                        "Establish market presence in key {nationality} cities"
                    ]
                }
            ])
        
        # Satisfaction-based recommendations
        if satisfaction_rate < 70:
            recommendations.append({
                'category': '🔧 Service Quality Improvement',
                'priority': 'HIGH',
                'action': f"Address {nationality} visitor concerns to improve {satisfaction_rate:.1f}% satisfaction rate. Focus on cultural sensitivity training and service adaptation.",
                'departments': ['Service Quality', 'Training Development', 'Cultural Affairs'],
                'timeline': '30-60 days implementation',
                'data_basis': f"Based on {satisfaction_rate:.1f}% satisfaction from {total_visitors:,} visitors",
                'success_metrics': [
                    "Improve satisfaction to 80%+ within 6 months",
                    "Reduce service complaints by 50%",
                    "Implement cultural awareness training programs"
                ]
            })
        
        # Preference-based recommendations
        if top_liked and len(top_liked) > 0:
            most_liked = top_liked[0]['thing']
            recommendations.append({
                'category': '✨ Excellence Amplification',
                'priority': 'MEDIUM',
                'action': f"Leverage {nationality} appreciation for '{most_liked}' in marketing and service enhancement. Use this strength to attract more visitors from this market.",
                'departments': ['Marketing', 'Service Development', 'Cultural Promotion'],
                'timeline': '30-45 days campaign development',
                'data_basis': f"Based on positive feedback for '{most_liked}' from {nationality} visitors",
                'success_metrics': [
                    f"Feature '{most_liked}' in {nationality}-targeted marketing",
                    "Develop specialized experiences around this strength",
                    "Measure increase in {nationality} booking interest"
                ]
            })
        
        if top_disliked and len(top_disliked) > 0:
            most_disliked = top_disliked[0]['thing']
            recommendations.append({
                'category': '⚠️ Critical Issue Resolution',
                'priority': 'HIGH',
                'action': f"Address {nationality} concerns about '{most_disliked}'. This represents a major barrier to market growth and satisfaction improvement.",
                'departments': ['Quality Control', 'Operations', 'Cultural Sensitivity'],
                'timeline': '14-30 days rapid response',
                'data_basis': f"Based on negative feedback about '{most_disliked}' from {nationality} visitors",
                'success_metrics': [
                    f"Reduce '{most_disliked}' complaints by 70%",
                    "Implement preventive measures",
                    "Restore {nationality} visitor confidence"
                ]
            })
    
    except Exception as e:
        # Fallback recommendations
        recommendations = [
            {
                'category': 'Data Analysis & Strategy',
                'priority': 'MEDIUM',
                'action': f"Conduct comprehensive analysis of {nationality} tourism patterns and preferences to develop targeted improvement strategy.",
                'departments': ['Tourism Research', 'Analytics', 'Strategic Planning'],
                'timeline': '30-60 days',
                'data_basis': f"Limited data available for {total_visitors:,} {nationality} visitors",
                'success_metrics': [
                    "Complete market analysis report",
                    "Develop actionable improvement plan",
                    "Establish data collection system"
                ]
            }
        ]
    
    return recommendations

