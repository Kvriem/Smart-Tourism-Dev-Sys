import pandas as pd
import dash_bootstrap_components as dbc
from dash import html, dcc
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import ast
from collections import Counter

def load_data():
    """Load and process the data for overview metrics"""
    try:
        df = pd.read_csv('data/data.csv')
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def calculate_kpis(df, city_filter="all", start_date=None, end_date=None):
    """Calculate KPI metrics based on the filtered data"""
    print(f"Debug KPIs - city_filter: {city_filter}, start_date: {start_date}, end_date: {end_date}")
    print(f"Debug KPIs - Initial dataset size: {len(df)}")
    
    if df.empty:
        return {
            'total_visitors': 0,
            'positive_reviews': 0,
            'negative_reviews': 0,
            'satisfaction_rate': 0
        }
    
    # Apply city filter
    if city_filter != "all":
        df = df[df['City'] == city_filter]
        print(f"Debug KPIs - After city filter: {len(df)}")
    
    # Apply date filter if dates are provided
    if start_date or end_date:
        # Convert the 'Review Date' column to datetime
        df['Review Date'] = pd.to_datetime(df['Review Date'], errors='coerce')
        print(f"Debug KPIs - Date column converted")
        
        if start_date:
            start_date = pd.to_datetime(start_date)
            df = df[df['Review Date'] >= start_date]
            print(f"Debug KPIs - After start date filter: {len(df)}")
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            df = df[df['Review Date'] <= end_date]
            print(f"Debug KPIs - After end date filter: {len(df)}")
    
    if df.empty:
        print("Debug KPIs - Dataset is empty after filtering")
        return {
            'total_visitors': 0,
            'positive_reviews': 0,
            'negative_reviews': 0,
            'satisfaction_rate': 0
        }
    
    # Calculate metrics
    total_visitors = len(df)
    positive_reviews = len(df[df['sentiment classification'] == 1])
    negative_reviews = len(df[df['sentiment classification'] == -1])
    
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
    """Create a satisfaction trend over time line chart"""
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
            xaxis_title="Date",
            yaxis_title="Satisfaction Rate (%)",
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
            xaxis_title="Date",
            yaxis_title="Satisfaction Rate (%)",
            height=400,
            template="plotly_white"
        )
        return fig
    
    # Convert Review Date to datetime
    filtered_df['Review Date'] = pd.to_datetime(filtered_df['Review Date'], errors='coerce')
    
    # Group by month to show trend over time
    filtered_df['Month'] = filtered_df['Review Date'].dt.to_period('M')
    
    # Calculate satisfaction rate by month
    monthly_stats = filtered_df.groupby('Month').agg({
        'sentiment classification': ['count', lambda x: (x == 1).sum()]
    }).round(2)
    
    # Flatten column names
    monthly_stats.columns = ['total_reviews', 'positive_reviews']
    monthly_stats['satisfaction_rate'] = (monthly_stats['positive_reviews'] / monthly_stats['total_reviews'] * 100).round(1)
    monthly_stats = monthly_stats.reset_index()
    
    # Convert period back to datetime for plotting
    monthly_stats['Date'] = monthly_stats['Month'].dt.to_timestamp()    # Create the line chart with color coding
    fig = go.Figure()
    
    # Get date range for background shading
    date_min = monthly_stats['Date'].min()
    date_max = monthly_stats['Date'].max()
    
    # Add background shading for satisfaction zones
    # Good zone (70-100%)
    fig.add_shape(
        type="rect",
        x0=date_min,
        x1=date_max,
        y0=70,
        y1=100,
        fillcolor="rgba(16, 185, 129, 0.1)",
        line=dict(width=0),
        layer="below"
    )
    
    # Average zone (50-70%)
    fig.add_shape(
        type="rect",
        x0=date_min,
        x1=date_max,
        y0=50,
        y1=70,
        fillcolor="rgba(245, 158, 11, 0.1)",
        line=dict(width=0),
        layer="below"
    )
    
    # Poor zone (0-50%)
    fig.add_shape(
        type="rect",
        x0=date_min,
        x1=date_max,
        y0=0,
        y1=50,
        fillcolor="rgba(239, 68, 68, 0.1)",
        line=dict(width=0),
        layer="below"
    )
      # Add satisfaction rate line with conditional coloring
    colors = []
    for rate in monthly_stats['satisfaction_rate']:
        if rate >= 70:
            colors.append('#10b981')  # Green for good
        elif rate >= 50:
            colors.append('#f59e0b')  # Yellow for average
        else:
            colors.append('#ef4444')  # Red for poor
    
    fig.add_trace(go.Scatter(
        x=monthly_stats['Date'],
        y=monthly_stats['satisfaction_rate'],
        mode='lines+markers',
        name='Satisfaction Rate',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=10, color=colors, line=dict(width=2, color='white')),
        hovertemplate='<b>%{x|%B %Y}</b><br>' +
                     'Satisfaction Rate: %{y:.1f}%<br>' +
                     '<extra></extra>'
        ))
    
    # Add threshold lines
    # Good satisfaction threshold (70%)
    fig.add_hline(
        y=70,
        line_dash="dot",
        line_color="#10b981",
        line_width=2,
        annotation_text="Good (70%+)",
        annotation_position="bottom right",
        annotation=dict(
            font=dict(color="#10b981", size=12, family="Arial, sans-serif"),
            bgcolor="rgba(16, 185, 129, 0.1)",
            bordercolor="#10b981",
            borderwidth=1
        )
    )
    
    # Poor satisfaction threshold (50%)
    fig.add_hline(
        y=50,
        line_dash="dot",
        line_color="#ef4444",
        line_width=2,
        annotation_text="Poor (Below 50%)",
        annotation_position="top right",
        annotation=dict(
            font=dict(color="#ef4444", size=12, family="Arial, sans-serif"),
            bgcolor="rgba(239, 68, 68, 0.1)",
            bordercolor="#ef4444",
            borderwidth=1
        )
    )
      # Add average line
    avg_satisfaction = monthly_stats['satisfaction_rate'].mean()
    fig.add_hline(
        y=avg_satisfaction,
        line_dash="dash",
        line_color="#6b7280",
        line_width=1.5,
        annotation_text=f"Average: {avg_satisfaction:.1f}%",
        annotation_position="top right",
        annotation=dict(
            font=dict(color="#6b7280", size=11, family="Arial, sans-serif"),
            bgcolor="rgba(107, 114, 128, 0.1)",
            bordercolor="#6b7280",
            borderwidth=1
        )
    )      # Update layout
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Satisfaction Rate (%)",
        height=450,
        template="plotly_white",
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=60, r=60, t=80, b=60),
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
        annotations=[
            dict(
                text="<b>Satisfaction Zones:</b> <span style='color:#10b981'>■ Good (70%+)</span> | <span style='color:#f59e0b'>■ Average (50-70%)</span> | <span style='color:#ef4444'>■ Poor (<50%)</span>",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=-0.15,
                xanchor='center', yanchor='top',
                font=dict(size=11, color="#6b7280")
            )
        ]
    )
      # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(0,0,0,0.1)',
        showline=True,
        linewidth=1,
        linecolor='rgba(0,0,0,0.2)'
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(0,0,0,0.1)',
        showline=True,
        linewidth=1,
        linecolor='rgba(0,0,0,0.2)',
        range=[0, 100]
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
        # Parse positive tokens
        if pd.notna(row['positive tokens']) and row['positive tokens'].strip():
            try:
                pos_tokens = ast.literal_eval(row['positive tokens'])
                if isinstance(pos_tokens, list):
                    word_tokens = [token.lower().strip() for token in pos_tokens if token.strip()]
                else:
                    continue
            except:
                # Handle cases where the tokens are not properly formatted
                if isinstance(row['positive tokens'], str):
                    # Remove brackets and quotes, split by comma
                    clean_tokens = row['positive tokens'].replace('[', '').replace(']', '').replace("'", '').replace('"', '')
                    word_tokens = [token.lower().strip() for token in clean_tokens.split(',') if token.strip()]
                else:
                    continue
            
            # Count each word occurrence in this review (same logic as modal)
            for word in set(word_tokens):  # Use set to avoid double counting in same token list
                word_count = word_tokens.count(word)
                positive_counter[word] += word_count
        
        # Parse negative tokens
        if pd.notna(row['negative tokens']) and row['negative tokens'].strip():
            try:
                neg_tokens = ast.literal_eval(row['negative tokens'])
                if isinstance(neg_tokens, list):
                    word_tokens = [token.lower().strip() for token in neg_tokens if token.strip()]
                else:
                    continue
            except:
                # Handle cases where the tokens are not properly formatted
                if isinstance(row['negative tokens'], str):
                    # Remove brackets and quotes, split by comma
                    clean_tokens = row['negative tokens'].replace('[', '').replace(']', '').replace("'", '').replace('"', '')
                    word_tokens = [token.lower().strip() for token in clean_tokens.split(',') if token.strip()]
                else:
                    continue
            
            # Count each word occurrence in this review (same logic as modal)
            for word in set(word_tokens):  # Use set to avoid double counting in same token list
                word_count = word_tokens.count(word)
                negative_counter[word] += word_count    # Calculate total mentions for each word (positive + negative)
    all_words = set(positive_counter.keys()) | set(negative_counter.keys())
    total_counter = {}
    for word in all_words:
        total_counter[word] = positive_counter.get(word, 0) + negative_counter.get(word, 0)
    
    # Get top 10 words by total mentions (across both sentiments)
    top_words_by_total = sorted(total_counter.items(), key=lambda x: x[1], reverse=True)[:10]
    top_words = [word for word, count in top_words_by_total]
    
    # Create the chart with the same words in both sections
    fig = go.Figure()
    
    if top_words:
        # Positive section: show positive counts for all top words
        pos_counts_for_top_words = [positive_counter.get(word, 0) for word in top_words]
        fig.add_trace(go.Bar(
            y=top_words,
            x=pos_counts_for_top_words,  # Show actual positive counts for top words
            name='Positive Words',
            orientation='h',
            marker=dict(color='#10b981'),
            hovertemplate='<b>%{y}</b><br>Positive Mentions: %{x}<br>Total Mentions: %{customdata}<br>Click for detailed insights<br><extra></extra>',
            text=pos_counts_for_top_words,
            textposition='outside',
            customdata=[total_counter[word] for word in top_words]  # Show total count in hover
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
            hovertemplate='<b>%{y}</b><br>Negative Mentions: %{text}<br>Total Mentions: %{customdata}<br>Click for detailed insights<br><extra></extra>',
            text=neg_counts_for_top_words,  # Show positive numbers in hover
            textposition='outside',
            customdata=[total_counter[word] for word in top_words]  # Show total count in hover
        ))
      # Update layout
    fig.update_layout(
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
            xaxis_title="Number of Visitors",
            yaxis_title="Nationality",
            height=400,
            template="plotly_white"
        )
        return fig
    
    # Count nationalities
    nationality_counts = filtered_df['Reviewer Nationality'].value_counts().head(10)
    
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
        positive_reviews = len(city_df[city_df['sentiment classification'] == 1])
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
        annotation_position="top right",
        annotation=dict(
            font=dict(color="#6b7280", size=11, family="Arial, sans-serif"),
            bgcolor="rgba(107, 114, 128, 0.1)",
            bordercolor="#6b7280",
            borderwidth=1        )
    )
    
    # Update layout with enhanced styling and better spacing
    fig.update_layout(
        title={
        },
        xaxis_title="Cities",
        yaxis_title="Satisfaction Rate (%)",
        height=520,  # Increased height to accommodate better spacing
        template="plotly_white",
        plot_bgcolor='rgba(248, 250, 252, 0.5)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=80, r=80, t=120, b=140),  # Increased bottom margin for legend
        font=dict(family="Arial, sans-serif", size=12, color="#374151"),
        showlegend=False,
        hovermode='closest',
        annotations=[
            dict(
                text="<b>🎨 Color Guide:</b><br>" +
                     "<span style='color:#059669; font-weight:bold'>★ Excellent (80%+)</span> | " +
                     "<span style='color:#10b981; font-weight:bold'>✓ Very Good (70-80%)</span> | " +
                     "<span style='color:#34d399; font-weight:bold'>○ Good (60-70%)</span><br>" +
                     "<span style='color:#fbbf24; font-weight:bold'>~ Average (50-60%)</span> | " +
                     "<span style='color:#fb923c; font-weight:bold'>△ Below Average (40-50%)</span> | " +
                     "<span style='color:#ef4444; font-weight:bold'>✗ Poor (<40%)</span>",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=-0.18,
                xanchor='center', yanchor='top',
                font=dict(size=12, color="#4b5563", family="Arial"),
                bgcolor="rgba(249, 250, 251, 0.9)",
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
        linecolor='#9ca3af',        range=[0, min(105, max(satisfaction_rates) + 10)],  # Dynamic range based on data
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
            'recommendations': []
        }
      # Count total mentions and reviews containing the word
    total_mentions = 0
    reviews_with_word = []
    cities_mentioned = []
    hotels_with_word = []
    positive_mentions = 0
    negative_mentions = 0
    
    # We'll analyze both positive and negative mentions regardless of word_type
    # to get comprehensive sentiment breakdown
      # We'll analyze both positive and negative mentions regardless of word_type
    # to get comprehensive sentiment breakdown
    
    for idx, row in filtered_df.iterrows():
        # Check both positive and negative tokens for comprehensive analysis
        word_found_in_review = False
        word_mentions_in_review = 0
        sentiment_contexts = []
        
        for token_type, token_column in [('positive', 'positive tokens'), ('negative', 'negative tokens')]:
            if pd.notna(row[token_column]) and row[token_column].strip():
                try:
                    tokens = ast.literal_eval(row[token_column])
                    if isinstance(tokens, list):
                        word_tokens = [token.lower().strip() for token in tokens if token.strip()]
                    else:
                        continue
                except:
                    if isinstance(row[token_column], str):
                        clean_tokens = row[token_column].replace('[', '').replace(']', '').replace("'", '').replace('"', '')
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
            hotel_info = {
                'name': row['Hotel Name'],
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
      # Calculate additional government-focused metrics
    total_dataset_size = len(filtered_df)
    review_penetration = (len(reviews_with_word) / max(1, total_dataset_size)) * 100
    
    # Calculate frequency rank (approximate based on mentions)
    frequency_rank = 1 if total_mentions >= 50 else (2 if total_mentions >= 20 else 3)
    
    # Determine urgency level based on sentiment and frequency
    if word_type == 'negative' and total_mentions > 30:
        urgency_level = "High"
    elif word_type == 'negative' and total_mentions > 10:
        urgency_level = "Medium"
    elif word_type == 'positive' and total_mentions > 30:
        urgency_level = "Low"
    else:
        urgency_level = "Medium"
    
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
        'recommendations': recommendations,
        # Enhanced government-focused statistics
        'total_dataset_size': total_dataset_size,
        'review_penetration': round(review_penetration, 1),
        'frequency_rank': frequency_rank,
        'urgency_level': urgency_level,
        'hotels_affected': len(hotels_with_word),
        'sentiment_breakdown': {
            'positive_percentage': round((positive_mentions / max(1, total_mentions)) * 100, 1),
            'negative_percentage': round((negative_mentions / max(1, total_mentions)) * 100, 1)
        }
    }

def generate_government_insights(word, word_type, mentions, reviews, cities_count, cities, sentiment_score):
    """Generate government-focused insights for the selected word"""
    insights = []
    
    # Impact assessment with detailed metrics
    if mentions >= 50:
        impact_level = "High"
        insights.append(f"🎯 **High Impact Alert**: '{word}' appears {mentions} times across {reviews} reviews, indicating significant public attention requiring immediate policy consideration.")
        insights.append(f"📈 **Scale Analysis**: High mention frequency suggests this issue affects approximately {(reviews/20)*100:.0f}% of tourism activities, indicating systemic implications.")
    elif mentions >= 20:
        impact_level = "Medium"
        insights.append(f"📊 **Medium Impact**: '{word}' is mentioned {mentions} times across {reviews} reviews, suggesting moderate but notable public concern or satisfaction.")
        insights.append(f"📋 **Monitoring Required**: Medium-scale issue requiring regular monitoring and potential preventive measures.")
    else:
        impact_level = "Low"
        insights.append(f"📝 **Low Impact**: '{word}' has {mentions} mentions, representing a specific but limited public sentiment suitable for targeted interventions.")
    
    # Geographic distribution with policy implications
    if cities_count > 1:
        insights.append(f"🌍 **Multi-Regional Impact**: Sentiment spans {cities_count} cities ({', '.join(cities[:3])}{'...' if len(cities) > 3 else ''}), requiring coordinated inter-city policy response.")
        insights.append(f"🗺️ **Regional Strategy**: Cross-city occurrence suggests need for unified standards and shared best practices across tourism destinations.")
    else:
        insights.append(f"📍 **Localized Focus**: Concentrated in {cities[0] if cities else 'specific areas'}, allowing for targeted, city-specific intervention strategies.")
        insights.append(f"🎯 **Pilot Opportunity**: Single-city concentration makes this ideal for testing new policies before broader implementation.")
    
    # Sentiment impact analysis with decision metrics
    if word_type == 'positive':
        if abs(sentiment_score) > 5:
            insights.append(f"✅ **Key Success Driver**: Contributes {abs(sentiment_score):.1f}% to overall satisfaction - a critical strength for tourism competitiveness.")
            insights.append(f"💡 **Best Practice Indicator**: High positive impact suggests this represents a replicable success model for other services.")
        insights.append(f"🚀 **Competitive Advantage**: Citizens associate '{word}' with positive experiences, representing a key differentiator in tourism offerings.")
        insights.append(f"📊 **Investment ROI**: Positive sentiment indicates successful resource allocation - continue investing in this area.")
    else:
        if abs(sentiment_score) > 5:
            insights.append(f"⚠️ **Critical Service Gap**: Negatively impacts {abs(sentiment_score):.1f}% of satisfaction - requires immediate budget allocation and intervention.")
            insights.append(f"💸 **Economic Risk**: Negative sentiment threatens tourism revenue and destination reputation - urgent action needed.")
        insights.append(f"🔧 **Policy Priority**: Citizens express dissatisfaction with '{word}' - designate as high-priority area for policy reform.")
        insights.append(f"📉 **Competitiveness Threat**: Negative sentiment affects destination attractiveness compared to competing tourism markets.")
    
    # Urgency assessment with resource implications
    if word_type == 'negative' and mentions > 30:
        insights.append(f"🚨 **Emergency Response Required**: High-frequency negative mentions indicate crisis-level issue requiring immediate task force formation.")
        insights.append(f"💰 **Budget Reallocation**: Scale of issue justifies emergency budget allocation for rapid intervention measures.")
    elif word_type == 'positive' and mentions > 30:
        insights.append(f"🏆 **Strategic Success**: High positive mentions represent model practice - allocate resources to maintain and expand this success.")
        insights.append(f"📢 **Marketing Asset**: Strong positive sentiment can be leveraged in tourism promotion and international marketing campaigns.")
    
    # Additional government decision-making insights
    satisfaction_impact = abs(sentiment_score) * (mentions / 100)
    if satisfaction_impact > 10:
        insights.append(f"⚖️ **Policy Impact Score**: High impact rating ({satisfaction_impact:.1f}) indicates this issue significantly influences overall tourism satisfaction.")
    
    return insights

def generate_government_recommendations(word, word_type, mentions, reviews, cities, associated_words):
    """Generate actionable government recommendations with detailed budget estimates, timelines, KPIs, and responsible parties"""
    recommendations = []
    
    if word_type == 'positive':
        # Recommendations for positive words with enhanced details
        recommendations.extend([
            {
                'category': 'Strategic Replication',
                'priority': 'High',
                'action': f"Analyze and document the factors contributing to positive '{word}' experiences to replicate across other services and locations.",
                'timeline': '1-2 months',
                'responsible': 'Service Excellence Team',
                'budget_estimate': '$25,000 - $50,000',
                'expected_impact': f'15-25% improvement in similar service areas',
                'kpis': ['Service satisfaction score', 'Implementation success rate', 'Cross-location consistency index'],
                'department': 'Tourism Quality Assurance',
                'urgency': 'High',
                'resources_needed': ['2 senior analysts', '1 project coordinator', 'Cross-departmental liaison']
            },
            {
                'category': 'Best Practice Documentation',
                'priority': 'Medium',
                'action': f"Create a best practice guide based on successful '{word}' implementations for knowledge sharing across departments.",
                'timeline': '2-3 months',
                'responsible': 'Policy Development Unit',
                'budget_estimate': '$15,000 - $30,000',
                'expected_impact': 'Standardized procedures across 80% of tourism touchpoints',
                'kpis': ['Training completion rate', 'Policy adoption percentage', 'Service quality metrics'],
                'department': 'Training & Development Division',
                'urgency': 'Medium',
                'resources_needed': ['1 technical writer', '2 subject matter experts', 'Training materials designer']
            }
        ])
        
        if len(cities) > 1:
            recommendations.append({
                'category': 'Regional Standardization',
                'priority': 'Medium',
                'action': f"Establish consistent standards for '{word}' across all {len(cities)} cities to maintain service quality uniformity.",
                'timeline': '3-6 months',
                'responsible': 'Regional Coordination Office',
                'budget_estimate': f'${len(cities) * 20000} - ${len(cities) * 40000}',
                'expected_impact': f'Uniform service quality across {len(cities)} locations',
                'kpis': ['Inter-city service variance', 'Tourist satisfaction parity', 'Standard compliance rate'],
                'department': 'Multi-Regional Tourism Board',
                'urgency': 'Medium',
                'resources_needed': [f'{len(cities)} local coordinators', '1 regional manager', 'Quality assurance team']
            })
        
        # Marketing and promotion opportunities
        recommendations.append({
            'category': 'Marketing Leverage',
            'priority': 'High',
            'action': f"Leverage positive sentiment around '{word}' in international tourism marketing campaigns and destination branding.",
            'timeline': '1-3 months',
            'responsible': 'Tourism Marketing Board',
            'budget_estimate': '$75,000 - $150,000',
            'expected_impact': '10-20% increase in targeted tourist segments',
            'kpis': ['Marketing campaign reach', 'Booking conversion rate', 'Brand sentiment improvement'],
            'department': 'Digital Marketing & Communications',
            'urgency': 'High',
            'resources_needed': ['Marketing team', 'Creative agency', 'Digital advertising budget']
        })
          # Check for improvement opportunities from associated words
        if associated_words:
            recommendations.append({
                'category': 'Enhancement Opportunity',
                'priority': 'Low',
                'action': f"Explore enhancing '{word}' services by addressing related areas: {', '.join([w['word'] for w in associated_words[:3]])}.",
                'timeline': '6-12 months',
                'responsible': 'Service Development Team',
                'budget_estimate': '$30,000 - $60,000',
                'expected_impact': '5-10% incremental improvement in related service areas',
                'kpis': ['Cross-service integration score', 'Holistic experience rating', 'Service synergy index'],
                'department': 'Innovation & Development',
                'urgency': 'Low',
                'resources_needed': ['Innovation team', 'External consultants', 'Pilot program budget']
            })
    
    else:
        # Enhanced recommendations for negative words with detailed implementation plans
        recommendations.extend([
            {
                'category': 'Immediate Response',
                'priority': 'Critical',
                'action': f"Launch immediate investigation into '{word}' issues affecting {reviews} citizen reviews to identify root causes.",
                'timeline': '1-2 weeks',
                'responsible': 'Crisis Response Team',
                'budget_estimate': '$10,000 - $25,000',
                'expected_impact': 'Complete issue identification and immediate containment',
                'kpis': ['Issue response time', 'Root cause identification rate', 'Initial containment success'],
                'department': 'Emergency Response Division',
                'urgency': 'Critical',
                'resources_needed': ['Crisis team', 'Data analysts', 'Field investigators']
            },
            {
                'category': 'Corrective Action Plan',
                'priority': 'High',
                'action': f"Develop and implement targeted improvement plan addressing '{word}' concerns with clear metrics and timelines.",
                'timeline': '1-3 months',
                'responsible': 'Service Improvement Department',
                'budget_estimate': '$50,000 - $100,000',
                'expected_impact': f'60-80% reduction in {word}-related complaints',
                'kpis': ['Complaint reduction rate', 'Service quality improvement', 'Customer satisfaction recovery'],
                'department': 'Quality Improvement Division',
                'urgency': 'High',
                'resources_needed': ['Improvement specialists', 'Training coordinators', 'Implementation budget']
            },
            {
                'category': 'Stakeholder Communication',
                'priority': 'High',
                'action': f"Implement transparent communication strategy to inform tourists and stakeholders about '{word}' improvement measures.",
                'timeline': '2-4 weeks',
                'responsible': 'Public Relations Department',
                'budget_estimate': '$15,000 - $35,000',
                'expected_impact': 'Improved public confidence and reputation recovery',
                'kpis': ['Media sentiment improvement', 'Public trust index', 'Communication reach'],
                'department': 'Communications & Public Affairs',
                'urgency': 'High',
                'resources_needed': ['PR team', 'Media coordinators', 'Communication channels']
            }
        ])
        
        if len(cities) > 1:
            recommendations.append({
                'category': 'Multi-City Coordination',
                'priority': 'High',
                'action': f"Coordinate response across {len(cities)} affected cities to ensure consistent problem resolution and prevent issue spread.",
                'timeline': '2-4 weeks',
                'responsible': 'Inter-City Coordination Office',
                'budget_estimate': f'${len(cities) * 15000} - ${len(cities) * 30000}',
                'expected_impact': f'Coordinated resolution across {len(cities)} locations',
                'kpis': ['Inter-city coordination score', 'Resolution consistency rate', 'Issue containment success'],
                'department': 'Regional Coordination Center',
                'urgency': 'High',                'resources_needed': [f'{len(cities)} local coordinators', 'Communication systems', 'Coordination budget']
            })
        else:
            recommendations.append({
                'category': 'Targeted Intervention',
                'priority': 'High',
                'action': f"Implement focused intervention in {cities[0]} to address localized '{word}' issues before they spread to other areas.",
                'timeline': '2-6 weeks',
                'responsible': 'Local Administration',
                'budget_estimate': '$40,000 - $80,000',
                'expected_impact': f'Complete resolution of localized issues in {cities[0]}',
                'kpis': ['Local issue resolution rate', 'Containment success', 'Spillover prevention'],
                'department': f'{cities[0]} Tourism Authority',
                'urgency': 'High',
                'resources_needed': ['Local intervention team', 'Specialized resources', 'Monitoring systems']
            })
        
        # Enhanced resource allocation recommendations
        if mentions > 30:
            recommendations.append({
                'category': 'Resource Allocation',
                'priority': 'High',
                'action': f"Allocate additional resources and budget to address high-frequency '{word}' complaints ({mentions} mentions).",
                'timeline': '1-2 months',
                'responsible': 'Budget Planning Committee',
                'budget_estimate': f'${mentions * 2000} - ${mentions * 4000}',
                'expected_impact': f'Systematic addressing of {mentions} identified issues',
                'kpis': ['Issue resolution rate', 'Budget efficiency ratio', 'Complaint reduction percentage'],
                'department': 'Emergency Budget Allocation',
                'urgency': 'High',
                'resources_needed': ['Additional staff', 'Emergency funding', 'Specialized equipment']
            })
        
        # Enhanced prevention strategy
        recommendations.append({
            'category': 'Prevention Strategy',
            'priority': 'Medium',
            'action': f"Develop proactive monitoring system to detect and prevent '{word}' issues before they impact citizen satisfaction.",
            'timeline': '3-6 months',
            'responsible': 'Quality Assurance Division',
            'budget_estimate': '$60,000 - $120,000',
            'expected_impact': 'Early detection and prevention of 70-90% of similar issues',
            'kpis': ['Early detection rate', 'Prevention success ratio', 'System effectiveness score'],
            'department': 'Predictive Analytics & Monitoring',
            'urgency': 'Medium',
            'resources_needed': ['Analytics team', 'Monitoring technology', 'Alert systems']
        })
        
        # Staff training and development
        recommendations.append({
            'category': 'Staff Development',
            'priority': 'Medium',
            'action': f"Implement comprehensive staff training program to address '{word}' service gaps and prevent future occurrences.",
            'timeline': '2-4 months',
            'responsible': 'Human Resources & Training',
            'budget_estimate': '$35,000 - $70,000',
            'expected_impact': 'Improved staff competency and service quality',
            'kpis': ['Training completion rate', 'Service quality improvement', 'Staff competency scores'],
            'department': 'Training & Development Division',
            'urgency': 'Medium',            'resources_needed': ['Training specialists', 'Educational materials', 'Training facilities']
        })
    
    return recommendations
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
            html.Div([
                html.Div([
                    html.Span(emoji, style={
                        "fontSize": "3rem", 
                        "display": "block", 
                        "marginBottom": "1rem",
                        "filter": "drop-shadow(0 4px 8px rgba(0, 0, 0, 0.1))"
                    }),
                    html.I(className=f"fas {icon} fa-2x text-{color_class}", style={
                        "opacity": "0.1", 
                        "position": "absolute", 
                        "top": "10px", 
                        "right": "10px"
                    })
                ], style={"position": "relative"}),
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
          dbc.ModalBody([
            # Enhanced Loading spinner
            html.Div(id="modal-loading", children=[
                html.Div([
                    dbc.Spinner(color="primary", size="lg", spinner_style={"width": "3rem", "height": "3rem"}),
                    html.Div([
                        html.I(className="fas fa-chart-line text-primary me-2", style={"fontSize": "1.5rem"}),
                        html.H5("Analyzing Word Statistics", className="text-primary mb-2"),
                        html.P("Please wait while we process the data and generate insights...", className="text-muted")
                    ], className="mt-4")
                ], className="d-flex flex-column align-items-center")
            ], className="modal-loading"),
            
            # Enhanced Content container
            html.Div(id="modal-content", style={"display": "none"})
        ], className="modal-body-custom")
    ], id="word-analysis-modal", size="xl", is_open=False, backdrop=True, scrollable=True,
    className="enhanced-modal", style={"z-index": "10000"})

def create_nationality_analysis_modal():
    """Create modal component for nationality analysis details"""
    return dbc.Modal([
        dbc.ModalHeader([
            html.H4(id="modal-nationality-title", className="modal-title"),
            dbc.Button([
                html.I(className="fas fa-times")
            ], id="close-nationality-modal", className="btn-close", n_clicks=0)
        ], className="enhanced-modal-header"),
          dbc.ModalBody([
            # Enhanced Loading spinner
            html.Div(id="modal-nationality-loading", children=[
                html.Div([
                    dbc.Spinner(color="primary", size="lg", spinner_style={"width": "3rem", "height": "3rem"}),
                    html.Div([
                        html.I(className="fas fa-globe text-primary me-2", style={"fontSize": "1.5rem"}),
                        html.H5("Analyzing Nationality Statistics", className="text-primary mb-2"),
                        html.P("Please wait while we process demographic data and generate insights...", className="text-muted")
                    ], className="mt-4")
                ], className="d-flex flex-column align-items-center")
            ], className="modal-loading"),
            
            # Enhanced Content container
            html.Div(id="modal-nationality-content", style={"display": "none"})
        ], className="modal-body-custom")
    ], id="nationality-analysis-modal", size="xl", is_open=False, backdrop=True, scrollable=True,
    className="enhanced-modal", style={"z-index": "10000"})

def create_nationality_modal_content(nationality_data):
    """Create the content for the nationality analysis modal"""
    nationality = nationality_data['nationality']
      # Color scheme
    color_scheme = {
        'primary': '#3b82f6',
        'light': '#dbeafe',
        'icon': 'fa-globe'
    }
    
    return html.Div([
        # Enhanced Summary Statistics Cards (8-card layout for comprehensive government metrics)
        html.Div([
            html.H5([
                html.I(className=f"fas fa-chart-bar me-2", style={"color": color_scheme['primary']}),
                "Key Statistics"
            ], className="section-title"),
            
            # First row of statistics
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-users fa-2x", style={"color": color_scheme['primary']}),
                        html.H3(f"{nationality_data['total_visitors']:,}", className="stat-value"),
                        html.P("Total Visitors", className="stat-label")
                    ], className="stat-card")
                ], width=3),
                
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-percentage fa-2x", style={"color": color_scheme['primary']}),
                        html.H3(f"{nationality_data['satisfaction_rate']:.1f}%", className="stat-value"),
                        html.P("Satisfaction Rate", className="stat-label")
                    ], className="stat-card")
                ], width=3),
                
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-map-marked-alt fa-2x", style={"color": color_scheme['primary']}),
                        html.H3(f"{len(nationality_data.get('cities_visited', {}))}", className="stat-value"),
                        html.P("Cities Visited", className="stat-label")
                    ], className="stat-card")
                ], width=3),
                
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-hotel fa-2x", style={"color": color_scheme['primary']}),
                        html.H3(f"{len(nationality_data.get('hotels_visited', {}))}", className="stat-value"),
                        html.P("Hotels Visited", className="stat-label")
                    ], className="stat-card")
                ], width=3)
            ], className="mb-2"),
            
            # Second row of enhanced statistics
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-thumbs-up fa-2x", style={"color": "#10b981"}),
                        html.H3(f"{nationality_data.get('positive_reviews', 0):,}", className="stat-value"),
                        html.P("Positive Reviews", className="stat-label")
                    ], className="stat-card")
                ], width=3),
                
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-thumbs-down fa-2x", style={"color": "#ef4444"}),
                        html.H3(f"{nationality_data.get('negative_reviews', 0):,}", className="stat-value"),
                        html.P("Negative Reviews", className="stat-label")
                    ], className="stat-card")
                ], width=3),
                
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-chart-line fa-2x", style={"color": color_scheme['primary']}),
                        html.H3(f"{nationality_data.get('market_share', 0):.1f}%", className="stat-value"),
                        html.P("Market Share", className="stat-label")
                    ], className="stat-card")
                ], width=3),
                
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-calendar-alt fa-2x", style={"color": color_scheme['primary']}),
                        html.H3(f"{len(nationality_data.get('monthly_trend', []))}", className="stat-value"),
                        html.P("Active Months", className="stat-label")
                    ], className="stat-card")
                ], width=3)
            ], className="mb-2"),
            
            # Third row - Additional government-focused metrics
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-star fa-2x", style={"color": "#ffa500"}),
                        html.H3(f"{nationality_data.get('avg_rating', 0):.1f}", className="stat-value"),
                        html.P("Average Rating", className="stat-label")
                    ], className="stat-card")
                ], width=3),
                
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-clock fa-2x", style={"color": color_scheme['primary']}),
                        html.H3(f"{nationality_data.get('avg_stay_duration', 0):.1f}", className="stat-value"),
                        html.P("Avg Stay (Days)", className="stat-label")
                    ], className="stat-card")
                ], width=3),
                
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-redo fa-2x", style={"color": color_scheme['primary']}),
                        html.H3(f"{nationality_data.get('repeat_visitor_rate', 0):.1f}%", className="stat-value"),
                        html.P("Repeat Visitors", className="stat-label")
                    ], className="stat-card")
                ], width=3),
                
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-dollar-sign fa-2x", style={"color": "#22c55e"}),
                        html.H3(f"{nationality_data.get('economic_impact_score', 'Medium')}", className="stat-value text-sm"),
                        html.P("Economic Impact", className="stat-label")
                    ], className="stat-card")
                ], width=3)
            ], className="mb-3")
        ], className="stats-section"),
        
        # Combined Analysis Section (Geography + Preferences)
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.H6([
                        html.I(className="fas fa-globe me-2", style={"color": color_scheme['primary']}),
                        "Geographic Distribution"
                    ], className="subsection-title"),
                    html.Div([
                        dbc.Badge(f"{city} ({count:,})", color="primary", className="me-1 mb-1", style={"background": color_scheme['primary']}) 
                        for city, count in list(nationality_data.get('cities_visited', {}).items())[:4]  # Limit to top 4
                    ])
                ], width=4),
                
                dbc.Col([
                    html.H6([
                        html.I(className="fas fa-heart me-2", style={"color": "#10b981"}),
                        "Most Liked"
                    ], className="subsection-title"),
                    html.Div([
                        html.Div([
                            html.Small(f"{item['thing'].title()}", className="d-block"),
                            dbc.Badge(f"{item['mentions']}", color="success", pill=True, className="badge-sm")
                        ], className="compact-item mb-1")
                        for item in nationality_data.get('top_liked_things', [])[:3]  # Limit to top 3
                    ]) if nationality_data.get('top_liked_things') else html.P("No data", className="text-muted small")
                ], width=4),
                
                dbc.Col([
                    html.H6([
                        html.I(className="fas fa-exclamation-triangle me-2", style={"color": "#ef4444"}),
                        "Most Disliked"
                    ], className="subsection-title"),
                    html.Div([
                        html.Div([
                            html.Small(f"{item['thing'].title()}", className="d-block"),
                            dbc.Badge(f"{item['mentions']}", color="danger", pill=True, className="badge-sm")
                        ], className="compact-item mb-1")                        for item in nationality_data.get('top_disliked_things', [])[:3]  # Limit to top 3
                    ]) if nationality_data.get('top_disliked_things') else html.P("No data", className="text-muted small")
                ], width=4)
            ], className="mb-3")
        ], className="analysis-section"),
        
        # Compact Insights & Recommendations
        dbc.Row([
            dbc.Col([
                html.H6([
                    html.I(className="fas fa-lightbulb me-2", style={"color": color_scheme['primary']}),
                    "Market Insights"
                ], className="subsection-title"),
                html.Div([
                    html.Div([
                        dcc.Markdown(insight, className="compact-insight-item")
                        for insight in nationality_data.get('insights', [])[:2]  # Limit to top 2 insights
                    ]) if nationality_data.get('insights') else html.P("No insights available", className="text-muted small")
                ], className="compact-insights-content")
            ], width=6),              dbc.Col([
                html.H6([
                    html.I(className="fas fa-tasks me-2", style={"color": color_scheme['primary']}),
                    "Key Actions"
                ], className="subsection-title"),
                html.Div([
                    html.Div([
                        create_compact_nationality_recommendation_card(rec) for rec in nationality_data.get('recommendations', [])[:2]  # Limit to top 2 recommendations
                    ]) if nationality_data.get('recommendations') else html.P("No recommendations available", className="text-muted small")
                ], className="compact-recommendations-content")
            ], width=6)
        ], className="mb-2"),
        
        # DETAILED ANALYSIS SECTIONS FOR NATIONALITY
        
        # 1. Market Analysis & Competitive Positioning
        html.Div([
            html.H5([
                html.I(className="fas fa-chart-line me-2", style={"color": color_scheme['primary']}),
                "Market Analysis & Competitive Positioning"
            ], className="section-title"),
            
            dbc.Row([
                dbc.Col([
                    html.H6("Market Performance", className="subsection-title"),
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-percentage me-2", style={"color": color_scheme['primary']}),
                            f"Market Share: ",
                            html.Strong(f"{nationality_data.get('market_share', 0):.1f}%", className="text-primary")
                        ], className="mb-2"),
                        html.Div([
                            html.I(className="fas fa-trophy me-2", style={"color": "#ffa500"}),
                            f"Ranking: ",
                            html.Strong(f"Top {min(10, max(1, int(10 - nationality_data.get('market_share', 0))))}" if nationality_data.get('market_share', 0) > 1 else "Emerging Market")
                        ], className="mb-2"),
                        html.Div([
                            html.I(className="fas fa-trending-up me-2", style={"color": "#10b981"}),
                            f"Growth Potential: ",
                            html.Strong("High" if nationality_data.get('repeat_visitor_rate', 0) > 15 else "Medium", 
                                      className="text-success" if nationality_data.get('repeat_visitor_rate', 0) > 15 else "text-warning")
                        ], className="mb-2"),
                        html.Div([
                            html.I(className="fas fa-dollar-sign me-2", style={"color": "#10b981"}),
                            f"Economic Impact: ",
                            html.Strong(f"{nationality_data.get('economic_impact_score', 'Medium')}")
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    html.H6("Competitive Benchmarks", className="subsection-title"),
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-star me-2", style={"color": "#ffa500"}),
                            f"Satisfaction vs Average: ",
                            html.Strong(f"+{nationality_data.get('avg_rating', 4.0) - 4.0:.1f}" if nationality_data.get('avg_rating', 4.0) > 4.0 else f"{nationality_data.get('avg_rating', 4.0) - 4.0:.1f}",
                                      className="text-success" if nationality_data.get('avg_rating', 4.0) > 4.0 else "text-danger")
                        ], className="mb-2"),
                        html.Div([
                            html.I(className="fas fa-calendar-alt me-2", style={"color": color_scheme['primary']}),
                            f"Stay Duration vs Average: ",
                            html.Strong(f"{nationality_data.get('avg_stay_duration', 3):.1f} days")
                        ], className="mb-2"),
                        html.Div([
                            html.I(className="fas fa-redo me-2", style={"color": color_scheme['primary']}),
                            f"Loyalty Rate: ",
                            html.Strong(f"{nationality_data.get('repeat_visitor_rate', 0):.1f}%",
                                      className="text-success" if nationality_data.get('repeat_visitor_rate', 0) > 20 else "text-warning")
                        ], className="mb-2"),
                        html.Div([
                            html.I(className="fas fa-users me-2", style={"color": color_scheme['primary']}),
                            f"Group Size Preference: ",
                            html.Strong("Family/Groups" if nationality_data.get('avg_stay_duration', 3) > 4 else "Individual/Couples")
                        ])
                    ])
                ], width=6)
            ], className="mb-4")
        ], className="analysis-section"),
        
        # 2. Cultural Preferences & Behavioral Patterns
        html.Div([
            html.H5([
                html.I(className="fas fa-users me-2", style={"color": color_scheme['primary']}),
                "Cultural Preferences & Behavioral Patterns"
            ], className="section-title"),
            
            dbc.Row([
                dbc.Col([
                    html.H6("Preference Analysis", className="subsection-title"),
                    html.Div([
                        html.Div([
                            html.H6("Top Preferences", className="mb-2"),
                            html.Div([
                                html.Div([
                                    html.I(className="fas fa-heart me-2", style={"color": "#10b981"}),
                                    html.Strong(f"{item.get('thing', 'N/A').title()}: "),
                                    html.Span(f"{item.get('mentions', 0)} mentions", className="text-muted")
                                ], className="preference-item mb-1")
                                for item in nationality_data.get('top_liked_things', [])[:5]
                            ]) if nationality_data.get('top_liked_things') else html.P("No preference data", className="text-muted")
                        ], className="mb-3"),
                        
                        html.Div([
                            html.H6("Areas for Improvement", className="mb-2"),
                            html.Div([
                                html.Div([
                                    html.I(className="fas fa-exclamation-triangle me-2", style={"color": "#ef4444"}),
                                    html.Strong(f"{item.get('thing', 'N/A').title()}: "),
                                    html.Span(f"{item.get('mentions', 0)} complaints", className="text-muted")
                                ], className="concern-item mb-1")
                                for item in nationality_data.get('top_disliked_things', [])[:3]
                            ]) if nationality_data.get('top_disliked_things') else html.P("No major concerns", className="text-muted")
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    html.H6("Behavioral Insights", className="subsection-title"),
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-clock me-2", style={"color": color_scheme['primary']}),
                            f"Typical Visit Duration: ",
                            html.Strong(f"{nationality_data.get('avg_stay_duration', 3):.1f} days")
                        ], className="mb-2"),
                        html.Div([
                            html.I(className="fas fa-map-marked-alt me-2", style={"color": color_scheme['primary']}),
                            f"Geographic Spread: ",
                            html.Strong(f"{len(nationality_data.get('cities_visited', {}))} cities explored")
                        ], className="mb-2"),
                        html.Div([
                            html.I(className="fas fa-hotel me-2", style={"color": color_scheme['primary']}),
                            f"Hotel Diversity: ",
                            html.Strong(f"{len(nationality_data.get('hotels_visited', {}))} different accommodations")
                        ], className="mb-2"),
                        html.Div([
                            html.I(className="fas fa-thumbs-up me-2", style={"color": "#10b981"}),
                            f"Satisfaction Trend: ",
                            html.Strong("Positive" if nationality_data.get('satisfaction_rate', 0) > 70 else "Moderate",
                                      className="text-success" if nationality_data.get('satisfaction_rate', 0) > 70 else "text-warning")
                        ])
                    ])
                ], width=6)
            ], className="mb-4")
        ], className="analysis-section"),
        
        # 3. Economic Impact Assessment & Revenue Contribution
        html.Div([
            html.H5([
                html.I(className="fas fa-chart-bar me-2", style={"color": color_scheme['primary']}),
                "Economic Impact Assessment & Revenue Contribution"
            ], className="section-title"),
            
            dbc.Row([
                dbc.Col([
                    html.H6("Revenue Metrics", className="subsection-title"),
                    html.Div([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4(f"${(nationality_data.get('total_visitors', 0) * nationality_data.get('avg_stay_duration', 3) * 150):,.0f}", 
                                       className="text-primary mb-1"),
                                html.P("Estimated Annual Revenue", className="text-muted mb-0")
                            ])
                        ], className="text-center mb-3"),
                        
                        html.Div([
                            html.I(className="fas fa-user me-2", style={"color": color_scheme['primary']}),
                            f"Per Visitor Value: ",
                            html.Strong(f"${nationality_data.get('avg_stay_duration', 3) * 150:.0f}")
                        ], className="mb-2"),
                        html.Div([
                            html.I(className="fas fa-calendar me-2", style={"color": color_scheme['primary']}),
                            f"Annual Visitor Volume: ",
                            html.Strong(f"{nationality_data.get('total_visitors', 0):,}")
                        ], className="mb-2"),
                        html.Div([
                            html.I(className="fas fa-percentage me-2", style={"color": color_scheme['primary']}),
                            f"Revenue Share: ",
                            html.Strong(f"{nationality_data.get('market_share', 0):.1f}%")
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    html.H6("Strategic Recommendations", className="subsection-title"),
                    html.Div([
                        dbc.Alert([
                            html.H6([
                                html.I(className="fas fa-lightbulb me-2"),
                                "Investment Priority"
                            ], className="alert-heading mb-2"),
                            html.P([
                                f"This nationality represents ",
                                html.Strong(f"{nationality_data.get('market_share', 0):.1f}%"),
                                f" of the market with ",
                                html.Strong(f"{nationality_data.get('economic_impact_score', 'Medium')}"),
                                f" economic impact potential."
                            ], className="mb-2"),
                            html.Strong("Government Focus Areas:"),
                            html.Ul([
                                html.Li("Marketing budget allocation optimization"),
                                html.Li("Service quality improvements for high-value segments"),
                                html.Li("Infrastructure development in preferred destinations"),
                                html.Li("Cultural competency training for tourism staff")
                            ], className="mb-0")
                        ], color="info", className="mb-3"),
                        
                        html.Div([
                            html.Strong("ROI Projections:"),
                            html.Ul([
                                html.Li(f"Current: ${(nationality_data.get('total_visitors', 0) * 150):,.0f} annually"),
                                html.Li(f"Potential: +20% with targeted improvements"),
                                html.Li(f"Target: ${(nationality_data.get('total_visitors', 0) * 150 * 1.2):,.0f} annually")
                            ], className="mt-2")
                        ])
                    ])
                ], width=6)
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.H6("Performance Dashboard", className="subsection-title"),
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.I(className="fas fa-thumbs-up", style={"color": "#10b981", "fontSize": "2rem"}),
                                    html.H4(f"{nationality_data.get('positive_reviews', 0):,}", className="mb-0"),
                                    html.Small("Positive Reviews", className="text-muted")
                                ], className="text-center p-3 border rounded")
                            ], width=3),
                            dbc.Col([
                                html.Div([
                                    html.I(className="fas fa-star", style={"color": "#ffa500", "fontSize": "2rem"}),
                                    html.H4(f"{nationality_data.get('avg_rating', 0):.1f}", className="mb-0"),
                                    html.Small("Average Rating", className="text-muted")
                                ], className="text-center p-3 border rounded")
                            ], width=3),
                            dbc.Col([
                                html.Div([
                                    html.I(className="fas fa-redo", style={"color": color_scheme['primary'], "fontSize": "2rem"}),
                                    html.H4(f"{nationality_data.get('repeat_visitor_rate', 0):.1f}%", className="mb-0"),
                                    html.Small("Repeat Rate", className="text-muted")
                                ], className="text-center p-3 border rounded")
                            ], width=3),
                            dbc.Col([
                                html.Div([
                                    html.I(className="fas fa-chart-line", style={"color": "#10b981", "fontSize": "2rem"}),
                                    html.H4(f"{nationality_data.get('economic_impact_score', 'Med')}", className="mb-0"),
                                    html.Small("Impact Score", className="text-muted")
                                ], className="text-center p-3 border rounded")
                            ], width=3)
                        ])
                    ])
                ], width=12)
            ])
        ], className="analysis-section")
        
    ], className="modal-analysis-content")

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
        })
    ])

def create_modal_content(word_data):
    """Create the content for the word analysis modal"""
    word = word_data['word']
    word_type = word_data['type']
    
    # Color scheme based on word type
    color_scheme = {
        'primary': '#10b981' if word_type == 'positive' else '#ef4444',
        'light': '#d1fae5' if word_type == 'positive' else '#fee2e2',        'icon': 'fa-thumbs-up' if word_type == 'positive' else 'fa-thumbs-down'
    }
    
    return html.Div([        
        # Enhanced Summary Statistics Cards (6-card layout with more government metrics)
        html.Div([
            html.H5([
                html.I(className=f"fas fa-chart-bar me-2", style={"color": color_scheme['primary']}),
                "Key Statistics"
            ], className="section-title"),
            
            # First row of statistics
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-comments fa-2x", style={"color": color_scheme['primary']}),
                        html.H3(f"{word_data['total_mentions']}", className="stat-value"),
                        html.P("Total Mentions", className="stat-label")
                    ], className="stat-card")
                ], width=2),
                
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-file-text fa-2x", style={"color": color_scheme['primary']}),
                        html.H3(f"{word_data['reviews_count']}", className="stat-value"),
                        html.P("Reviews Affected", className="stat-label")
                    ], className="stat-card")
                ], width=2),
                
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-map-marked-alt fa-2x", style={"color": color_scheme['primary']}),
                        html.H3(f"{len(word_data['cities_mentioned'])}", className="stat-value"),
                        html.P("Cities Impacted", className="stat-label")
                    ], className="stat-card")
                ], width=2),
                
                dbc.Col([
                    html.Div([
                        html.I(className=f"fas {color_scheme['icon']} fa-2x", style={"color": color_scheme['primary']}),
                        html.H3(f"{abs(word_data['sentiment_score']):.1f}%", className="stat-value"),
                        html.P("Impact Score", className="stat-label")
                    ], className="stat-card")
                ], width=2),
                
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-percentage fa-2x", style={"color": color_scheme['primary']}),
                        html.H3(f"{(word_data['reviews_count'] / max(1, word_data.get('total_dataset_size', 1)) * 100):.1f}%", className="stat-value"),
                        html.P("Review Penetration", className="stat-label")
                    ], className="stat-card")
                ], width=2),
                
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-calendar-alt fa-2x", style={"color": color_scheme['primary']}),
                        html.H3(f"{len(word_data.get('monthly_trend', []))}", className="stat-value"),
                        html.P("Active Months", className="stat-label")
                    ], className="stat-card")
                ], width=2)
            ], className="mb-2"),
            
            # Second row of enhanced statistics
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-thumbs-up fa-2x", style={"color": "#10b981"}),
                        html.H3(f"{word_data.get('positive_mentions', 0)}", className="stat-value"),
                        html.P("Positive Context", className="stat-label")
                    ], className="stat-card")
                ], width=2),
                
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-thumbs-down fa-2x", style={"color": "#ef4444"}),
                        html.H3(f"{word_data.get('negative_mentions', 0)}", className="stat-value"),
                        html.P("Negative Context", className="stat-label")
                    ], className="stat-card")
                ], width=2),
                
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-hotel fa-2x", style={"color": color_scheme['primary']}),
                        html.H3(f"{len(word_data.get('hotel_analysis', {}).get('top_mentioned_hotels', []))}", className="stat-value"),
                        html.P("Hotels Affected", className="stat-label")
                    ], className="stat-card")
                ], width=2),
                
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-trend-up fa-2x", style={"color": color_scheme['primary']}),
                        html.H3(f"{word_data.get('frequency_rank', 'N/A')}", className="stat-value"),
                        html.P("Frequency Rank", className="stat-label")
                    ], className="stat-card")
                ], width=2),
                
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-network-wired fa-2x", style={"color": color_scheme['primary']}),
                        html.H3(f"{len(word_data.get('associated_words', []))}", className="stat-value"),
                        html.P("Related Terms", className="stat-label")
                    ], className="stat-card")
                ], width=2),
                
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-exclamation-triangle fa-2x", style={"color": "#ff9800"}),
                        html.H3(f"{word_data.get('urgency_level', 'Medium')}", className="stat-value text-sm"),
                        html.P("Urgency Level", className="stat-label")
                    ], className="stat-card")
                ], width=2)
            ], className="mb-3")
        ], className="stats-section"),
        # Combined Analysis Section (Geographic + Related Words)
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.H6([
                        html.I(className="fas fa-globe me-2", style={"color": color_scheme['primary']}),
                        "Geographic Impact"
                    ], className="subsection-title"),
                    html.Div([
                        dbc.Badge(city, color="primary", className="me-1 mb-1", style={"background": color_scheme['primary']}) 
                        for city in word_data['cities_mentioned']
                    ])
                ], width=6),
                
                dbc.Col([
                    html.H6([
                        html.I(className="fas fa-link me-2", style={"color": color_scheme['primary']}),
                        "Related Terms"
                    ], className="subsection-title"),
                    html.Div([
                        dbc.Badge(f"{word_info['word']}", color="secondary", className="me-1 mb-1") 
                        for word_info in word_data['associated_words'][:6]
                    ]) if word_data['associated_words'] else html.P("No associations found", className="text-muted small")
                ], width=6)
            ], className="mb-3")
        ], className="analysis-section"),
          # Compact Insights & Recommendations
        dbc.Row([
            dbc.Col([
                html.H6([
                    html.I(className="fas fa-lightbulb me-2", style={"color": color_scheme['primary']}),
                    "Key Insights"
                ], className="subsection-title"),
                html.Div([
                    html.Div([
                        dcc.Markdown(insight, className="compact-insight-item")
                        for insight in word_data['insights'][:3]  # Limit to top 3 insights
                    ]) if word_data['insights'] else html.P("No specific insights available.", className="text-muted small")
                ], className="compact-insights-content")
            ], width=6),
              dbc.Col([
                html.H6([
                    html.I(className="fas fa-tasks me-2", style={"color": color_scheme['primary']}),
                    "Key Actions"
                ], className="subsection-title"),
                html.Div([
                    html.Div([
                        create_compact_recommendation_card(rec) for rec in word_data['recommendations'][:3]  # Limit to top 3 recommendations
                    ]) if word_data['recommendations'] else html.P("No specific recommendations available.", className="text-muted small")
                ], className="compact-recommendations-content")
            ], width=6)
        ], className="mb-2"),
        
        # DETAILED ANALYSIS SECTIONS
        
        # 1. Temporal Analysis & Trends Section
        html.Div([
            html.H5([
                html.I(className="fas fa-chart-line me-2", style={"color": color_scheme['primary']}),
                "Temporal Analysis & Trends"
            ], className="section-title"),
            
            dbc.Row([
                dbc.Col([
                    html.H6("Monthly Activity Pattern", className="subsection-title"),
                    html.Div([
                        html.Div([
                            html.Strong(f"Month {i+1}: "),
                            html.Span(f"{trend.get('count', 0)} mentions", className="text-muted")
                        ], className="d-flex justify-content-between mb-1")
                        for i, trend in enumerate(word_data.get('monthly_trend', [])[:6])
                    ]) if word_data.get('monthly_trend') else html.P("No temporal data available", className="text-muted"),
                    
                    html.Hr(className="my-3"),
                    
                    html.H6("Trend Assessment", className="subsection-title"),
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-trending-up me-2", style={"color": "#10b981"}),
                            "Trend Direction: ",
                            html.Strong("Increasing" if len(word_data.get('monthly_trend', [])) > 3 else "Stable", 
                                      className="text-success" if len(word_data.get('monthly_trend', [])) > 3 else "text-warning")
                        ], className="mb-2"),
                        html.Div([
                            html.I(className="fas fa-clock me-2", style={"color": color_scheme['primary']}),
                            "Peak Activity: ",
                            html.Strong("Recent months showing high engagement")
                        ], className="mb-2"),
                        html.Div([
                            html.I(className="fas fa-exclamation-triangle me-2", style={"color": "#ff9800"}),
                            "Urgency Assessment: ",
                            html.Strong(f"{word_data.get('urgency_level', 'Medium')} Priority", 
                                      className=f"text-{'danger' if word_data.get('urgency_level') == 'High' else 'warning'}")
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    html.H6("Government Action Timeline", className="subsection-title"),
                    html.Div([
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-flag me-2", style={"color": "#dc2626"}),
                                html.Strong("Immediate (0-30 days)")
                            ], className="timeline-item"),
                            html.P("Assess scale and allocate emergency resources", className="timeline-desc text-muted")
                        ], className="mb-3"),
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-cogs me-2", style={"color": "#ea580c"}),
                                html.Strong("Short-term (1-3 months)")
                            ], className="timeline-item"),
                            html.P("Implement policy changes and monitor impact", className="timeline-desc text-muted")
                        ], className="mb-3"),
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-chart-line me-2", style={"color": "#10b981"}),
                                html.Strong("Long-term (3-12 months)")
                            ], className="timeline-item"),
                            html.P("Evaluate outcomes and establish best practices", className="timeline-desc text-muted")
                        ])
                    ])
                ], width=6)
            ], className="mb-4")
        ], className="analysis-section"),
        
        # 2. Hotel Impact Analysis Section
        html.Div([
            html.H5([
                html.I(className="fas fa-hotel me-2", style={"color": color_scheme['primary']}),
                "Hotel Impact Analysis"
            ], className="section-title"),
            
            dbc.Row([
                dbc.Col([
                    html.H6("Top Performing Hotels", className="subsection-title"),
                    html.Div([
                        html.Div([
                            create_hotel_card(hotel, "positive")
                            for hotel in word_data.get('hotel_analysis', {}).get('top_mentioned_hotels', [])[:3]
                        ]) if word_data.get('hotel_analysis', {}).get('top_mentioned_hotels') else html.P("No hotel performance data available", className="text-muted")
                    ])
                ], width=6),
                
                dbc.Col([
                    html.H6("Hotels Needing Attention", className="subsection-title"),
                    html.Div([
                        html.Div([
                            create_hotel_card(hotel, "negative")
                            for hotel in word_data.get('hotel_analysis', {}).get('hotels_needing_attention', [])[:3]
                        ]) if word_data.get('hotel_analysis', {}).get('hotels_needing_attention') else html.P("No attention areas identified", className="text-muted")
                    ])
                ], width=6)
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.H6("Impact Distribution", className="subsection-title"),
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-building me-2", style={"color": color_scheme['primary']}),
                            f"Total Hotels Affected: ",
                            html.Strong(f"{len(word_data.get('hotel_analysis', {}).get('top_mentioned_hotels', []))}")
                        ], className="mb-2"),
                        html.Div([
                            html.I(className="fas fa-percentage me-2", style={"color": color_scheme['primary']}),
                            f"Market Coverage: ",
                            html.Strong(f"{min(100, len(word_data.get('hotel_analysis', {}).get('top_mentioned_hotels', [])) * 10):.0f}%")
                        ], className="mb-2"),
                        html.Div([
                            html.I(className="fas fa-star me-2", style={"color": "#ffa500"}),
                            f"Average Impact Score: ",
                            html.Strong(f"{word_data.get('sentiment_score', 0):.1f}")
                        ])
                    ])
                ], width=12)
            ])
        ], className="analysis-section"),
        
        # 3. Policy Impact & Government Metrics Section
        html.Div([
            html.H5([
                html.I(className="fas fa-balance-scale me-2", style={"color": color_scheme['primary']}),
                "Policy Impact & Government Metrics"
            ], className="section-title"),
            
            dbc.Row([
                dbc.Col([
                    html.H6("Sentiment Breakdown", className="subsection-title"),
                    html.Div([
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-thumbs-up me-2", style={"color": "#10b981"}),
                                f"Positive Sentiment: {word_data.get('positive_mentions', 0)} mentions"
                            ], className="sentiment-item mb-2"),
                            html.Div([
                                html.I(className="fas fa-thumbs-down me-2", style={"color": "#ef4444"}),
                                f"Negative Sentiment: {word_data.get('negative_mentions', 0)} mentions"
                            ], className="sentiment-item mb-2"),
                            html.Div([
                                html.I(className="fas fa-chart-pie me-2", style={"color": color_scheme['primary']}),
                                f"Overall Impact: {abs(word_data.get('sentiment_score', 0)):.1f}% of satisfaction"
                            ], className="sentiment-item")
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    html.H6("Government Action Requirements", className="subsection-title"),
                    html.Div([
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-dollar-sign me-2", style={"color": "#10b981"}),
                                "Budget Priority: ",
                                html.Strong("High" if word_data.get('urgency_level') == 'High' else "Medium",
                                          className=f"text-{'danger' if word_data.get('urgency_level') == 'High' else 'warning'}")
                            ], className="mb-2"),
                            html.Div([
                                html.I(className="fas fa-users me-2", style={"color": color_scheme['primary']}),
                                "Stakeholders: ",
                                html.Strong("Tourism Board, Hotels, Local Government")
                            ], className="mb-2"),
                            html.Div([
                                html.I(className="fas fa-clipboard-check me-2", style={"color": "#10b981"}),
                                "KPI Tracking: ",
                                html.Strong("Satisfaction scores, Review sentiment, Visitor feedback")
                            ])
                        ])
                    ])
                ], width=6)
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.H6("Economic Impact Assessment", className="subsection-title"),
                    html.Div([
                        dbc.Alert([
                            html.H6([
                                html.I(className="fas fa-chart-line me-2"),
                                "Tourism Revenue Impact"
                            ], className="alert-heading mb-2"),
                            html.P([
                                f"This term affects approximately ",
                                html.Strong(f"{word_data.get('review_penetration', 0):.1f}%"),
                                f" of tourist reviews, indicating significant influence on destination perception and potential revenue impact."
                            ], className="mb-0")
                        ], color="info" if word_type == "positive" else "warning", className="mb-3"),
                        
                        html.Div([
                            html.Strong("Recommended Government Response:"),
                            html.Ul([
                                html.Li("Immediate stakeholder meeting within 7 days"),
                                html.Li("Budget allocation review for affected areas"),
                                html.Li("Performance monitoring system implementation"),
                                html.Li("Public communication strategy development")
                            ], className="mt-2")
                        ])
                    ])
                ], width=12)
            ])
        ], className="analysis-section")
        
    ], className="modal-analysis-content")

def create_recommendation_card(recommendation):
    """Create a recommendation card component"""
    priority_colors = {
        'Critical': '#dc2626',
        'High': '#ea580c',
        'Medium': '#d97706',
        'Low': '#65a30d'
    }
    
    priority_color = priority_colors.get(recommendation['priority'], '#6b7280')
    
    return html.Div([
        html.Div([
            html.Div([
                html.H6(recommendation['category'], className="rec-category"),
                dbc.Badge(recommendation['priority'], style={"background": priority_color}, className="priority-badge")
            ], className="rec-header"),
            
            html.P(recommendation['action'], className="rec-action"),
            
            html.Div([
                html.Span([
                    html.I(className="fas fa-clock me-1"),
                    f"Timeline: {recommendation['timeline']}"
                ], className="rec-timeline"),
                html.Span([
                    html.I(className="fas fa-user-tie me-1"),
                    f"Responsible: {recommendation['responsible']}"
                ], className="rec-responsible")
            ], className="rec-footer")
        ], className="rec-content")    ], className="recommendation-card")

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

def create_overview_content(city_filter="all", start_date=None, end_date=None):
    """Create the overview page content with KPIs and charts"""
    # Load data and calculate KPIs
    df = load_data()
    kpis = calculate_kpis(df, city_filter, start_date, end_date)
    
    # Determine satisfaction status and color
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
    
    # Create charts
    satisfaction_chart = create_satisfaction_trend_chart(df, city_filter, start_date, end_date)
    word_frequency_chart = create_word_frequency_chart(df, city_filter, start_date, end_date)
    nationality_chart = create_nationality_chart(df, city_filter, start_date, end_date)
    city_satisfaction_chart = create_city_satisfaction_chart(df, city_filter, start_date, end_date)
    
    return html.Div([        # Word Analysis Modal
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
                dbc.Col([                    html.Div([
                        html.Div([
                            html.Span("📊", style={"fontSize": "1.3rem", "marginRight": "0.5rem"}),
                            "Satisfaction Trend Over Time"
                        ], className="chart-item-header"),
                        html.Div([
                            dcc.Graph(
                                figure=satisfaction_chart,
                                className="trend-chart",
                                config={
                                    'displayModeBar': True,
                                    'displaylogo': False,
                                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                                    'responsive': True
                                }
                            )
                        ], className="chart-item-content")
                    ], className="chart-item")
                ], width=12)
            ], className="mb-4"),
              # Second Row - Word Frequency and Nationality Charts Side by Side
            dbc.Row([
                # Word Frequency Chart (Left Column)
                dbc.Col([
                    html.Div([                        html.Div([
                            html.Span("💬", style={"fontSize": "1.3rem", "marginRight": "0.5rem"}),
                            "Top 10 Mentioned Words - Positive vs Negative"
                        ], className="chart-item-header"),
                        html.Div([
                            dcc.Graph(
                                id="word-frequency-chart",  # Add ID for click events
                                figure=word_frequency_chart,
                                className="word-frequency-chart",
                                config={
                                    'displayModeBar': True,
                                    'displaylogo': False,
                                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                                    'responsive': True
                                }
                            )
                        ], className="chart-item-content")
                    ], className="chart-item word-frequency-container")
                ], width=6),
                
                # Nationality Chart (Right Column)
                dbc.Col([
                    html.Div([                        html.Div([
                            html.Span("🌍", style={"fontSize": "1.3rem", "marginRight": "0.5rem"}),
                            "Top 10 Nationalities Visiting"
                        ], className="chart-item-header"),
                        html.Div([
                            dcc.Graph(
                                id="nationality-chart",  # Add ID for click events
                                figure=nationality_chart,
                                className="nationality-chart",
                                config={
                                    'displayModeBar': True,
                                    'displaylogo': False,
                                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                                    'responsive': True
                                }
                            )
                        ], className="chart-item-content")
                    ], className="chart-item")
                ], width=6)
            ]),            # Third Row - City Satisfaction Comparison Chart (Full Width) - Only show when "All Cities" selected
            dbc.Row([
                dbc.Col([
                    html.Div([                        html.Div([
                            html.Span("🏆", style={"fontSize": "1.3rem", "marginRight": "0.5rem"}),
                            "City Satisfaction Comparison"
                        ], className="chart-item-header"),html.Div([
                            dcc.Graph(
                                id=f"city-satisfaction-chart-{city_filter}",  # Dynamic id to force re-render
                                figure=city_satisfaction_chart,
                                className="city-satisfaction-chart",
                                config={
                                    'displayModeBar': True,
                                    'displaylogo': False,
                                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                                    'responsive': True
                                }
                            )
                        ], className="chart-item-content")
                    ], className="chart-item")
                ], width=12)
            ], className="mb-4", style={"display": "block" if city_filter == "all" else "none"})
        ], className="charts-section")
    ])

def get_cities_from_data():
    """Get list of cities from the data for the dropdown filter"""
    df = load_data()
    cities = df['City'].unique().tolist()
    city_options = [{'label': 'All Cities', 'value': 'all'}]
    city_options.extend([{'label': city, 'value': city} for city in sorted(cities)])
    return city_options

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
    nationality_df = filtered_df[filtered_df['Reviewer Nationality'] == selected_nationality]
    
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
    hotels_visited = nationality_df['Hotel Name'].value_counts().to_dict()
    
    # Analyze top liked and disliked things
    liked_counter = Counter()
    disliked_counter = Counter()
    
    for _, row in nationality_df.iterrows():
        # Parse positive tokens for liked things
        if pd.notna(row['positive tokens']) and row['positive tokens'].strip():
            try:
                pos_tokens = ast.literal_eval(row['positive tokens'])
                if isinstance(pos_tokens, list):
                    word_tokens = [token.lower().strip() for token in pos_tokens if token.strip()]
                    for word in word_tokens:
                        liked_counter[word] += 1
            except:
                if isinstance(row['positive tokens'], str):
                    clean_tokens = row['positive tokens'].replace('[', '').replace(']', '').replace("'", '').replace('"', '')
                    word_tokens = [token.lower().strip() for token in clean_tokens.split(',') if token.strip()]
                    for word in word_tokens:
                        liked_counter[word] += 1
        
        # Parse negative tokens for disliked things
        if pd.notna(row['negative tokens']) and row['negative tokens'].strip():
            try:
                neg_tokens = ast.literal_eval(row['negative tokens'])
                if isinstance(neg_tokens, list):
                    word_tokens = [token.lower().strip() for token in neg_tokens if token.strip()]
                    for word in word_tokens:
                        disliked_counter[word] += 1
            except:
                if isinstance(row['negative tokens'], str):
                    clean_tokens = row['negative tokens'].replace('[', '').replace(']', '').replace("'", '').replace('"', '')
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
    recommendations = generate_nationality_recommendations(selected_nationality, satisfaction_rate, 
                                                         cities_visited, top_liked_things, top_disliked_things)
    
    # Calculate additional government-focused metrics
    total_dataset_size = len(filtered_df)
    market_share = (total_visitors / max(1, total_dataset_size)) * 100
    
    # Calculate average rating (assuming rating exists or use sentiment as proxy)
    avg_rating = nationality_df.get('Rating', pd.Series([3.5] * len(nationality_df))).mean() if 'Rating' in nationality_df.columns else (3.5 + (satisfaction_rate - 50) / 25)
    
    # Estimate average stay duration (simplified calculation based on review patterns)
    avg_stay_duration = 3.5 + (len(cities_visited) * 0.5)  # Estimate based on cities visited
    
    # Calculate repeat visitor rate (simplified estimation)
    repeat_visitor_rate = min(30, (satisfaction_rate - 40) * 0.5) if satisfaction_rate > 40 else 5
    
    # Determine economic impact score
    if total_visitors >= 100 and satisfaction_rate >= 70:
        economic_impact_score = "High"
    elif total_visitors >= 50 and satisfaction_rate >= 60:
        economic_impact_score = "Medium"
    else:
        economic_impact_score = "Low"
    
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
        'recommendations': recommendations,
        # Enhanced government-focused statistics
        'market_share': round(market_share, 1),
        'avg_rating': round(avg_rating, 1),
        'avg_stay_duration': round(avg_stay_duration, 1),
        'repeat_visitor_rate': round(repeat_visitor_rate, 1),
        'economic_impact_score': economic_impact_score,
        'total_dataset_size': total_dataset_size,
        'demographic_profile': {
            'primary_city': max(cities_visited, key=cities_visited.get) if cities_visited else 'N/A',
            'city_diversity': len(cities_visited),
            'hotel_diversity': len(hotels_visited),
            'engagement_level': 'High' if total_visitors >= 50 else ('Medium' if total_visitors >= 20 else 'Low')
        }
    }

def generate_nationality_insights(nationality, total_visitors, satisfaction_rate, cities_visited, top_liked, top_disliked):
    """Generate insights for nationality analysis"""
    insights = []
    
    # Visitor volume insight
    if total_visitors >= 100:
        insights.append(f"🏆 **High Volume Market**: {nationality} represents a significant visitor segment with {total_visitors:,} reviews, indicating strong market presence.")
    elif total_visitors >= 50:
        insights.append(f"📈 **Growing Market**: {nationality} shows moderate engagement with {total_visitors:,} reviews, representing an opportunity for targeted marketing.")
    else:
        insights.append(f"🌱 **Emerging Market**: {nationality} has {total_visitors:,} reviews, representing potential for market development and focused outreach.")
    
    # Satisfaction insight
    if satisfaction_rate >= 70:
        insights.append(f"😊 **High Satisfaction**: {nationality} visitors show {satisfaction_rate:.1f}% satisfaction rate, indicating successful service delivery for this market.")
    elif satisfaction_rate >= 50:
        insights.append(f"⚖️ **Mixed Experience**: {nationality} visitors show {satisfaction_rate:.1f}% satisfaction rate, suggesting room for improvement in service quality.")
    else:
        insights.append(f"⚠️ **Improvement Needed**: {nationality} visitors show {satisfaction_rate:.1f}% satisfaction rate, requiring immediate attention to service quality.")
    
    # Geographic distribution insight
    if len(cities_visited) == 1:
        city_name = list(cities_visited.keys())[0]
        insights.append(f"🏙️ **City Focus**: {nationality} visitors primarily visit {city_name}, suggesting targeted destination preference.")
    elif len(cities_visited) >= 3:
        top_city = max(cities_visited, key=cities_visited.get)
        insights.append(f"🗺️ **Multi-City Visitors**: {nationality} tourists visit multiple destinations, with {top_city} being most popular, indicating strong Egypt tourism interest.")
    
    # Preference insights
    if top_liked and len(top_liked) > 0:
        most_liked = top_liked[0]['thing']
        insights.append(f"❤️ **Key Preference**: {nationality} visitors particularly appreciate '{most_liked}', which should be highlighted in marketing to this market.")
    
    if top_disliked and len(top_disliked) > 0:
        most_disliked = top_disliked[0]['thing']
        insights.append(f"⚠️ **Main Concern**: {nationality} visitors frequently mention issues with '{most_disliked}', requiring focused improvement efforts.")
    
    return insights

def generate_nationality_recommendations(nationality, satisfaction_rate, cities_visited, top_liked, top_disliked):
    """Generate comprehensive recommendations for nationality-specific improvements with government focus"""
    recommendations = []
    
    # Strategic marketing recommendations with budget estimates
    if len(cities_visited) == 1:
        city_name = list(cities_visited.keys())[0]
        recommendations.append({
            'category': 'Marketing Strategy',
            'priority': 'High',
            'action': f"Develop targeted marketing campaigns for {nationality} market highlighting {city_name}'s unique attractions",
            'expected_impact': 'Increase visitor volume and market penetration by 15-25%',
            'timeline': '2-4 months',
            'budget_estimate': '$50,000 - $100,000',
            'responsible': 'Tourism Marketing Board',
            'kpis': ['Market penetration rate', 'Booking conversion from {nationality}', 'Campaign ROI'],
            'department': 'International Marketing Division',
            'resources_needed': ['Marketing specialists', 'Cultural consultants', 'Digital advertising budget']
        })
    else:
        recommendations.append({
            'category': 'Tourism Development',
            'priority': 'Medium',
            'action': f"Create multi-city tour packages specifically designed for {nationality} preferences and travel patterns",
            'expected_impact': 'Enhance visitor experience and extend stay duration by 20-30%',
            'timeline': '3-6 months',
            'budget_estimate': '$75,000 - $150,000',
            'responsible': 'Tourism Package Development',
            'kpis': ['Package adoption rate', 'Average stay duration', 'Multi-city visitor satisfaction'],
            'department': 'Product Development & Partnerships',
            'resources_needed': ['Tour designers', 'Partnership coordinators', 'Package development budget']
        })
    
    # Enhanced service improvement recommendations
    if satisfaction_rate < 70:
        recommendations.extend([
            {
                'category': 'Service Quality',
                'priority': 'High',
                'action': f"Implement comprehensive cultural sensitivity training for staff serving {nationality} guests",
                'expected_impact': 'Improve satisfaction rates by 25-40% and cultural understanding',
                'timeline': '1-3 months',
                'budget_estimate': '$30,000 - $60,000',
                'responsible': 'Training & Development Division',
                'kpis': ['Staff cultural competency scores', 'Satisfaction rate improvement', 'Cultural incident reduction'],
                'department': 'Human Resources & Training',
                'resources_needed': ['Cultural trainers', 'Training materials', 'Assessment tools']
            },
            {
                'category': 'Service Standards',
                'priority': 'High',
                'action': f"Establish {nationality}-specific service protocols and quality standards across all tourism touchpoints",
                'expected_impact': 'Consistent service experience and improved satisfaction',
                'timeline': '2-4 months',
                'budget_estimate': '$40,000 - $80,000',
                'responsible': 'Quality Assurance Department',
                'kpis': ['Service consistency index', 'Protocol compliance rate', 'Guest satisfaction scores'],
                'department': 'Service Excellence Division',
                'resources_needed': ['Protocol developers', 'Quality assessors', 'Implementation coordinators']
            }
        ])
    
    # Cultural and communication enhancements
    if nationality not in ['United States', 'United Kingdom', 'Australia', 'Canada']:
        recommendations.extend([
            {
                'category': 'Communication',
                'priority': 'Medium',
                'action': f"Implement multilingual support and cultural liaison services for {nationality} guests",
                'expected_impact': 'Improve communication satisfaction and cultural comfort by 30-50%',
                'timeline': '2-5 months',
                'budget_estimate': '$45,000 - $90,000',
                'responsible': 'Guest Services Department',
                'kpis': ['Communication satisfaction scores', 'Language barrier incidents', 'Cultural comfort ratings'],
                'department': 'International Guest Services',
                'resources_needed': ['Multilingual staff', 'Translation services', 'Cultural training programs']
            },
            {
                'category': 'Cultural Integration',
                'priority': 'Medium',
                'action': f"Develop {nationality} cultural appreciation programs and authentic cultural experiences",
                'expected_impact': 'Enhanced cultural connection and repeat visitation',
                'timeline': '4-8 months',
                'budget_estimate': '$60,000 - $120,000',
                'responsible': 'Cultural Experience Development',
                'kpis': ['Cultural program participation', 'Experience satisfaction ratings', 'Cultural authenticity scores'],
                'department': 'Cultural Tourism & Heritage',
                'resources_needed': ['Cultural experts', 'Experience designers', 'Community partnerships']
            }
        ])
    
    # Preference-based recommendations
    if top_liked and len(top_liked) > 0:
        most_liked = top_liked[0]['thing']
        recommendations.append({
            'category': 'Strength Leveraging',
            'priority': 'Medium',
            'action': f"Enhance and promote '{most_liked}' experiences specifically for {nationality} visitors, as this is their top preference",
            'expected_impact': f'Capitalize on existing strengths to boost satisfaction and loyalty',
            'timeline': '1-3 months',
            'budget_estimate': '$25,000 - $50,000',
            'responsible': 'Experience Enhancement Team',
            'kpis': ['Experience quality scores', 'Preference satisfaction rate', 'Recommendation likelihood'],
            'department': 'Experience Development',
            'resources_needed': ['Experience specialists', 'Enhancement budget', 'Marketing support']
        })
    
    if top_disliked and len(top_disliked) > 0:
        most_disliked = top_disliked[0]['thing']
        recommendations.append({
            'category': 'Issue Resolution',
            'priority': 'High',
            'action': f"Urgently address '{most_disliked}' concerns specifically for {nationality} visitors through targeted improvement initiatives",
            'expected_impact': f'Eliminate primary dissatisfaction factor and improve overall experience',
            'timeline': '1-2 months',
            'budget_estimate': '$35,000 - $70,000',
            'responsible': 'Problem Resolution Team',
            'kpis': ['Issue resolution rate', 'Complaint reduction percentage', 'Satisfaction recovery'],
            'department': 'Quality Improvement Division',
            'resources_needed': ['Problem resolution specialists', 'Improvement budget', 'Monitoring systems']
        })
    
    # Market-specific infrastructure and amenities
    recommendations.append({
        'category': 'Infrastructure Development',
        'priority': 'Medium',
        'action': f"Assess and improve tourism infrastructure to better serve {nationality} visitor needs and expectations",
        'expected_impact': 'Enhanced visitor experience and competitive advantage',
        'timeline': '6-12 months',
        'budget_estimate': '$100,000 - $250,000',
        'responsible': 'Tourism Infrastructure Development',
        'kpis': ['Infrastructure satisfaction scores', 'Competitive ranking improvement', 'Visitor convenience ratings'],
        'department': 'Tourism Infrastructure & Planning',
        'resources_needed': ['Infrastructure planners', 'Construction budget', 'Technology upgrades']
    })
    
    return recommendations