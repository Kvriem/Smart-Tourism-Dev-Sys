# Fast data processing functions with performance optimization
import pandas as pd
from performance_cache import cache_result, aggregation_cache
from database_config import load_data_from_database

@cache_result("get_cities", ttl=1800)
def get_cities_fast():
    """Get unique cities list with caching"""
    df = load_data_from_database()
    if df.empty or 'City' not in df.columns:
        return []
    
    cities = df['City'].dropna().unique().tolist()
    return sorted(cities)

@cache_result("filter_data", ttl=300)
def filter_data_fast(city_filter="all", start_date=None, end_date=None):
    """Fast data filtering with caching"""
    df = load_data_from_database()
    
    if df.empty:
        return df
    
    filtered_df = df.copy()
    
    # Apply city filter
    if city_filter != "all" and 'City' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['City'] == city_filter]
    
    # Apply date filter if dates are provided
    if (start_date or end_date) and 'Review Date' in filtered_df.columns:
        if start_date:
            filtered_df = filtered_df[filtered_df['Review Date'] >= pd.to_datetime(start_date)]
        if end_date:
            filtered_df = filtered_df[filtered_df['Review Date'] <= pd.to_datetime(end_date)]
    
    print(f"⚡ Filtered data: {len(filtered_df)} records (from {len(df)} total)")
    return filtered_df

@cache_result("basic_stats", ttl=600)
def get_basic_stats_fast(city_filter="all", start_date=None, end_date=None):
    """Get basic statistics with caching"""
    cache_key = f"{city_filter}:{start_date}:{end_date}"
    cached_stats = aggregation_cache.get_city_stats(cache_key)
    
    if cached_stats is not None:
        print("⚡ Using aggregation cache for basic stats")
        return cached_stats
    
    df = filter_data_fast(city_filter, start_date, end_date)
    
    if df.empty:
        stats = {
            'total_reviews': 0,
            'positive_reviews': 0,
            'negative_reviews': 0,
            'neutral_reviews': 0,
            'satisfaction_rate': 0,
            'total_hotels': 0,
            'total_cities': 0
        }
    else:
        total_reviews = len(df)
        
        if 'sentiment classification' in df.columns:
            sentiment_counts = df['sentiment classification'].value_counts().to_dict()
            positive_reviews = sentiment_counts.get(1, 0)
            negative_reviews = sentiment_counts.get(-1, 0)
            neutral_reviews = sentiment_counts.get(0, 0)
        else:
            positive_reviews = negative_reviews = neutral_reviews = 0
        
        satisfaction_rate = (positive_reviews / total_reviews * 100) if total_reviews > 0 else 0
        
        stats = {
            'total_reviews': total_reviews,
            'positive_reviews': positive_reviews,
            'negative_reviews': negative_reviews,
            'neutral_reviews': neutral_reviews,
            'satisfaction_rate': round(satisfaction_rate, 1),
            'total_hotels': df['Hotel Name'].nunique() if 'Hotel Name' in df.columns else 0,
            'total_cities': df['City'].nunique() if 'City' in df.columns else 0
        }
    
    aggregation_cache.set_city_stats(cache_key, stats)
    return stats
