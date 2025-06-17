"""
Advanced performance caching system for the dashboard application.
Implements multi-level caching with memory, disk, and processed data caches.
"""

import pandas as pd
import pickle
import json
import hashlib
import time
from datetime import datetime, timedelta
from functools import wraps
import os
import threading
from typing import Any, Dict, Optional, Callable
import plotly.graph_objects as go

class PerformanceCache:
    """Advanced caching system with memory and disk layers"""
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
        self.memory_cache = {}
        self.memory_timestamps = {}
        self.processed_cache = {}  # For processed data like aggregations
        self.chart_cache = {}      # For chart objects
        self.lock = threading.Lock()
        
        # Cache configuration
        self.MEMORY_TTL = 300      # 5 minutes for memory cache
        self.DISK_TTL = 3600       # 1 hour for disk cache
        self.PROCESSED_TTL = 600   # 10 minutes for processed data
        self.CHART_TTL = 300       # 5 minutes for charts
        self.MAX_MEMORY_ITEMS = 50 # Maximum items in memory cache
        
        self.ensure_cache_dir()
    
    def ensure_cache_dir(self):
        """Ensure cache directory exists"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def _generate_key(self, prefix: str, **kwargs) -> str:
        """Generate a cache key from parameters"""
        # Create a hash from all parameters
        key_string = f"{prefix}:{json.dumps(kwargs, sort_keys=True, default=str)}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _cleanup_memory_cache(self):
        """Remove expired items from memory cache"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.memory_timestamps.items()
            if current_time - timestamp > self.MEMORY_TTL
        ]
        
        for key in expired_keys:
            self.memory_cache.pop(key, None)
            self.memory_timestamps.pop(key, None)
        
        # If still too many items, remove oldest
        if len(self.memory_cache) > self.MAX_MEMORY_ITEMS:
            sorted_items = sorted(self.memory_timestamps.items(), key=lambda x: x[1])
            items_to_remove = len(self.memory_cache) - self.MAX_MEMORY_ITEMS
            
            for key, _ in sorted_items[:items_to_remove]:
                self.memory_cache.pop(key, None)
                self.memory_timestamps.pop(key, None)
    
    def get_from_memory(self, key: str) -> Optional[Any]:
        """Get item from memory cache"""
        with self.lock:
            self._cleanup_memory_cache()
            if key in self.memory_cache:
                timestamp = self.memory_timestamps.get(key, 0)
                if time.time() - timestamp <= self.MEMORY_TTL:
                    return self.memory_cache[key]
                else:
                    # Expired
                    self.memory_cache.pop(key, None)
                    self.memory_timestamps.pop(key, None)
        return None
    
    def set_in_memory(self, key: str, value: Any):
        """Set item in memory cache"""
        with self.lock:
            self.memory_cache[key] = value
            self.memory_timestamps[key] = time.time()
            self._cleanup_memory_cache()
    
    def get_from_disk(self, key: str) -> Optional[Any]:
        """Get item from disk cache"""
        file_path = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(file_path):
            try:
                # Check if file is not too old
                file_time = os.path.getmtime(file_path)
                if time.time() - file_time <= self.DISK_TTL:
                    with open(file_path, 'rb') as f:
                        return pickle.load(f)
                else:
                    # File too old, remove it
                    os.remove(file_path)
            except Exception:
                # Corrupted file, remove it
                try:
                    os.remove(file_path)
                except:
                    pass
        return None
    
    def set_in_disk(self, key: str, value: Any):
        """Set item in disk cache"""
        file_path = os.path.join(self.cache_dir, f"{key}.pkl")
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Warning: Failed to save to disk cache: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache (memory first, then disk)"""
        # Try memory first
        value = self.get_from_memory(key)
        if value is not None:
            return value
        
        # Try disk cache
        value = self.get_from_disk(key)
        if value is not None:
            # Store in memory for next time
            self.set_in_memory(key, value)
            return value
        
        return None
    
    def set(self, key: str, value: Any, memory_only: bool = False):
        """Set item in cache"""
        self.set_in_memory(key, value)
        if not memory_only:
            self.set_in_disk(key, value)
    
    def clear_all(self):
        """Clear all caches"""
        with self.lock:
            self.memory_cache.clear()
            self.memory_timestamps.clear()
            self.processed_cache.clear()
            self.chart_cache.clear()
        
        # Clear disk cache
        try:
            for file in os.listdir(self.cache_dir):
                if file.endswith('.pkl') and file != 'data_cache.pkl':
                    os.remove(os.path.join(self.cache_dir, file))
        except Exception:
            pass
    
    def cache_processed_data(self, key: str, data: Any):
        """Cache processed data with shorter TTL"""
        cache_key = f"processed:{key}"
        self.processed_cache[cache_key] = {
            'data': data,
            'timestamp': time.time()
        }
    
    def get_processed_data(self, key: str) -> Optional[Any]:
        """Get processed data from cache"""
        cache_key = f"processed:{key}"
        if cache_key in self.processed_cache:
            cached_item = self.processed_cache[cache_key]
            if time.time() - cached_item['timestamp'] <= self.PROCESSED_TTL:
                return cached_item['data']
            else:
                del self.processed_cache[cache_key]
        return None
    
    def cache_chart(self, key: str, chart: go.Figure):
        """Cache chart with shorter TTL"""
        cache_key = f"chart:{key}"
        self.chart_cache[cache_key] = {
            'chart': chart,
            'timestamp': time.time()
        }
    
    def get_cached_chart(self, key: str) -> Optional[go.Figure]:
        """Get cached chart"""
        cache_key = f"chart:{key}"
        if cache_key in self.chart_cache:
            cached_item = self.chart_cache[cache_key]
            if time.time() - cached_item['timestamp'] <= self.CHART_TTL:
                return cached_item['chart']
            else:
                del self.chart_cache[cache_key]
        return None

# Global cache instance
performance_cache = PerformanceCache()

def cache_result(cache_key_prefix: str = "", ttl: int = 300, use_disk: bool = True):
    """
    Decorator to cache function results
    
    Args:
        cache_key_prefix: Prefix for cache key
        ttl: Time to live in seconds
        use_disk: Whether to use disk cache
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key_data = {
                'function': func.__name__,
                'args': str(args),
                'kwargs': kwargs
            }
            cache_key = performance_cache._generate_key(cache_key_prefix or func.__name__, **key_data)
            
            # Try to get from cache
            cached_result = performance_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Cache result
            performance_cache.set(cache_key, result, memory_only=not use_disk)
            
            print(f"⚡ {func.__name__} executed in {execution_time:.2f}s and cached")
            return result
        
        return wrapper
    return decorator

def cache_processed_result(cache_key_prefix: str = ""):
    """
    Decorator specifically for processed data caching
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key_data = {
                'function': func.__name__,
                'args': str(args),
                'kwargs': kwargs
            }
            cache_key = performance_cache._generate_key(cache_key_prefix or func.__name__, **key_data)
            
            # Try to get from processed cache
            cached_result = performance_cache.get_processed_data(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Cache result in processed cache
            performance_cache.cache_processed_data(cache_key, result)
            
            print(f"🔄 {func.__name__} processed in {execution_time:.2f}s and cached")
            return result
        
        return wrapper
    return decorator

def cache_chart_result(cache_key_prefix: str = ""):
    """
    Decorator specifically for chart caching
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key_data = {
                'function': func.__name__,
                'args': str(args),
                'kwargs': kwargs
            }
            cache_key = performance_cache._generate_key(cache_key_prefix or func.__name__, **key_data)
            
            # Try to get from chart cache
            cached_chart = performance_cache.get_cached_chart(cache_key)
            if cached_chart is not None:
                return cached_chart
            
            # Execute function
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Cache chart
            if isinstance(result, go.Figure):
                performance_cache.cache_chart(cache_key, result)
            
            print(f"📊 {func.__name__} chart created in {execution_time:.2f}s and cached")
            return result
        
        return wrapper
    return decorator

# Utility functions for data optimization
def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage
    """
    if df.empty:
        return df
    
    # Optimize memory usage
    optimized_df = df.copy()
    
    # Convert object columns that contain numbers to appropriate types
    for col in optimized_df.select_dtypes(include=['object']).columns:
        # Try to convert to category if low cardinality
        if optimized_df[col].nunique() / len(optimized_df) < 0.1:  # Less than 10% unique values
            optimized_df[col] = optimized_df[col].astype('category')
    
    # Downcast numeric types
    for col in optimized_df.select_dtypes(include=['int64']).columns:
        optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='integer')
    
    for col in optimized_df.select_dtypes(include=['float64']).columns:
        optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
    
    return optimized_df

def get_filtered_data_key(city_filter: str = "all", start_date: str = None, end_date: str = None) -> str:
    """Generate a consistent key for filtered data"""
    return f"filtered_data:{city_filter}:{start_date}:{end_date}"

# Pre-computed aggregations cache
class AggregationCache:
    """Cache for pre-computed aggregations"""
    
    def __init__(self):
        self.cache = {}
        self.timestamps = {}
        self.TTL = 600  # 10 minutes
    
    def get_city_stats(self, city: str = "all") -> Optional[Dict]:
        """Get cached city statistics"""
        key = f"city_stats:{city}"
        if key in self.cache:
            if time.time() - self.timestamps[key] <= self.TTL:
                return self.cache[key]
            else:
                del self.cache[key]
                del self.timestamps[key]
        return None
    
    def set_city_stats(self, city: str = "all", stats: Dict = None):
        """Cache city statistics"""
        key = f"city_stats:{city}"
        self.cache[key] = stats
        self.timestamps[key] = time.time()
    
    def clear(self):
        """Clear aggregation cache"""
        self.cache.clear()
        self.timestamps.clear()

# Global aggregation cache
aggregation_cache = AggregationCache()
