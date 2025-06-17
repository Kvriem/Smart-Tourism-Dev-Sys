"""
Performance monitoring and cache management utility for the dashboard.
"""

import time
import psutil
import os
from datetime import datetime
from performance_cache import performance_cache, aggregation_cache
from database_config import _global_data_cache, _global_cache_timestamp

def get_memory_usage():
    """Get current memory usage statistics"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
        'percent': process.memory_percent(),       # Percentage of total system memory
        'available_mb': psutil.virtual_memory().available / 1024 / 1024
    }

def get_cache_statistics():
    """Get cache usage statistics"""
    stats = {
        'performance_cache': {
            'memory_items': len(performance_cache.memory_cache),
            'processed_items': len(performance_cache.processed_cache),
            'chart_items': len(performance_cache.chart_cache)
        },
        'aggregation_cache': {
            'items': len(aggregation_cache.cache)
        },
        'global_cache': {
            'has_data': _global_data_cache is not None,
            'timestamp': _global_cache_timestamp
        }
    }
    
    return stats

def print_performance_report():
    """Print a comprehensive performance report"""
    print("\n" + "=" * 60)
    print("📊 DASHBOARD PERFORMANCE REPORT")
    print("=" * 60)
    
    # Memory usage
    memory = get_memory_usage()
    print(f"💾 Memory Usage:")
    print(f"   RSS: {memory['rss_mb']:.1f} MB")
    print(f"   VMS: {memory['vms_mb']:.1f} MB")
    print(f"   Percent: {memory['percent']:.1f}%")
    print(f"   Available: {memory['available_mb']:.1f} MB")
    
    # Cache statistics
    cache_stats = get_cache_statistics()
    print(f"\n🗄️ Cache Statistics:")
    print(f"   Performance Cache: {cache_stats['performance_cache']['memory_items']} items")
    print(f"   Processed Cache: {cache_stats['performance_cache']['processed_items']} items")
    print(f"   Chart Cache: {cache_stats['performance_cache']['chart_items']} items")
    print(f"   Aggregation Cache: {cache_stats['aggregation_cache']['items']} items")
    print(f"   Global Cache: {'✅ Active' if cache_stats['global_cache']['has_data'] else '❌ Empty'}")
    
    # Cache directory size
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
    if os.path.exists(cache_dir):
        total_size = 0
        file_count = 0
        for root, dirs, files in os.walk(cache_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
                    file_count += 1
        
        print(f"   Disk Cache: {file_count} files, {total_size / 1024 / 1024:.1f} MB")
    
    print("=" * 60)

def clear_all_caches():
    """Clear all caches and print confirmation"""
    print("🧹 Clearing all caches...")
    
    # Clear performance caches
    performance_cache.clear_all()
    aggregation_cache.clear()
    
    # Clear global cache
    global _global_data_cache, _global_cache_timestamp
    from database_config import _global_data_cache, _global_cache_timestamp
    _global_data_cache = None
    _global_cache_timestamp = None
    
    print("✅ All caches cleared")

def optimize_memory():
    """Run garbage collection and memory optimization"""
    print("🔧 Running memory optimization...")
    
    import gc
    
    # Force garbage collection
    collected = gc.collect()
    print(f"   Collected {collected} objects")
    
    # Get memory usage after optimization
    memory_after = get_memory_usage()
    print(f"   Memory after optimization: {memory_after['rss_mb']:.1f} MB")

def monitor_performance(duration_seconds=60, interval_seconds=10):
    """Monitor performance over time"""
    print(f"📈 Starting performance monitoring for {duration_seconds} seconds...")
    
    start_time = time.time()
    measurements = []
    
    while time.time() - start_time < duration_seconds:
        memory = get_memory_usage()
        cache_stats = get_cache_statistics()
        
        measurement = {
            'timestamp': datetime.now(),
            'memory_mb': memory['rss_mb'],
            'memory_percent': memory['percent'],
            'cache_items': (cache_stats['performance_cache']['memory_items'] + 
                          cache_stats['performance_cache']['processed_items'] + 
                          cache_stats['performance_cache']['chart_items'])
        }
        
        measurements.append(measurement)
        
        print(f"⏱️ {measurement['timestamp'].strftime('%H:%M:%S')} - "
              f"Memory: {measurement['memory_mb']:.1f}MB ({measurement['memory_percent']:.1f}%) - "
              f"Cache: {measurement['cache_items']} items")
        
        time.sleep(interval_seconds)
    
    print("📊 Monitoring complete")
    return measurements

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "report":
            print_performance_report()
        elif command == "clear":
            clear_all_caches()
            print_performance_report()
        elif command == "optimize":
            optimize_memory()
            print_performance_report()
        elif command == "monitor":
            duration = int(sys.argv[2]) if len(sys.argv) > 2 else 60
            monitor_performance(duration)
        else:
            print("Usage: python performance_monitor.py [report|clear|optimize|monitor [duration]]")
    else:
        print_performance_report()
