#!/usr/bin/env python3
"""
Optimized startup script for the dashboard application.
Preloads data and starts the server with performance optimizations.
"""

import sys
import os
import time
import threading
from datetime import datetime

# Add the project directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def preload_data():
    """Preload data in the background for faster initial page loads"""
    print("🚀 Starting background data preload...")
    start_time = time.time()
    
    try:
        # Import and preload essential data
        from data_processing_optimized import get_cities_fast, get_basic_stats_fast
        from database_config import load_data_from_database
        
        # Preload main data
        print("📊 Preloading main dataset...")
        df = load_data_from_database()
        
        if not df.empty:
            print(f"✅ Main dataset loaded: {len(df)} records")
            
            # Preload cities
            print("🏙️ Preloading cities...")
            cities = get_cities_fast()
            print(f"✅ Cities loaded: {len(cities)} cities")
            
            # Preload basic stats
            print("📈 Preloading basic statistics...")
            stats = get_basic_stats_fast()
            print(f"✅ Basic stats loaded: {stats['satisfaction_rate']:.1f}% satisfaction rate")
            
            elapsed = time.time() - start_time
            print(f"🎉 Data preload completed in {elapsed:.2f} seconds")
        else:
            print("⚠️ No data available during preload")
            
    except Exception as e:
        print(f"❌ Error during data preload: {e}")
        import traceback
        traceback.print_exc()

def optimize_app_settings():
    """Apply performance optimizations to the app"""
    print("⚙️ Applying performance optimizations...")
    
    # Set environment variables for better performance
    os.environ['DASH_DEBUG'] = 'False'
    os.environ['DASH_DEV_TOOLS_HOT_RELOAD'] = 'False'
    os.environ['DASH_DEV_TOOLS_UI'] = 'False'
    os.environ['DASH_DEV_TOOLS_PROPS_CHECK'] = 'False'
    
    # Memory optimizations
    import gc
    gc.set_threshold(700, 10, 10)  # More aggressive garbage collection
    
    print("✅ Performance optimizations applied")

def main():
    """Main application startup with performance optimizations"""
    print("=" * 60)
    print("🎯 ANALYTICS DASHBOARD - PERFORMANCE OPTIMIZED")
    print("=" * 60)
    
    # Apply optimizations
    optimize_app_settings()
    
    # Start background data preload
    preload_thread = threading.Thread(target=preload_data, daemon=True)
    preload_thread.start()
    
    # Import and start the main app
    print("🌐 Starting Dash server...")
    
    try:
        from app import app
        
        # Configure the server for better performance
        app.server.config['SEND_FILE_MAX_AGE_DEFAULT'] = 300  # 5 minutes cache for static files
        
        # Start the server
        print("🚀 Dashboard starting at http://127.0.0.1:8050")
        print("⚡ Performance optimizations enabled")
        print("📊 Data preloading in background...")
        print("-" * 60)
        
        app.run_server(
            debug=False,
            host='127.0.0.1',
            port=8050,
            dev_tools_hot_reload=False,
            dev_tools_ui=False,
            dev_tools_props_check=False,
            threaded=True
        )
        
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
