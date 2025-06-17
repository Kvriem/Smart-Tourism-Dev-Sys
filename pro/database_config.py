import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text
import os
import pickle
from datetime import datetime
import hashlib
import json
import time
from performance_cache import performance_cache, cache_result, optimize_dataframe, aggregation_cache

# Database connection configuration
DATABASE_URL = "postgresql://neondb_owner:npg_ExFXHY8yiNT0@ep-lingering-term-ab7pbfql-pooler.eu-west-2.aws.neon.tech/neondb?sslmode=require"

# Cache configuration
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
CACHE_FILE = os.path.join(CACHE_DIR, 'data_cache.pkl')
CACHE_METADATA_FILE = os.path.join(CACHE_DIR, 'cache_metadata.json')
CACHE_DURATION = 3600  # 1 hour cache duration in seconds

# Global data cache for session-level caching
_global_data_cache = None
_global_cache_timestamp = None
_cache_lock = False

def get_database_connection():
    """Create and return a database connection"""
    try:
        # Add connection timeout and pool settings
        engine = create_engine(
            DATABASE_URL,
            pool_timeout=30,  # Increased timeout for large datasets
            pool_recycle=3600,  # Recycle connections every hour
            pool_pre_ping=True,  # Verify connections before use
            connect_args={
                "connect_timeout": 30,  # Increased connection timeout
                "keepalives": 1,        # Enable keepalives
                "keepalives_idle": 30,  # Send keepalive after 30 seconds of inactivity
                "keepalives_interval": 10,  # Check connection every 10 seconds
                "keepalives_count": 5    # Allow 5 retries before giving up
            }
        )
        # Test the connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()  # Fetch the result
        return engine
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def process_dataframe(df):
    """Process the dataframe to match expected format"""
    if df.empty:
        return df
        
    # Rename columns to match expected format
    df = df.rename(columns={
        'reviewer_nationality': 'Nationality',
        'review_date': 'Review Date',
        'sentiment_classification': 'sentiment classification',
        'positive_tokens': 'positive tokens',
        'negative_tokens': 'negative tokens',
        'city': 'City',
        'hotel_name': 'Hotel Name',
        'reviewer_name': 'Reviewer Name',
        'travel_type': 'Travel Type',
        'room_type': 'Room Type',
        'check_in_date': 'Check-in Date',
        'positive_review': 'Positive Review',
        'negative_review': 'Negative Review',
        'duration': 'Duration'
    })
    
    # Convert date columns
    df['Review Date'] = pd.to_datetime(df['Review Date'], errors='coerce')
    
    # Convert array columns to proper format for processing
    if 'positive tokens' in df.columns:
        df['positive tokens'] = df['positive tokens'].apply(lambda x: str(x) if x is not None else '[]')
    
    if 'negative tokens' in df.columns:
        df['negative tokens'] = df['negative tokens'].apply(lambda x: str(x) if x is not None else '[]')
      # Ensure sentiment classification is in the correct format (-1, 0, 1)
    if 'sentiment classification' in df.columns:
        df['sentiment classification'] = df['sentiment classification'].apply(
            lambda x: int(x) if pd.notnull(x) else 0
        )
    
    print(f"Successfully processed {len(df)} rows")
    print(f"Sample cities: {df['City'].unique()[:3].tolist() if 'City' in df.columns else 'No cities found'}")
    
    return df

def ensure_cache_dir():
    """Ensure cache directory exists"""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        print(f"[SUCCESS] Created cache directory: {CACHE_DIR}")

def get_cached_metadata():
    """Get metadata from cache file"""
    if not os.path.exists(CACHE_METADATA_FILE):
        return {"last_updated": None, "last_id": None, "record_count": 0}
    
    try:
        with open(CACHE_METADATA_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading cache metadata: {e}")
        return {"last_updated": None, "last_id": None, "record_count": 0}

def update_cache_metadata(metadata):
    """Update cache metadata"""
    ensure_cache_dir()
    try:
        with open(CACHE_METADATA_FILE, 'w') as f:
            json.dump(metadata, f)
    except Exception as e:
        print(f"Error updating cache metadata: {e}")

def load_cached_data():
    """Load data from cache file"""
    if not os.path.exists(CACHE_FILE):
        return None
    
    try:
        with open(CACHE_FILE, 'rb') as f:
            cached_data = pickle.load(f)
            print(f"[SUCCESS] Loaded {len(cached_data)} records from cache")
            return cached_data
    except Exception as e:
        print(f"Error loading cached data: {e}")
        return None

def save_to_cache(df):
    """Save DataFrame to cache"""
    ensure_cache_dir()
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(df, f)
        
        # Update metadata
        metadata = {
            "last_updated": datetime.now().isoformat(),
            "record_count": int(len(df)),
            "last_id": int(df["id"].max()) if "id" in df.columns and not df.empty else None
        }
        update_cache_metadata(metadata)
        print(f"[SUCCESS] Saved {len(df)} records to cache")
    except Exception as e:
        print(f"Error saving to cache: {e}")

@cache_result("database_load", ttl=3600, use_disk=True)
def load_data_from_database(force_reload=False):
    """Load data from PostgreSQL database with optimized performance and caching"""
    global _global_data_cache, _global_cache_timestamp, _cache_lock
    
    # If we're already loading, wait for it to complete
    if _cache_lock:
        print("⏳ Data loading in progress, waiting...")
        while _cache_lock:
            time.sleep(0.1)
        if _global_data_cache is not None:
            return _global_data_cache
    
    # Check global cache first for immediate response during page switches
    current_time = time.time()
    if (not force_reload and _global_data_cache is not None and 
        _global_cache_timestamp and 
        (current_time - _global_cache_timestamp) < 300):  # 5 minute cache for page switches
        print(f"[FAST] Using global cache ({len(_global_data_cache)} records) - instant response")
        return _global_data_cache
    
    _cache_lock = True
    start_time = time.time()
    
    try:
        # Try to load from cache first if not forcing reload
        if not force_reload:
            cached_data = load_cached_data()
            metadata = get_cached_metadata()
            
            # Check if cache is valid
            if cached_data is not None and metadata.get("last_updated"):
                # Always check for new records regardless of cache age
                if metadata.get("last_id") is not None:
                    engine = get_database_connection()
                    if engine is not None:
                        try:
                            # Check if there are any new records without fetching them yet
                            check_query = f"""
                            SELECT COUNT(*) 
                            FROM gold.final_reviews 
                            WHERE id > {metadata['last_id']}
                            """
                            
                            with engine.connect() as conn:
                                new_count = conn.execute(text(check_query)).scalar()
                            
                            if new_count > 0:
                                print(f"[SUCCESS] Found {new_count} new records to add")
                                
                                # Query only new records
                                new_records_query = f"""
                                SELECT 
                                    id,
                                    city,
                                    hotel_name,
                                    reviewer_name,
                                    reviewer_nationality,
                                    duration,
                                    check_in_date,
                                    review_date,
                                    travel_type,
                                    room_type,
                                    positive_review,
                                    negative_review,
                                    sentiment_classification,
                                    positive_tokens,
                                    negative_tokens,
                                    inserted_at
                                FROM gold.final_reviews
                                WHERE id > {metadata['last_id']}
                                ORDER BY id ASC
                                """
                                new_df = pd.read_sql(new_records_query, engine)
                                
                                # Process the new data to match expected format
                                processed_new_df = process_dataframe(new_df)
                                
                                # Combine with cached data
                                combined_df = pd.concat([cached_data, processed_new_df], ignore_index=True)
                                
                                # Optimize DataFrame memory usage
                                combined_df = optimize_dataframe(combined_df)
                                
                                # Save the updated data to cache
                                save_to_cache(combined_df)
                                
                                # Update global cache
                                _global_data_cache = combined_df
                                _global_cache_timestamp = current_time
                                
                                print(f"[SUCCESS] Cache updated with {new_count} new records, total: {len(combined_df)}")
                                return combined_df
                            else:
                                # No new records found, use existing cache
                                elapsed = time.time() - start_time
                                
                                # Optimize existing data if not already optimized
                                if not hasattr(cached_data, '_optimized'):
                                    cached_data = optimize_dataframe(cached_data)
                                    cached_data._optimized = True
                                
                                # Update global cache
                                _global_data_cache = cached_data
                                _global_cache_timestamp = current_time
                                
                                print(f"[SUCCESS] No new records found, using cached data ({elapsed:.2f}s)")
                                return cached_data
                        except Exception as e:
                            print(f"Error fetching incremental data: {e}")
                            print("[WARNING] Falling back to cached data")
                            
                            # Update global cache with existing data
                            _global_data_cache = cached_data
                            _global_cache_timestamp = current_time
                            return cached_data
                
                # If we get here, just return the cached data
                elapsed = time.time() - start_time
                
                # Update global cache
                _global_data_cache = cached_data
                _global_cache_timestamp = current_time
                
                print(f"[SUCCESS] Using existing cache with {len(cached_data)} records ({elapsed:.2f}s)")
                return cached_data
        
        # If we get here, we need to load all data from scratch
        print("[CONNECT] Attempting to connect to PostgreSQL database for full data reload...")
        engine = get_database_connection()
        if engine is None:
            print("[ERROR] Failed to connect to database - will try to use cache")
            cached_data = load_cached_data()
            if cached_data is not None:
                _global_data_cache = cached_data
                _global_cache_timestamp = current_time
                return cached_data
            return pd.DataFrame()
        
        print("[SUCCESS] Connected successfully, executing query...")
        
        # Query to load all data from final_reviews table in gold schema
        query = """
        SELECT 
            id,
            city,
            hotel_name,
            reviewer_name,
            reviewer_nationality,
            duration,
            check_in_date,
            review_date,
            travel_type,
            room_type,
            positive_review,
            negative_review,
            sentiment_classification,
            positive_tokens,
            negative_tokens,
            inserted_at
        FROM gold.final_reviews
        ORDER BY id ASC
        """
        
        # Execute query with cross-platform timeout handling
        try:
            # Use pandas read_sql which respects the engine's connection timeout
            print("⏳ Querying database, this may take a moment for all rows...")
            
            # Use batched loading to improve performance
            # First, get the count of records
            count_query = "SELECT COUNT(*) FROM gold.final_reviews"
            with engine.connect() as conn:
                count = conn.execute(text(count_query)).scalar()
            
            # If the count is large, use batched loading
            if count > 10000:
                print(f"[DATA] Large dataset detected ({count} records). Using optimized batched loading...")
                
                # Use batched loading with larger batch sizes for better performance
                batch_size = 25000  # Increased batch size significantly
                batches = []
                
                # Get the min and max IDs
                min_max_query = "SELECT MIN(id), MAX(id) FROM gold.final_reviews"
                with engine.connect() as conn:
                    min_id, max_id = conn.execute(text(min_max_query)).fetchone()
                
                print(f"[DATA] Loading data in optimized batches from ID {min_id} to {max_id}")
                
                # Load data in batches with progress reporting
                total_batches = (max_id - min_id) // batch_size + 1
                for batch_num, start_id in enumerate(range(min_id, max_id + 1, batch_size), 1):
                    end_id = min(start_id + batch_size - 1, max_id)
                    
                    batch_query = f"""
                    SELECT 
                        id,
                        city,
                        hotel_name,
                        reviewer_name,
                        reviewer_nationality,
                        duration,
                        check_in_date,
                        review_date,
                        travel_type,
                        room_type,
                        positive_review,
                        negative_review,
                        sentiment_classification,
                        positive_tokens,
                        negative_tokens,
                        inserted_at
                    FROM gold.final_reviews
                    WHERE id BETWEEN {start_id} AND {end_id}
                    ORDER BY id ASC
                    """
                    
                    batch_start = time.time()
                    batch_df = pd.read_sql(batch_query, engine)
                    batch_time = time.time() - batch_start
                    
                    batches.append(batch_df)
                    
                    # Report progress
                    progress = batch_num / total_batches * 100
                    print(f"[FAST] Batch {batch_num}/{total_batches}: {len(batch_df)} records in {batch_time:.1f}s ({progress:.1f}%)")
                
                # Combine all batches efficiently
                print("[PROCESS] Combining batches...")
                df = pd.concat(batches, ignore_index=True)
                print(f"[SUCCESS] Successfully loaded all {len(df)} records in {total_batches} optimized batches")
            else:
                # For smaller datasets, load all at once
                df = pd.read_sql(query, engine)
                
            print(f"[SUCCESS] Query executed successfully, got {len(df)} rows")
        except Exception as query_error:
            print(f"Database query failed: {query_error}")
            return pd.DataFrame()
            
        # Process the data
        processed_df = process_dataframe(df)
        
        # Optimize DataFrame memory usage
        processed_df = optimize_dataframe(processed_df)
        
        # Save to cache
        save_to_cache(processed_df)
        
        # Update global cache
        _global_data_cache = processed_df
        _global_cache_timestamp = current_time
        
        # Report on performance
        end_time = time.time()
        load_time = end_time - start_time
        print(f"⏱️ Data loaded and optimized in {load_time:.2f} seconds")
        
        return processed_df
        
    except Exception as e:
        print(f"[ERROR] Error loading data from database: {e}")
        print("   This is normal if the database is unreachable - falling back to cache if available")
        import traceback
        traceback.print_exc()
        
        # Try to use cached data as a last resort
        cached_data = load_cached_data()
        if cached_data is not None:
            _global_data_cache = cached_data
            _global_cache_timestamp = current_time
            return cached_data
        
        return pd.DataFrame()
    
    finally:
        _cache_lock = False

def get_cities_from_database():
    """Get unique cities from the database"""
    try:
        engine = get_database_connection()
        if engine is None:
            return []
        query = "SELECT DISTINCT city FROM gold.final_reviews WHERE city IS NOT NULL ORDER BY city"
        result = pd.read_sql(text(query), engine)
        cities = result['city'].tolist()
        return cities
        
    except Exception as e:
        print(f"Error loading cities from database: {e}")
        return []
