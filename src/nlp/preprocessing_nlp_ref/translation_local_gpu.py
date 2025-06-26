#!/usr/bin/env python3
"""
High-Performance Translation Script for Local Execution
Optimized for parallel processing and automatic translation
"""

import pandas as pd
import numpy as np
import time
import logging
import warnings
from datetime import datetime
from functools import lru_cache
import gc
import os
import sys
import tempfile
from typing import List, Tuple, Optional

# Fix temporary directory issue
if not os.path.exists('/tmp'):
    os.makedirs('/tmp', exist_ok=True)
os.environ['TMPDIR'] = '/tmp'

# Built-in parallel processing (no external dependencies)
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
import threading
from queue import Queue

# Progress tracking (optional)
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Simple progress fallback
    class tqdm:
        def __init__(self, iterable=None, desc="", total=None):
            self.iterable = iterable
            self.desc = desc
            self.total = total
            
        def __iter__(self):
            return iter(self.iterable)
        
        def __enter__(self):
            return self
            
        def __exit__(self, *args):
            pass

# Translation and language detection (essential)
try:
    from deep_translator import GoogleTranslator
    import langdetect
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False
    print("❌ Translation packages not available. Please install: pip install deep-translator langdetect")
    sys.exit(1)

# Database
from sqlalchemy import create_engine, text
import psycopg2
from urllib.parse import urlparse

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('translation_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Performance configuration
MAX_WORKERS = min(32, cpu_count() * 4)
CHUNK_SIZE = 1000
CACHE_SIZE = 100000
BATCH_SIZE = 50

# Database connection - matching preprocessing script
DB_CONNECTION_STRING = "postgresql://neondb_owner:npg_ExFXHY8yiNT0@ep-lingering-term-ab7pbfql-pooler.eu-west-2.aws.neon.tech/neondb?sslmode=require"

print(f"🚀 Local Translation Pipeline Configuration:")
print(f"   Max Workers: {MAX_WORKERS}")
print(f"   CPU Cores: {cpu_count()}")
print(f"   Cache Size: {CACHE_SIZE}")

class PerformanceCache:
    def __init__(self, maxsize=CACHE_SIZE):
        self.language_cache = {}
        self.translation_cache = {}
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock()
    
    def get_language(self, text):
        with self.lock:
            if text in self.language_cache:
                self.hits += 1
                return self.language_cache[text]
            
            self.misses += 1
            try:
                if isinstance(text, str) and len(text.strip()) > 3:
                    lang = langdetect.detect(text)
                else:
                    lang = 'en'
                
                if len(self.language_cache) < self.maxsize:
                    self.language_cache[text] = lang
                return lang
            except:
                return 'en'
    
    def get_translation(self, text, source_lang):
        cache_key = f"{source_lang}:{text}"
        with self.lock:
            if cache_key in self.translation_cache:
                self.hits += 1
                return self.translation_cache[cache_key]
            
            self.misses += 1
            try:
                if source_lang != 'en' and isinstance(text, str) and text.strip():
                    translated = GoogleTranslator(source=source_lang, target='en').translate(text)
                    if len(self.translation_cache) < self.maxsize:
                        self.translation_cache[cache_key] = translated
                    return translated
                return text
            except Exception as e:
                logger.warning(f"Translation failed: {str(e)}")
                return text
    
    def get_stats(self):
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.language_cache) + len(self.translation_cache)
        }

# Global performance cache
perf_cache = PerformanceCache(CACHE_SIZE)

def preprocess_texts(texts):
    """Basic text preprocessing (CPU-based)"""
    return [str(t).strip().lower() if isinstance(t, str) else "" for t in texts]

def fast_language_detection(text):
    """Ultra-fast language detection with heuristics"""
    if not isinstance(text, str) or len(text.strip()) < 4:
        return 'en'
    
    text_lower = text.lower().strip()
    
    # Quick English detection heuristics
    english_patterns = [
        'the ', ' and ', ' was ', ' were ', ' have ', ' this ', ' that ',
        ' with ', ' very ', ' good ', ' great ', ' nice ', ' bad ', ' hotel ',
        ' room ', ' staff ', ' location ', ' service ', ' clean ', ' breakfast '
    ]
    
    english_count = sum(1 for pattern in english_patterns if pattern in text_lower)
    if english_count >= 2:
        return 'en'
    
    return perf_cache.get_language(text)

def translate_text_batch(text_batch):
    """Translate a batch of texts in parallel"""
    results = []
    
    for text in text_batch:
        if not isinstance(text, str) or len(text.strip()) < 4:
            results.append(text)
            continue
        
        detected_lang = fast_language_detection(text)
        
        if detected_lang == 'en':
            results.append(text)
        else:
            translated = perf_cache.get_translation(text, detected_lang)
            results.append(translated)
    
    return results

def create_optimized_engine():
    """Create highly optimized database engine - matching preprocessing script pattern"""
    logger.info("Creating optimized database engine...")
    try:
        engine = create_engine(
            DB_CONNECTION_STRING,
            pool_size=20,
            max_overflow=40,
            pool_pre_ping=True,
            pool_recycle=1800,
            connect_args={
                "connect_timeout": 60,
                "application_name": "local_translation_gpu"
            }
        )
        logger.info("Optimized database engine created successfully")
        return engine
    except Exception as e:
        logger.error(f"Failed to create optimized database engine: {str(e)}")
        raise

def read_data_optimized(engine, limit=None):
    """Read data with optimization for large datasets - matching preprocessing pattern"""
    query = "SELECT * FROM silver.reviews_cleaned"
    if limit:
        query += f" LIMIT {limit}"
    
    logger.info(f"Reading data from silver.reviews_cleaned...")
    start_time = time.time()
    
    try:
        # Use direct psycopg2 connection like in preprocessing script
        parsed = urlparse(DB_CONNECTION_STRING)
        
        # Create psycopg2 connection
        conn = psycopg2.connect(
            host=parsed.hostname,
            port=parsed.port or 5432,
            database=parsed.path[1:],  # Remove leading '/'
            user=parsed.username,
            password=parsed.password,
            sslmode='require'
        )
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        elapsed = time.time() - start_time
        logger.info(f"Successfully loaded {len(df)} records from silver.reviews_cleaned in {elapsed:.2f} seconds")
        
        # Log data quality metrics like preprocessing script
        if len(df) > 0:
            non_null_positive = df['positive_review'].notna().sum()
            non_null_negative = df['negative_review'].notna().sum()
            
            logger.info(f"Translation data quality metrics:")
            logger.info(f"  Records with positive reviews: {non_null_positive}")
            logger.info(f"  Records with negative reviews: {non_null_negative}")
            logger.info(f"  Dataset shape: {df.shape}")
            logger.info(f"  Columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error reading from silver.reviews_cleaned: {e}")
        raise

def ensure_table_exists_optimized(engine):
    """Optimized table creation - enhanced error handling"""
    logger.info("Setting up optimized translated table...")
    
    try:
        with engine.begin() as conn:
            # Create silver schema if it doesn't exist
            logger.info("Creating silver schema if not exists...")
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS silver"))
            
            # Drop existing table if it exists
            logger.info("Dropping existing silver_translated table if exists...")
            conn.execute(text("DROP TABLE IF EXISTS silver.silver_translated"))
            
            # Create table with proper schema
            logger.info("Creating silver_translated table...")
            create_sql = """
            CREATE TABLE silver.silver_translated (
                id SERIAL PRIMARY KEY,
                city TEXT,
                hotel_name TEXT,
                reviewer_name TEXT,
                reviewer_nationality TEXT,
                duration TEXT,
                check_in_date TEXT,
                travel_type TEXT,
                room_type TEXT,
                review_date TEXT,
                positive_review TEXT,
                negative_review TEXT,
                ingestion_timestamp TIMESTAMP,
                sentiment_classification INTEGER,
                negative_review_translated TEXT,
                positive_review_translated TEXT
            );
            
            CREATE INDEX idx_silver_translated_city ON silver.silver_translated(city);
            CREATE INDEX idx_silver_translated_hotel ON silver.silver_translated(hotel_name);
            """
            
            conn.execute(text(create_sql))
            logger.info("silver_translated table created successfully")
        
        logger.info("Optimized table setup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to setup optimized table: {str(e)}")
        raise

def translate_reviews_ultra_fast(df):
    """Ultra-fast translation using parallel processing"""
    logger.info(f"🚀 Starting parallel translation for {len(df)} records...")
    start_time = time.time()
    
    df_result = df.copy()
    
    # Process negative reviews
    if 'negative_review' in df.columns:
        logger.info("⚡ Processing negative reviews with parallel translation...")
        
        excluded = ['nothing', 'negative feedback', '0', '', 'na', 'n/a']
        negative_texts = df['negative_review'].fillna('').astype(str)
        
        processed_texts = preprocess_texts(negative_texts.tolist())
        
        needs_translation = []
        for i, text in enumerate(negative_texts):
            if (isinstance(text, str) and 
                len(text.strip()) > 3 and 
                text.lower().strip() not in excluded):
                needs_translation.append((i, text))
        
        logger.info(f"   Found {len(needs_translation)} negative reviews to process")
        
        if needs_translation:
            indices, texts = zip(*needs_translation)
            batches = [texts[i:i+BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
            
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                if TQDM_AVAILABLE:
                    translated_batches = list(tqdm(
                        executor.map(translate_text_batch, batches),
                        desc="Translating negative reviews",
                        total=len(batches)
                    ))
                else:
                    translated_batches = list(executor.map(translate_text_batch, batches))
                    logger.info(f"   Processed {len(batches)} batches of negative reviews")
            
            translated_texts = [text for batch in translated_batches for text in batch]
            
            df_result['negative_review_translated'] = df['negative_review'].copy()
            for idx, translated in zip(indices, translated_texts):
                df_result.iloc[idx, df_result.columns.get_loc('negative_review_translated')] = translated
        else:
            df_result['negative_review_translated'] = df['negative_review']
    
    # Process positive reviews
    if 'positive_review' in df.columns:
        logger.info("⚡ Processing positive reviews with parallel translation...")
        
        excluded = ['nothing', 'positive review', '0', '', 'na', 'n/a']
        positive_texts = df['positive_review'].fillna('').astype(str)
        
        processed_texts = preprocess_texts(positive_texts.tolist())
        
        needs_translation = []
        for i, text in enumerate(positive_texts):
            if (isinstance(text, str) and 
                len(text.strip()) > 3 and 
                text.lower().strip() not in excluded):
                needs_translation.append((i, text))
        
        logger.info(f"   Found {len(needs_translation)} positive reviews to process")
        
        if needs_translation:
            indices, texts = zip(*needs_translation)
            batches = [texts[i:i+BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
            
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                if TQDM_AVAILABLE:
                    translated_batches = list(tqdm(
                        executor.map(translate_text_batch, batches),
                        desc="Translating positive reviews",
                        total=len(batches)
                    ))
                else:
                    translated_batches = list(executor.map(translate_text_batch, batches))
                    logger.info(f"   Processed {len(batches)} batches of positive reviews")
            
            translated_texts = [text for batch in translated_batches for text in batch]
            
            df_result['positive_review_translated'] = df['positive_review'].copy()
            for idx, translated in zip(indices, translated_texts):
                df_result.iloc[idx, df_result.columns.get_loc('positive_review_translated')] = translated
        else:
            df_result['positive_review_translated'] = df['positive_review']
    
    elapsed_time = time.time() - start_time
    logger.info(f"🎯 Parallel translation completed in {elapsed_time:.2f}s")
    logger.info(f"📈 Processing rate: {len(df)/elapsed_time:.2f} records/second")
    
    stats = perf_cache.get_stats()
    logger.info(f"🎯 Cache performance: {stats['hit_rate']:.1f}% hit rate, {stats['cache_size']} items cached")
    
    return df_result

def ingest_data_ultra_fast(df, engine):
    """Ultra-fast data ingestion with enhanced error handling"""
    logger.info(f"Starting ultra-fast ingestion of {len(df)} records...")
    start_time = time.time()
    
    try:
        # Define columns in correct order
        columns_order = [
            "city", "hotel_name", "reviewer_name", "reviewer_nationality",
            "duration", "check_in_date", "travel_type", "room_type",
            "review_date", "positive_review", "negative_review",
            "ingestion_timestamp", "sentiment_classification",
            "negative_review_translated", "positive_review_translated"
        ]
        
        # Filter to available columns
        available_columns = [col for col in columns_order if col in df.columns]
        df_filtered = df[available_columns].copy()
        
        logger.info(f"Filtered DataFrame to {len(available_columns)} columns for ingestion")
        
        # Truncate existing data
        logger.info("Truncating existing data from silver.silver_translated...")
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE silver.silver_translated"))
        logger.info("Existing data truncated")
        
        # Insert data in chunks using direct SQL like preprocessing script
        chunk_size = 2000
        total_chunks = (len(df_filtered) + chunk_size - 1) // chunk_size
        
        logger.info(f"Inserting data in {total_chunks} chunks...")
        
        with engine.begin() as conn:
            # Create the insert statement
            placeholders = ", ".join([f":{col}" for col in available_columns])
            insert_sql = f"""
            INSERT INTO silver.silver_translated ({", ".join(available_columns)})
            VALUES ({placeholders})
            """
            
            # Insert in chunks
            for i in range(0, len(df_filtered), chunk_size):
                chunk_num = i // chunk_size + 1
                chunk = df_filtered.iloc[i:i+chunk_size]
                
                # Convert chunk to records
                records = chunk.to_dict('records')
                
                # Insert the records
                for record in records:
                    conn.execute(text(insert_sql), record)
                
                if chunk_num % 5 == 0:
                    logger.info(f"   Inserted chunk {chunk_num}/{total_chunks}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Ultra-fast ingestion completed successfully!")
        logger.info(f"Inserted {len(df_filtered)} records in {elapsed_time:.2f} seconds")
        logger.info(f"Insertion rate: {len(df_filtered)/elapsed_time:.2f} records/second")
        
    except Exception as e:
        logger.error(f"Failed to ingest translated data: {str(e)}")
        raise

def main():
    """Main execution function - enhanced error handling"""
    logger.info("🚀 STARTING LOCAL TRANSLATION PIPELINE")
    logger.info("=" * 70)
    
    overall_start = time.time()
    
    try:
        # Database setup
        logger.info("Step 1: Creating optimized database connection...")
        engine = create_optimized_engine()
        ensure_table_exists_optimized(engine)
        logger.info("✓ Database connection and table setup completed")
        
        # Load data
        logger.info("Step 2: Loading data from source table...")
        df = read_data_optimized(engine)
        logger.info(f"✓ Loaded {len(df)} records from silver.reviews_cleaned")
        
        # Memory optimization
        logger.info("Step 3: Optimizing memory usage...")
        gc.collect()
        logger.info("✓ Memory optimization completed")
        
        # Translation
        logger.info("Step 4: Running parallel translation...")
        translated_df = translate_reviews_ultra_fast(df)
        logger.info(f"✓ Translation completed for {len(translated_df)} records")
        
        # Ingestion
        logger.info("Step 5: Ingesting translated data...")
        ingest_data_ultra_fast(translated_df, engine)
        logger.info("✓ Data ingestion completed")
        
        # Final statistics
        total_time = time.time() - overall_start
        overall_rate = len(df) / total_time if total_time > 0 else 0
        
        logger.info("🎉 TRANSLATION PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        logger.info(f"📊 Total Records: {len(df):,}")
        logger.info(f"⏱️  Total Time: {total_time:.2f} seconds")
        logger.info(f"🚀 Overall Rate: {overall_rate:.2f} records/second")
        logger.info(f"💾 Final Dataset Shape: {translated_df.shape}")
        
        final_stats = perf_cache.get_stats()
        logger.info(f"🎯 Final Cache Stats:")
        logger.info(f"   Hit Rate: {final_stats['hit_rate']:.1f}%")
        logger.info(f"   Total Items: {final_stats['cache_size']:,}")
        logger.info("=" * 70)
        
        return translated_df
        
    except Exception as e:
        logger.error(f"❌ TRANSLATION PIPELINE FAILED: {str(e)}")
        logger.error("=" * 70)
        raise
    finally:
        if 'engine' in locals():
            logger.info("Closing database connection...")
            engine.dispose()
            logger.info("✓ Database connection closed")
